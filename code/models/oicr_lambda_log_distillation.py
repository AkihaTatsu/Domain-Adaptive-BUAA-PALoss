import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.config import cfg

from models.vgg16 import VGG16Backbone

import logging

logger = logging.getLogger(__name__)

from pdb import set_trace as pause

import utils.vgg_weights_helper as vgg_utils

from layers.refinement.oicr import OICR         as Refinement
from layers.losses.oicr_losses import OICRLosses   as Losses
from layers.losses.mil_loss import mil_loss
from layers.refinement_agents import RefinementAgents
from layers.distillation import Distillation
from layers.mil import MIL

from layers.roi_pooling.roi_pool import RoiPoolLayer
from layers.adaptative_supervision_functions import get_adaptative_lambda

# Domain Adaptive part
from da_model.da_faster_rcnn.DA import _InstanceDA
from da_model.da_faster_rcnn.DA import _ImageDA


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.backbone = VGG16Backbone()

        self.box_features = RoiPoolLayer(self.backbone.dim_out, self.backbone.spatial_scale)
        self.mil = MIL(self.box_features.dim_out, cfg.MODEL.NUM_CLASSES)
        self.refinement_agents = RefinementAgents(self.box_features.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.Refine_Losses = [Losses() for i in range(cfg.REFINE_TIMES)]

        self.inner_iter = 0
        self._init_modules()

        '''Loss of domain adaptation'''
        self.dout_base_model = 512  # For vgg16, the dim is 512
        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()
        self.consistency_loss = torch.nn.MSELoss(size_average=False)

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def set_inner_iter(self, inner_iter):
        self.inner_iter = inner_iter

    def forward(self, data, rois, labels, t_data, t_rois, t_labels):

        with torch.set_grad_enabled(self.training):

            backbone_feat = self.backbone(data)
            box_feat = self.box_features(backbone_feat, rois)
            mil_score = self.mil(box_feat)
            refine_score = self.refinement_agents(box_feat)

            im_cls_score = mil_score.sum(dim=0, keepdim=True)

            return_dict = {}  # Return value
            if self.training:

                return_dict['losses'] = {}

                # image classification loss
                loss_im_cls = mil_loss(im_cls_score, labels)
                imloss = loss_im_cls.detach()

                # refinement loss
                boxes = rois[:, 1:]
                im_labels = labels

                plot_dict = {}

                lambda_gt = 0.5
                lambda_ign = 0.1
                # lambda_gt, lambda_ign  = get_adaptative_lambda(self.inner_iter)
                return_dict['delta'] = lambda_gt
                for i_refine, refine in enumerate(refine_score):
                    if i_refine == 0:
                        refinement_output = Refinement(boxes, mil_score, im_labels, refine, lambda_gt=lambda_gt,
                                                       lambda_ign=lambda_ign)
                    else:
                        refinement_output = Refinement(boxes, refine_score[i_refine - 1],
                                                       im_labels, refine, lambda_gt=lambda_gt, lambda_ign=lambda_ign)

                    refine_loss = self.Refine_Losses[i_refine](refine,
                                                               refinement_output['labels'],
                                                               refinement_output['cls_loss_weights'],
                                                               refinement_output['gt_assignment'],
                                                               refinement_output['im_labels_real'])

                    refine_loss = (1.0 + torch.exp(-imloss).detach()) * refine_loss
                    return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()

                return_dict['losses']['loss_im_cls'] = loss_im_cls

                # pytorch0.4 bug on gathering scalar(0-dim) tensors
                for k, v in return_dict['losses'].items():
                    return_dict['losses'][k] = v.unsqueeze(0)
            else:
                final_scores = refine_score[0]
                for i in range(1, cfg.REFINE_TIMES):
                    final_scores += refine_score[i]

                final_scores /= cfg.REFINE_TIMES

                return_dict['final_scores'] = final_scores

            '''Domain adaptation loss calc'''
            ''' ============= Calc feat and pooled feat ============= '''
            t_backbone_feat = self.backbone(t_data)
            t_box_feat = self.box_features(t_backbone_feat, t_rois)

            ''' ============= Domain Adaptation Loss ============= '''
            DA_img_loss_cls = 0
            DA_ins_loss_cls = 0

            t_DA_img_loss_cls = 0
            t_DA_ins_loss_cls = 0

            DA_cst_loss = 0
            t_DA_cst_loss = 0
            
            need_backprop = torch.ones(1).long()
            t_need_backprop = torch.zeros(1).long()
            assert need_backprop.detach() == 1 and t_need_backprop.detach() == 0

            # Source image domain adaptation
            base_score, base_label = self.RCNN_imageDA(backbone_feat, need_backprop)

            base_prob = F.log_softmax(base_score, dim=1)
            DA_img_loss_cls = F.nll_loss(base_prob, base_label)

            instance_sigmoid, same_size_label = self.RCNN_instanceDA(box_feat, need_backprop)
            instance_loss = nn.BCELoss()
            DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

            # consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
            consistency_prob = F.softmax(base_score, dim=1)[:, 1, :, :]
            consistency_prob = torch.mean(consistency_prob)
            consistency_prob = consistency_prob.repeat(instance_sigmoid.size())

            DA_cst_loss = self.consistency_loss(
                instance_sigmoid, consistency_prob.detach())

            # Target image domain adaptation
            t_base_score, t_base_label = self.RCNN_imageDA(t_backbone_feat, t_need_backprop)

            # Image DA
            t_base_prob = F.log_softmax(t_base_score, dim=1)
            t_DA_img_loss_cls = F.nll_loss(t_base_prob, t_base_label)

            t_instance_sigmoid, t_same_size_label = self.RCNN_instanceDA(t_box_feat, t_need_backprop)
            t_instance_loss = nn.BCELoss()

            t_DA_ins_loss_cls = t_instance_loss(t_instance_sigmoid, t_same_size_label)

            t_consistency_prob = F.softmax(t_base_score, dim=1)[:, 0, :, :]
            t_consistency_prob = torch.mean(t_consistency_prob)
            t_consistency_prob = t_consistency_prob.repeat(
                t_instance_sigmoid.size())

            t_DA_cst_loss = self.consistency_loss(
                t_instance_sigmoid, t_consistency_prob.detach())

            return_dict['DA_losses'] = {
                'DA_img_loss_cls': DA_img_loss_cls,
                'DA_ins_loss_cls': DA_ins_loss_cls,
                'DA_cst_loss': DA_cst_loss,
                't_DA_img_loss_cls': t_DA_img_loss_cls,
                't_DA_ins_loss_cls': t_DA_ins_loss_cls,
                't_DA_cst_loss': t_DA_cst_loss,
            }
            
            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['DA_losses'].items():
                return_dict['DA_losses'][k] = v.unsqueeze(0)

            return return_dict

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron


def loot_model(args):
    print("Using model description:", args.model)
    model = DetectionModel()
    return model


