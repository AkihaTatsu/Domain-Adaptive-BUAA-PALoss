# Domain Adaptive Fast R-CNN with Bottom-Up Aggregated Attention and Phase-Aware Loss

This model is based on the work of [BUAA & PALoss](https://github.com/Horatio9702/BUAA_PALoss), [Domain Adaptive Faster R-CNN](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch) etc.
## Implementation
### Environment
```python
pip install -r requirements.txt
```

### Dataset
Put your dataset in `data` folder, and add dataset info in:
+ `code/datasets/dataset_catalog.py`
  + `IM_DIR` refers to the path of folder containing all images.
  + `ANN_FN` refers to the path of annotation file (in this case, a json).
+ `code/tasks/train.py` - `dataset_mapping`
  + add the category num for your training part of dataset.
+ `code/tasks/test.py` - `dataset_mapping`
  + add the category num for your testing part of dataset.

**Pay attention:** Your dataset is required to be in COCO-format; that is, the annotation must be a COCO json format. If you use VOC-format dataset, please switch it to COCO format using `voc_to_coco.py`. Also pay attention that image name in COCO json file must contain suffix (like .jpg), please add it if it is missed.

### Selective Search Proposal Files
SS proposal files (.pkl) can be generated in many ways (we provide `selective_search_vertex.py` to generate one using opencv package), but ensure the proposal coordinates are the coordinates of the top-left and bottom-right vertices. If not, use `ss_pkl_change.py`.

Generated files is suggested to be put in `data/selective_search_data` folder.

### Backbone network
Put it in `data/pretrained_model` folder.

### Config
Add your yaml config file in `configs/baselines`.

## Training
```python
python code/tasks/train.py [--args]
```
Detailed info using `python code/tasks/train.py -h` or refer to `train.sh`.

## Testing
Results of training are stored in `snapshots/oicr_lambda_log_distillation/(TIME)/ckpt` folder. `bkp.pth` is the final result.
```python
python code/tasks/test.py [--args]
```
Detailed info using `python code/tasks/test.py -h` or refer to `test.sh`.
