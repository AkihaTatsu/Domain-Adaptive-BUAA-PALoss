import xml.etree.ElementTree as ET
import os
import json
from datetime import datetime
import sys
import argparse
from tqdm import tqdm

image_id = 0
annotation_id = 0

image_list = []
instance_list = []
category_list = []
category_dict = {}

voc_category_dict = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}


class ImageInfo(object):
    def __init__(self):        
        self.file_name = None
        self.size = {'width': None, 'height': None}
        self.id = None


class InstanceInfo(object):
    def __init__(self):      
        self.area = None
        self.iscrowd = 0
        self.bbox = []
        self.category_name = None
        self.category_id = None
        self.ignore = 0
        self.segmentation = []
        self.image_id = None
        self.id = None


def xml_file_parse(path):
    '''解析单个XML文件'''
    tree = ET.parse(path)
    root = tree.getroot()
    img_info = ImageInfo()

    # 总标签检查
    if root.tag != 'annotation':
        raise Exception(
            'Error in file {}: Pascal voc xml root element should be annotation, rather than {}'.format(path, root.tag))

    # 图片名字
    img_info.file_name = root.findtext('filename')
    assert img_info.file_name is not None, 'Error in file {}: File name not in the file'.format(
        path)

    # 如果图片名字为数字则令其为id
    if img_info.file_name.split('.')[0].isdigit():
        img_info.id = int(img_info.file_name.split('.')[0])
    else:
        global image_id
        img_info.id = image_id
        image_id += 1

    # 提取图片size (width, height, depth)
    size_info = root.find('size')
    assert size_info is not None, 'Error in file {}: Size not in the file'.format(
        path)
    for size_elem in size_info:
        img_info.size[size_elem.tag] = int(size_elem.text)
    if img_info.size['width'] is None:
        raise Exception('Error in file {}: Width not found'.format(path))
    if img_info.size['height'] is None:
        raise Exception('Error in file {}: Height not found'.format(path))

    image_list.append(img_info)
    ######################
    # 提取图片object标注信息
    object_info = root.findall('object')
    if len(object_info) == 0:
        return -1

    for object in object_info:
        instance_info = InstanceInfo()

        # 类别名称
        instance_info.category_name = object.findtext('name')
        if instance_info.category_name is None:
            raise Exception(
                'Error in file {}: Structrue of xml broken at bbox tag: name'.format(path))
        if instance_info.category_name not in category_list:
            # 添加至类别列表
            category_list.append(instance_info.category_name)

        # 候选框
        bndbox_info = object.find('bndbox')
        bndbox = {
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None,
        }
        for box in bndbox_info:
            bndbox[box.tag] = int(box.text)

        if bndbox['xmin'] is None:
            raise Exception(
                'Error in file {}: Structrue of xml broken at bbox tag: xmin'.format(path))
        if bndbox['xmax'] is None:
            raise Exception(
                'Error in file {}: Structrue of xml broken at bbox tag: xmax'.format(path))
        if bndbox['ymin'] is None:
            raise Exception(
                'Error in file {}: Structrue of xml broken at bbox tag: ymin'.format(path))
        if bndbox['ymax'] is None:
            raise Exception(
                'Error in file {}: Structrue of xml broken at bbox tag: ymax'.format(path))

        instance_info.bbox = [
            bndbox['xmin'],
            bndbox['ymin'],
            bndbox['xmax'] - bndbox['xmin'],
            bndbox['ymax'] - bndbox['ymin'],
        ]

        # 面积与ID
        instance_info.area = instance_info.bbox[2] * instance_info.bbox[3]
        instance_info.image_id = img_info.id
        global annotation_id
        annotation_id += 1
        instance_info.id = annotation_id

        instance_list.append(instance_info)

def all_xml_parse(data_dir, type):
    '''解析所有XML文件'''
    assert os.path.exists(data_dir), "Data path:{} does not exist".format(data_dir)
    labelfile = type + ".txt"
    image_sets_file = os.path.join(data_dir, "ImageSets", "Main", labelfile)
    xml_files_list = []

    if os.path.isfile(image_sets_file):
        ids = []
        with open(image_sets_file, 'r') as f:
            for line in f.readlines():
                ids.append(line.strip())
        xml_files_list = [os.path.join(data_dir, "Annotations", f"{i}.xml") for i in ids]
    elif os.path.isdir(data_dir):
        # 修改此处xml的路径即可
        xml_dir = os.path.join(data_dir, "labels/voc")
        xml_list = os.listdir(xml_dir)
        xml_files_list = [os.path.join(xml_dir, i) for i in xml_list]

    print("Parsing xmls...")
    for xml_file in tqdm(xml_files_list, ascii=True):
        try:
            xml_file_parse(xml_file)
        except Exception as e:
            tqdm.write(str(e))
        
def category_id_generate():
    # '''生成排序后的类别序号字典，并将类别序号添加到实例类中'''
    # category_list.sort()
    # for i in range(len(category_list)):
    #     category_dict[category_list[i]] = i
    #     
    # for instance in instance_list:
    #     instance.category_id = category_dict[instance.category_name]
    '''按照voc2007的标准将类别序号添加到实例类中'''
    for instance in instance_list:
        instance.category_id = voc_category_dict[instance.category_name]

def generate_json(json_path):
    '''生成json文件'''
    coco = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': [],
    }

    # 添加图像信息
    print("Building images...")
    for image in tqdm(image_list, ascii=True):
        image_item = {
            'file_name': image.file_name,
            'height': image.size['height'],
            'width': image.size['width'],
            'id': image.id,
        }
        coco['images'].append(image_item)
    
    # 添加实例信息
    print("Building instances...")
    for instance in tqdm(instance_list, ascii=True):
        instance_item = {
            'area': instance.area,
            'iscrowd': instance.iscrowd,
            'bbox': instance.bbox,
            'category_id': instance.category_id,
            'ignore': instance.ignore,
            'segmentation': instance.segmentation,
            'image_id': instance.image_id,
            'id': instance.id,
        }
        coco['annotations'].append(instance_item)
    
    # 添加类别信息
    print("Building categories...")
    for category in tqdm(category_list, ascii=True):
        category_item = {
            'supercategory': "None",
            # 'id': category_dict[category],
            'id': voc_category_dict[category],
            'name': category
        }
        coco['categories'].append(category_item)
    
    json.dump(coco, open(json_path, 'w'), indent=4, separators=(',', ': '))

    print("class nums:{}".format(len(coco['categories'])))
    print("image nums:{}".format(len(coco['images'])))
    print("instance nums:{}".format(len(coco['annotations'])))


def arg_parse():
    """
    脚本说明：
        本脚本用于将VOC格式的标注文件.xml转换为coco格式的标注文件.json
    参数说明：
        voc_data_dir:两种格式
            1.voc2012文件夹的路径，会自动找到voc2012/imageSets/Main/xx.txt
            2.xml标签文件存放的文件夹
        json_save_path:json文件输出的文件夹
        type:主要用于voc2012查找xx.txt,如train.txt.如果用格式2，则不会用到该参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--voc-dir', type=str,
                        default='data/label/voc', help='voc path')
    parser.add_argument('-s', '--save-path', type=str,
                        default='data/convert/coco/train.json', help='json save path')
    parser.add_argument('-t', '--type', type=str, default='trainval',
                        help='only use in voc2012/2007; select file type')
    return parser.parse_args()


def main():
    args = arg_parse()
    all_xml_parse(args.voc_dir, args.type)
    category_id_generate()
    generate_json(args.save_path)


if __name__ == '__main__':
    main()
