from dataset.transform import *

from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import json
import os


def get_file_name(root, layout_txt):
    with open(os.path.join(root, layout_txt)) as layout_txt:
        file_name = layout_txt.read().split('\n')[:-1]
    return file_name


def xml2dict(xml):
    data = {c.tag: None for c in xml}
    for c in xml:
        def add(data, tag, text):
            if data[tag] is None:
                data[tag] = text
            elif isinstance(data[tag], list):
                data[tag].append(text)
            else:
                data[tag] = [data[tag], text]
            return data

        if len(c) == 0:
            data = add(data, c.tag, c.text)
        else:
            data = add(data, c.tag, xml2dict(c))
    return data


class VOC0712Dataset(Dataset):
    def __init__(self, root, class_path, transforms, mode, data_range=None, get_info=False):
        # label: xmin, ymin, xmax, ymax, class

        with open(class_path, 'r') as f:
            json_str = f.read()
            self.classes = json.loads(json_str)
        layout_txt = None
        if mode == 'train':
            root = [root[0], root[0], root[1], root[1]]
            layout_txt = [r'ImageSets\Main\train.txt', r'ImageSets\Main\val.txt',
                          r'ImageSets\Main\train.txt', r'ImageSets\Main\val.txt']
        elif mode == 'test':
            if not isinstance(root, list):
                root = [root]
            layout_txt = [r'ImageSets\Main\test.txt']
        assert layout_txt is not None, 'Unknown mode'

        self.transforms = transforms
        self.get_info = get_info

        self.image_list = []
        self.annotation_list = []
        for r, txt in zip(root, layout_txt):
            self.image_list += [os.path.join(r, 'JPEGImages', t + '.jpg') for t in get_file_name(r, txt)]
            self.annotation_list += [os.path.join(r, 'Annotations', t + '.xml') for t in get_file_name(r, txt)]

        if data_range is not None:
            self.image_list = self.image_list[data_range[0]: data_range[1]]
            self.annotation_list = self.annotation_list[data_range[0]: data_range[1]]

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        image_size = image.size
        label = self.label_process(self.annotation_list[idx])

        if self.transforms is not None:
            image, label = self.transforms(image, label)
        if self.get_info:
            return image, label, os.path.basename(self.image_list[idx]).split('.')[0], image_size
        else:
            return image, label

    def label_process(self, annotation):
        xml = ET.parse(os.path.join(annotation)).getroot()
        data = xml2dict(xml)['object']
        if isinstance(data, list):
            label = [[float(d['bndbox']['xmin']), float(d['bndbox']['ymin']),
                     float(d['bndbox']['xmax']), float(d['bndbox']['ymax']),
                     self.classes[d['name']]]
                     for d in data]
        else:
            label = [[float(data['bndbox']['xmin']), float(data['bndbox']['ymin']),
                     float(data['bndbox']['xmax']), float(data['bndbox']['ymax']),
                     self.classes[data['name']]]]
        label = np.array(label)
        return label


if __name__ == "__main__":
    from dataset.draw_bbox import draw

    root0712 = [r'F:\AI\Dataset\VOC2007\VOCdevkit\VOC2007', r'F:\AI\Dataset\VOC2012\VOCdevkit\VOC2012']

    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
        Resize(448)
    ])
    ds = VOC0712Dataset(root0712, 'classes.json', transforms, 'train', get_info=True)
    print(len(ds))
    for i, (image, label, image_name, image_size) in enumerate(ds):
        if i <= 1000:
            continue
        elif i >= 1010:
            break
        else:
            print(label.dtype)
            print(tuple(image.size()[1:]))
            draw(image, label, ds.classes)
    print('VOC2007Dataset')
