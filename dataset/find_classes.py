from dataset.data import xml2dict

import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
import os


json_path = './classes.json'

root = r'F:\AI\Dataset\VOC2012\VOCdevkit\VOC2012'
annotation_root = os.path.join(root, 'Annotations')
annotation_list = os.listdir(annotation_root)
annotation_list = [os.path.join(annotation_root, a) for a in annotation_list]

s = set()
for annotation in tqdm(annotation_list):
    xml = ET.parse(os.path.join(annotation)).getroot()
    data = xml2dict(xml)['object']
    if isinstance(data, list):
        for d in data:
            s.add(d['name'])
    else:
        s.add(data['name'])

s = list(s)
s.sort()
data = {value: i for i, value in enumerate(s)}
json_str = json.dumps(data)

with open(json_path, 'w') as f:
    f.write(json_str)
