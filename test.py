from dataset.data import VOC0712Dataset, Compose, ToTensor, Resize
from dataset.draw_bbox import draw
from model.yolo import yolo, output_process
from voc_eval import voc_eval

import torch

import pandas as pd
from tqdm import tqdm
import json


class CFG:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root = r'F:\AI\Dataset\VOC2007\VOCdevkit\VOC2007'
    class_path = r'dataset/classes.json'
    model_path = r'log/ex7/yolo.pth'
    detpath = r'det\{}.txt'
    annopath = r'F:\AI\Dataset\VOC2007\VOCdevkit\VOC2007\Annotations\{}.xml'
    imagesetfile = r'F:\AI\Dataset\VOC2007\VOCdevkit\VOC2007\ImageSets\Main\test.txt'
    classname = None

    test_range = None
    show_image = False
    get_ap = True

    backbone = 'resnet'
    S = 7
    B = 2
    image_size = 448

    get_info = True
    transforms = Compose([
        ToTensor(),
        Resize(448, keep_ratio=False)
    ])

    num_classes = 0

    conf_th = 0.2
    iou_th = 0.5


def test():
    device = torch.device(CFG.device)
    print('Test:\nDevice:{}'.format(device))

    dataset = VOC0712Dataset(CFG.root, CFG.class_path, CFG.transforms, 'test',
                             data_range=CFG.test_range, get_info=CFG.get_info)

    with open(CFG.class_path, 'r') as f:
        json_str = f.read()
        classes = json.loads(json_str)
        CFG.classname = list(classes.keys())
        CFG.num_classes = len(CFG.classname)

    yolo_net = yolo(s=CFG.S, cell_out_ch=CFG.B * 5 + CFG.num_classes, backbone_name=CFG.backbone)
    yolo_net.to(device)

    yolo_net.load_state_dict(torch.load(CFG.model_path))

    bboxes = []
    with torch.no_grad():
        for image, label, image_name, input_size in tqdm(dataset):
            image = image.unsqueeze(dim=0)
            image = image.to(device)

            output = yolo_net(image)
            output = output_process(output.cpu(), CFG.image_size, CFG.S, CFG.B, CFG.conf_th, CFG.iou_th)

            if CFG.show_image:
                draw(image.squeeze(dim=0), output.squeeze(dim=0), classes, show_conf=True)
                draw(image.squeeze(dim=0), label, classes, show_conf=True)

            # 还原
            output[:, :, [0, 2]] = output[:, :, [0, 2]] * input_size[0] / CFG.image_size
            output[:, :, [1, 3]] = output[:, :, [1, 3]] * input_size[1] / CFG.image_size

            output = output.squeeze(dim=0).numpy().tolist()
            if len(output) > 0:
                pred = [[image_name, output[i][-3] * output[i][-2]] + output[i][:4] + [int(output[i][-1])]
                        for i in range(len(output))]
                bboxes += pred
    det_list = [[] for _ in range(CFG.num_classes)]
    for b in bboxes:
        det_list[b[-1]].append(b[:-1])

    if CFG.get_ap:
        map = 0
        for idx in range(CFG.num_classes):
            file_path = CFG.detpath.format(CFG.classname[idx])
            txt = '\n'.join([' '.join([str(i) for i in item]) for item in det_list[idx]])
            with open(file_path, 'w') as f:
                f.write(txt)

            rec, prec, ap = voc_eval(CFG.detpath, CFG.annopath, CFG.imagesetfile, CFG.classname[idx])
            print(rec)
            print(prec)
            map += ap
            print(ap)

        map /= CFG.num_classes
        print('mAP', map)


if __name__ == '__main__':
    test()
