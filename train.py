from dataset.data import VOC0712Dataset
from dataset.transform import *
from model.yolo import yolo, yolo_loss
from scheduler import Scheduler

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import pandas as pd
import json
import os

import warnings


class CFG:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root0712 = [r'F:\AI\Dataset\VOC2007\VOCdevkit\VOC2007', r'F:\AI\Dataset\VOC2012\VOCdevkit\VOC2012']
    class_path = r'./dataset/classes.json'
    model_root = r'./log/ex7'
    model_path = None
    
    backbone = 'resnet'
    pretrain = 'model/resnet50-19c8e357.pth'
    with_amp = True
    S = 7
    B = 2
    image_size = 448

    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
        Resize(448, keep_ratio=False)
    ])

    start_epoch = 0
    epoch = 135
    batch_size = 16
    num_workers = 2
	
    freeze_backbone_till = 30

    scheduler_params = {
        'lr_start': 1e-3 / 4,
        'step_warm_ep': 10,
        'step_1_lr': 1e-2 / 4,
        'step_1_ep': 75,
        'step_2_lr': 1e-3 / 4,
        'step_2_ep': 40,
        'step_3_lr': 1e-4 / 4,
        'step_3_ep': 10
    }

    momentum = 0.9
    weight_decay = 0.0005


def collate_fn(batch):
    return tuple(zip(*batch))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train():
    device = torch.device(CFG.device)
    print('Train:\nDevice:{}'.format(device))

    with open(CFG.class_path, 'r') as f:
        json_str = f.read()
        classes = json.loads(json_str)
        CFG.num_classes = len(classes)

    train_ds = VOC0712Dataset(CFG.root0712, CFG.class_path, CFG.transforms, 'train')
    test_ds = VOC0712Dataset(CFG.root0712, CFG.class_path, CFG.transforms, 'test')

    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, collate_fn=collate_fn)

    yolo_net = yolo(s=CFG.S, cell_out_ch=CFG.B * 5 + CFG.num_classes, backbone_name=CFG.backbone, pretrain=CFG.pretrain)
    yolo_net.to(device)

    if CFG.model_path is not None:
        yolo_net.load_state_dict(torch.load(CFG.model_path))

    if CFG.freeze_backbone_till != -1:
        print('Freeze Backbone')
        for param in yolo_net.backbone.parameters():
            param.requires_grad_(False)

    param = [p for p in yolo_net.parameters() if p.requires_grad]
    optimizer = optim.SGD(param, lr=CFG.scheduler_params['lr_start'],
                          momentum=CFG.momentum, weight_decay=CFG.weight_decay)
    criterion = yolo_loss(CFG.device, CFG.S, CFG.B, CFG.image_size, len(train_ds.classes))
    scheduler = Scheduler(optimizer, **CFG.scheduler_params)
    scaler = GradScaler()

    for _ in range(CFG.start_epoch):
        scheduler.step()

    best_train_loss = 1e+9
    train_losses = []
    test_losses = []
    lrs = []
    for epoch in range(CFG.start_epoch, CFG.epoch):
        if CFG.freeze_backbone_till != -1 and epoch >= CFG.freeze_backbone_till:
            print('Unfreeze Backbone')
            for param in yolo_net.backbone.parameters():
                param.requires_grad_(True)
            CFG.freeze_backbone_till = -1
        # Train
        yolo_net.train()
        loss_score = AverageMeter()
        dl = tqdm(train_dl, total=len(train_dl))
        for images, labels in dl:
            batch_size = len(labels)

            images = torch.stack(images)
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            optimizer.zero_grad()

            if CFG.with_amp:
                with autocast():
                    outputs = yolo_net(images)
                    loss, xy_loss, wh_loss, conf_loss, class_loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = yolo_net(images)
                loss, xy_loss, wh_loss, conf_loss, class_loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            loss_score.update(loss.detach().item(), batch_size)
            dl.set_postfix(Mode='Train', AvgLoss=loss_score.avg, Loss=loss.detach().item(),
                           Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        train_losses.append(loss_score.avg)
        print('Train Loss: {:.4f}'.format(loss_score.avg))

        if best_train_loss > loss_score.avg:
            print('Save yolo_net to {}'.format(os.path.join(CFG.model_root, 'yolo.pth')))
            torch.save(yolo_net.state_dict(), os.path.join(CFG.model_root, 'yolo.pth'))
            best_train_loss = loss_score.avg

        loss_score.reset()
        with torch.no_grad():
            # Test
            yolo_net.eval()
            dl = tqdm(test_dl, total=len(test_dl))
            for images, labels in dl:
                batch_size = len(labels)

                images = torch.stack(images)
                images = images.to(device)
                labels = [label.to(device) for label in labels]

                outputs = yolo_net(images)
                loss, xy_loss, wh_loss, conf_loss, class_loss = criterion(outputs, labels)

                loss_score.update(loss.detach().item(), batch_size)
                dl.set_postfix(Mode='Test', AvgLoss=loss_score.avg, Loss=loss.detach().item(), Epoch=epoch)
        test_losses.append(loss_score.avg)
        print('Test Loss: {:.4f}'.format(loss_score.avg))

        df = pd.DataFrame({'Train Loss': train_losses, 'Test Loss': test_losses, 'LR': lrs})
        df.to_csv(os.path.join(CFG.model_root, 'result.csv'), index=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train()
