import torch
import torchvision
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class ToTensor:
    def __init__(self):
        self.totensor = torchvision.transforms.ToTensor()

    def __call__(self, image, label):
        image = self.totensor(image)
        label = torch.tensor(label)
        return image, label


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        """
        :param label: xmin, ymin, xmax, ymax
        如果图片被水平翻转,那么label的xmin与xmax会互换，变成 xmax, ymin, xmin, ymax
        由于YOLO的输出是(center_x, center_y, w, h) ,因此label的xmin与xmax换位不会影响损失计算与训练
        但是需要注意w,h计算时使用abs
        """
        if random.random() < self.p:
            height, width = image.shape[-2:]
            image = image.flip(-1)      # 水平翻转
            bbox = label[:, :4]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [0, 2]]
            label[:, :4] = bbox
        return image, label


class Resize:
    def __init__(self, image_size, keep_ratio=True):
        """
        :param image_size: int
        """
        self.image_size = image_size
        self.keep_ratio = keep_ratio

    def __call__(self, image, label):
        """
        :param in_image: tensor [3, h, w]
        :param label: xmin, ymin, xmax, ymax
        :return:
        """
        # 将所有图片左上角对齐构成448*448tensor的Transform

        h, w = tuple(image.size()[1:])
        label[:, [0, 2]] = label[:, [0, 2]] / w
        label[:, [1, 3]] = label[:, [1, 3]] / h

        if self.keep_ratio:
            r_h = min(self.image_size / h, self.image_size / w)
            r_w = r_h
        else:
            r_h = self.image_size / h
            r_w = self.image_size / w

        h, w = int(r_h * h), int(r_w * w)
        h, w = min(h, self.image_size), min(w, self.image_size)
        label[:, [0, 2]] = label[:, [0, 2]] * w
        label[:, [1, 3]] = label[:, [1, 3]] * h

        T = torchvision.transforms.Resize([h, w])

        Padding = torch.nn.ZeroPad2d((0, self.image_size - w, 0, self.image_size - h))
        image = Padding(T(image))

        assert list(image.size()) == [3, self.image_size, self.image_size]
        return image, label
