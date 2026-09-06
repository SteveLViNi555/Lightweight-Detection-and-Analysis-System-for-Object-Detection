import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random

class VOCDataset(Dataset):
    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_to_idx = {cls: i+1 for i, cls in enumerate(self.classes)}
        with open(os.path.join(self.root, f'ImageSets/Main/{image_set}.txt')) as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, f'JPEGImages/{img_id}.jpg')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        xml_path = os.path.join(self.root, f'Annotations/{img_id}.xml')
        boxes, labels = self._parse_annotation(xml_path, w, h)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        image = cv2.resize(image, (300, 300))

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        return image, boxes, labels

    def _parse_annotation(self, xml_path, img_w, img_h):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes:
                continue
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / img_w
            ymin = float(bbox.find('ymin').text) / img_h
            xmax = float(bbox.find('xmax').text) / img_w
            ymax = float(bbox.find('ymax').text) / img_h
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])
        return boxes, labels

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, labels):
        if random.random() < self.prob:
            _, w, _ = image.shape
            image = cv2.flip(image, 1)
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        return image, boxes, labels

class RandomBrightness:
    def __init__(self, delta=32):
        self.delta = delta

    def __call__(self, image, boxes, labels):
        if random.random() < 0.5:
            delta = random.uniform(-self.delta, self.delta)
            image = cv2.add(image, delta)
        return image, boxes, labels