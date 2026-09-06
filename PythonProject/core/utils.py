import torch
import numpy as np

def generate_priors(feature_maps, image_size, strides, min_sizes, max_sizes, aspect_ratios):
    """生成多尺度默认框 (cx, cy, w, h) 归一化"""
    priors = []
    anchors_per_fm = []
    for k, f in enumerate(feature_maps):
        count = 0
        for i in range(f):
            for j in range(f):
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                # 基础尺度
                size = min_sizes[k] / image_size
                priors.append([cx, cy, size, size])
                count += 1

                size = np.sqrt(min_sizes[k] * max_sizes[k]) / image_size
                priors.append([cx, cy, size, size])
                count += 1

                # 额外小尺度 (0.5 * min_size)，提升小目标覆盖
                size = min_sizes[k] * 0.5 / image_size
                priors.append([cx, cy, size, size])
                count += 1

                # 宽高比 anchor
                for ratio in aspect_ratios[k]:
                    w = min_sizes[k] / image_size * np.sqrt(ratio)
                    h = min_sizes[k] / image_size / np.sqrt(ratio)
                    priors.append([cx, cy, w, h])
                    priors.append([cx, cy, h, w])
                    count += 2

        anchors_per_fm.append(count // (f * f))
    return torch.tensor(priors, dtype=torch.float32), anchors_per_fm

def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                      boxes[:, :2] + boxes[:, 2:]/2), 1)

def center_size(boxes):
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,
                      boxes[:, 2:] - boxes[:, :2]), 1)

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union

def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = []
    if boxes.numel() == 0:
        return torch.LongTensor(keep)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes = boxes[idx]
    while boxes.numel() > 0:
        keep.append(idx[0])
        if boxes.numel() == 1:
            break
        iou = jaccard(boxes[0:1, :], boxes[1:, :]).squeeze(0)
        idx = idx[1:][iou < overlap]
        boxes = boxes[1:][iou < overlap]
    return torch.LongTensor(keep)

def decode_boxes(loc, priors, variance=[0.1, 0.2]):
    boxes = torch.cat([
        priors[:, :2] + loc[:, :2] * variance[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variance[1])
    ], 1)
    boxes = torch.cat([
        boxes[:, :2] - boxes[:, 2:] / 2,
        boxes[:, :2] + boxes[:, 2:] / 2
    ], 1)
    return boxes