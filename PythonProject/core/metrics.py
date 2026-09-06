import torch
import numpy as np
from .utils import jaccard

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_voc(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, num_classes=21, iou_thresh=0.5):
    aps = []
    all_recalls = []
    all_precisions = []
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

    for c in range(1, num_classes):
        gt_class_boxes = []
        for img_idx in range(len(gt_boxes)):
            for box, label in zip(gt_boxes[img_idx], gt_labels[img_idx]):
                if label == c:
                    gt_class_boxes.append([img_idx] + box.tolist())

        pred_class_data = []
        for img_idx in range(len(pred_boxes)):
            for box, label, score in zip(pred_boxes[img_idx], pred_labels[img_idx], pred_scores[img_idx]):
                if label == c:
                    pred_class_data.append([img_idx, score] + box.tolist())

        if len(gt_class_boxes) == 0:
            aps.append(0.0)
            all_recalls.append(np.array([0]))
            all_precisions.append(np.array([1]))
            continue

        pred_class_data.sort(key=lambda x: x[1], reverse=True)
        gt_detected = [False] * len(gt_class_boxes)
        tp = np.zeros(len(pred_class_data))
        fp = np.zeros(len(pred_class_data))

        for i, pred in enumerate(pred_class_data):
            img_idx = pred[0]
            pred_box = pred[2:]
            gt_in_img = [box for box in gt_class_boxes if box[0] == img_idx]
            if not gt_in_img:
                fp[i] = 1
                continue
            ious = jaccard(torch.tensor([pred_box]), torch.tensor([box[1:] for box in gt_in_img]))
            max_iou, max_idx = ious.max(1)
            gt_global_idx = gt_class_boxes.index(gt_in_img[max_idx.item()])
            if max_iou >= iou_thresh and not gt_detected[gt_global_idx]:
                tp[i] = 1
                gt_detected[gt_global_idx] = True
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / (len(gt_class_boxes) + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        all_recalls.append(recalls)
        all_precisions.append(precisions)

    mAP = np.mean(aps)
    return mAP, aps, all_recalls, all_precisions