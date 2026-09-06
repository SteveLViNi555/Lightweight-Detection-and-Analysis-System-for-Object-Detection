import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from core.dataset import VOCDataset
from core.model import SSD
from core.utils import decode_boxes, nms
from core.metrics import evaluate_voc
import yaml

CLASS_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels

def detect_on_dataset(model, dataloader, device, priors, conf_thresh=0.01, nms_thresh=0.45):
    model.eval()
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []
    all_gt_boxes, all_gt_labels = [], []

    with torch.no_grad():
        for images, boxes_gt, labels_gt in dataloader:
            images = images.to(device)
            loc_preds, conf_preds = model(images)

            for batch_idx in range(images.size(0)):
                loc = loc_preds[batch_idx]
                conf = conf_preds[batch_idx]
                boxes = decode_boxes(loc, priors, variance=[0.1, 0.2])
                boxes = boxes.clamp(0, 1)
                scores = torch.softmax(conf, dim=1)  # (num_priors, num_classes)

                # 逐类别 NMS（跳过背景类 0，标准 SSD 后处理）
                final_boxes, final_labels, final_scores = [], [], []
                for c in range(1, scores.size(1)):
                    cls_scores = scores[:, c]
                    mask = cls_scores > conf_thresh
                    if mask.sum() == 0:
                        continue
                    cls_boxes = boxes[mask]
                    cls_scores = cls_scores[mask]
                    keep_idx = nms(cls_boxes, cls_scores, overlap=nms_thresh)
                    final_boxes.append(cls_boxes[keep_idx])
                    final_labels.append(torch.full((len(keep_idx),), c, dtype=torch.long))
                    final_scores.append(cls_scores[keep_idx])

                if final_boxes:
                    boxes = torch.cat(final_boxes)
                    labels = torch.cat(final_labels)
                    scores_out = torch.cat(final_scores)
                else:
                    boxes = torch.zeros((0, 4))
                    labels = torch.zeros((0,), dtype=torch.long)
                    scores_out = torch.zeros((0,))

                if len(all_pred_boxes) < 5:
                    unique, counts = torch.unique(labels, return_counts=True)
                    print(f"图片 {len(all_pred_boxes)} 预测标签分布: {dict(zip(unique.tolist(), counts.tolist()))}")

                all_pred_boxes.append(boxes.cpu())
                all_pred_labels.append(labels.cpu())
                all_pred_scores.append(scores_out.cpu())
                all_gt_boxes.append(boxes_gt[batch_idx])
                all_gt_labels.append(labels_gt[batch_idx])

    return all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels

def plot_pr_curves(recalls_list, precisions_list, save_path='pr_curves.png'):
    plt.figure(figsize=(12, 8))
    has_curve = False
    for c in range(1, 21):
        if len(recalls_list[c-1]) > 1:
            plt.plot(recalls_list[c-1], precisions_list[c-1], label=CLASS_NAMES[c])
            has_curve = True
    if not has_curve:
        plt.text(0.5, 0.5, 'No valid PR curves (mAP=0)', ha='center', va='center', fontsize=20)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    if has_curve:
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"PR曲线已保存至 {save_path}")

def find_error_samples(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    error_samples = []
    for idx in range(len(gt_boxes)):
        if len(pred_boxes[idx]) == 0 and len(gt_boxes[idx]) > 0:
            error_samples.append(f"图片 {idx}: 漏检 (FN) - 有 {len(gt_boxes[idx])} 个目标未检出")
        elif len(pred_boxes[idx]) > len(gt_boxes[idx]) * 2 and len(gt_boxes[idx]) > 0:
            error_samples.append(f"图片 {idx}: 误报 (FP) - 预测了 {len(pred_boxes[idx])} 个框，实际只有 {len(gt_boxes[idx])} 个")
    return error_samples[:5]

if __name__ == '__main__':
    with open('config/default.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SSD(backbone=cfg['backbone'], num_classes=cfg['num_classes']).to(device)
    weights_path = 'runs/ssd_epoch_200.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"加载权重: {weights_path}")
    else:
        print("警告: 未找到权重文件，使用随机初始化")

    priors = model.priors.to(device)
    val_dataset = VOCDataset(root='data/VOCdevkit/VOC2007', image_set='val', transform=None)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=collate_fn)

    print("开始在验证集上推理...")
    pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels = detect_on_dataset(
        model, val_loader, device, priors, conf_thresh=0.01, nms_thresh=0.45
    )

    print("计算mAP指标...")
    mAP, aps, recalls_list, precisions_list = evaluate_voc(
        pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
        num_classes=cfg['num_classes'], iou_thresh=0.5
    )

    print(f"\n========== 评估结果 ==========")
    print(f"mAP@0.5: {mAP:.4f}")
    print("各类别 AP:")
    for i, ap in enumerate(aps):
        print(f"  {CLASS_NAMES[i+1]}: {ap:.4f}")

    plot_pr_curves(recalls_list, precisions_list)
    errors = find_error_samples(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    print("\n典型错误样本 (前5个):")
    for err in errors:
        print(f"  - {err}")

    with open('mAP_results.txt', 'w') as f:
        f.write(f"mAP@0.5: {mAP:.6f}\n")
        f.write("各类别 AP:\n")
        for i, ap in enumerate(aps):
            f.write(f"  {CLASS_NAMES[i+1]}: {ap:.6f}\n")

    with open('error_samples.txt', 'w') as f:
        f.write("典型错误样本 (前5个):\n")
        for err in errors:
            f.write(f"  {err}\n")
    print("评估结果已保存。")