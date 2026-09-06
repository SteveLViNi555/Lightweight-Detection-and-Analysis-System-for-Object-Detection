"""诊断脚本：检查模型原始输出，定位全背景预测的根因"""
import torch
import yaml
import cv2
import numpy as np
from core.model import SSD
from core.utils import decode_boxes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config/default.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

model = SSD(backbone=cfg['backbone'], num_classes=cfg['num_classes']).to(device)
model.load_state_dict(torch.load('runs/ssd_epoch_150.pth', map_location=device))
model.eval()

priors = model.priors.to(device)

# 读取一张验证集图片
img = cv2.imread('data/VOCdevkit/VOC2007/JPEGImages/000001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (300, 300))
img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_t = (img_t - mean) / std
img_t = img_t.unsqueeze(0).to(device)

with torch.no_grad():
    loc, conf = model(img_t)

print(f"loc shape: {loc.shape}")
print(f"conf shape: {conf.shape}")

conf = conf[0]  # (num_priors, num_classes)
loc = loc[0]    # (num_priors, 4)

print(f"\n===== 原始 conf logits 统计 =====")
print(f"  最小值: {conf.min().item():.4f}")
print(f"  最大值: {conf.max().item():.4f}")
print(f"  均值:   {conf.mean().item():.4f}")

# 每个类别的平均 logit
print(f"\n===== 各类别平均 logit =====")
for c in range(21):
    avg = conf[:, c].mean().item()
    max_v = conf[:, c].max().item()
    print(f"  class {c:2d}: mean={avg:+.4f}, max={max_v:+.4f}")

# softmax 后的分数
scores = torch.softmax(conf, dim=1)  # (num_priors, 21)
print(f"\n===== Softmax 后分数统计 =====")
for c in range(21):
    avg = scores[:, c].mean().item()
    max_v = scores[:, c].max().item()
    print(f"  class {c:2d}: mean={avg:.6f}, max={max_v:.6f}")

# 看看有多少 anchor 的最大概率类别不是背景
max_scores, labels = scores.max(dim=1)
non_bg = labels != 0
print(f"\n===== 预测结果 =====")
print(f"  总 anchor 数: {labels.shape[0]}")
print(f"  预测为背景: {(labels == 0).sum().item()}")
print(f"  预测为非背景: {non_bg.sum().item()}")

if non_bg.sum() > 0:
    print(f"\n  非背景 anchor 的类别分布:")
    unique, counts = torch.unique(labels[non_bg], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    class {u.item()}: {c.item()} 个")
    print(f"\n  非背景 anchor 的最高 softmax 分数:")
    print(f"    min={max_scores[non_bg].min().item():.6f}")
    print(f"    max={max_scores[non_bg].max().item():.6f}")
    print(f"    mean={max_scores[non_bg].mean().item():.6f}")

# 检查正样本匹配情况
from core.dataset import VOCDataset
import xml.etree.ElementTree as ET
from core.utils import jaccard, point_form, center_size

dataset = VOCDataset(root='data/VOCdevkit/VOC2007', image_set='val')
_, gt_boxes, gt_labels = dataset[0]
gt_boxes = gt_boxes.to(device)
gt_labels = gt_labels.to(device)

print(f"\n===== 图片 000001 GT 信息 =====")
print(f"  GT boxes: {gt_boxes.shape[0]}")
for b, l in zip(gt_boxes, gt_labels):
    print(f"    class {l.item()}: [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]")

# 模拟 loss 中的锚框匹配
overlaps = jaccard(priors, point_form(gt_boxes))
best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

pos_mask = best_prior_overlap.squeeze(1) >= 0.5
print(f"\n===== 锚框匹配结果 =====")
print(f"  IoU >= 0.5 的正样本 anchor 数: {pos_mask.sum().item()}")
print(f"  最大 IoU per GT: {best_truth_overlap.squeeze(0).tolist()}")

# 检查正样本 anchor 的 logit
if pos_mask.sum() > 0:
    pos_logits = conf[pos_mask]
    print(f"\n===== 正样本 anchor 的 logit =====")
    print(f"  数量: {pos_mask.sum().item()}")
    print(f"  background (class 0) logit: mean={pos_logits[:, 0].mean().item():.4f}, max={pos_logits[:, 0].max().item():.4f}")
    for c in range(1, 21):
        if (pos_logits[:, c].max() > -5).item():
            print(f"  class {c:2d} logit: mean={pos_logits[:, c].mean().item():.4f}, max={pos_logits[:, c].max().item():.4f}")

    pos_scores = torch.softmax(pos_logits, dim=1)
    pos_max_scores, pos_labels = pos_scores.max(dim=1)
    print(f"\n  正样本 anchor softmax 后最大概率类别:")
    print(f"    预测为背景: {(pos_labels == 0).sum().item()}")
    print(f"    预测为非背景: {(pos_labels != 0).sum().item()}")
    if (pos_labels != 0).sum() > 0:
        non_bg_pos = pos_labels != 0
        print(f"    非背景预测的分数: {pos_max_scores[non_bg_pos].tolist()}")
