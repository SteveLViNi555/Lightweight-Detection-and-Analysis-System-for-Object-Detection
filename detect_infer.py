import torch
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader
from core.dataset import VOCDataset
from core.model import SSD
from core.utils import decode_boxes, nms

CLASS_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels

if __name__ == '__main__':
    with open('config/default.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = SSD(backbone=cfg['backbone'], num_classes=cfg['num_classes']).to(device)
    weights_path = 'runs/ssd_epoch_150.pth'   # 改成你最新的权重
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"加载权重: {weights_path}")
    else:
        print("权重不存在，退出")
        exit()

    priors = model.priors.to(device)

    # 加载验证集
    val_dataset = VOCDataset(root='data/VOCdevkit/VOC2007', image_set='val', transform=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model.eval()
    print("\n正在检查模型分类能力（阈值极低，0.0001）...\n")

    total_predictions = {i: 0 for i in range(21)}
    total_boxes = 0

    with torch.no_grad():
        for idx, (images, boxes_gt, labels_gt) in enumerate(val_loader):
            if idx >= 50:  # 检查前50张图
                break

            images = images.to(device)
            loc_preds, conf_preds = model(images)

            loc = loc_preds[0]
            conf = conf_preds[0]
            boxes = decode_boxes(loc, priors, variance=[0.1, 0.2])
            boxes = boxes.clamp(0, 1)
            scores = torch.softmax(conf, dim=1)
            max_scores, labels = scores.max(dim=1)

            # 🔑 关键：用极低阈值，几乎不过滤
            keep = max_scores > 0.00001
            boxes = boxes[keep]
            labels = labels[keep]
            scores = max_scores[keep]

            if boxes.numel() > 0:
                keep_idx = nms(boxes, scores, overlap=0.45)
                labels = labels[keep_idx]

            # 统计类别分布
            for label in labels:
                total_predictions[label.item()] += 1
                total_boxes += 1

            # 每10张图打印一次进度
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx+1} 张图片...")

    # 打印结果
    print("\n" + "="*50)
    print(f"共处理 50 张图片，总预测框数: {total_boxes}")
    print("\n各类别预测数量（阈值 0.0001）:")
    print("-"*30)

    non_zero_found = False
    for i, count in total_predictions.items():
        if count > 0:
            print(f"  {CLASS_NAMES[i]}: {count}")
            if i != 0:
                non_zero_found = True

    if non_zero_found:
        print("\n✅ 模型出现了非背景类别！说明分类头有微弱学习能力。")
        print("   建议继续训练 20~30 个 epoch，mAP 有望提升。")
    else:
        print("\n❌ 所有预测都是背景（类别0）。")
        print("   分类头完全没有学会区分物体。")
        print("   可能原因：定位损失权重过高，分类头梯度被压制。")
        print("   建议：进一步降低定位损失权重（如 0.05），提高分类损失权重（如 3.0）。")