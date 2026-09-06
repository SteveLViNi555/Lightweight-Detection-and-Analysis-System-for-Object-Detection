"""验证 batch 推理是否与单张推理结果一致"""
import torch
import yaml
from torch.utils.data import DataLoader
from core.model import SSD
from core.dataset import VOCDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config/default.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

model = SSD(backbone=cfg['backbone'], num_classes=cfg['num_classes']).to(device)
model.load_state_dict(torch.load('runs/ssd_epoch_100.pth', map_location=device))
model.eval()

dataset = VOCDataset(root='data/VOCdevkit/VOC2007', image_set='val', transform=None)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels

# 1. 单张推理（无 DataLoader）
print("===== 单张推理（前8张）=====")
single_results = []
with torch.no_grad():
    for i in range(8):
        img, _, _ = dataset[i]
        img = img.unsqueeze(0).to(device)
        loc, conf = model(img)
        scores = torch.softmax(conf[0], dim=1)
        max_scores, pred_labels = scores.max(dim=1)
        nonbg = (pred_labels != 0).sum().item()
        single_results.append(nonbg)
        print(f"  图{i}: 非背景={nonbg}")

# 2. batch=8 推理（与 detect_eval 完全一致，含 num_workers）
print("\n===== batch=8 推理（num_workers=2，与 detect_eval 一致）=====")
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2,
                    pin_memory=True, collate_fn=collate_fn)
with torch.no_grad():
    for images, _, _ in loader:
        images = images.to(device)
        loc_preds, conf_preds = model(images)
        for b in range(images.size(0)):
            scores = torch.softmax(conf_preds[b], dim=1)
            max_scores, pred_labels = scores.max(dim=1)
            nonbg = (pred_labels != 0).sum().item()
            print(f"  图{b}: 非背景={nonbg} (单张时={single_results[b]})")
        break

# 3. batch=8 推理（num_workers=0 对照）
print("\n===== batch=8 推理（num_workers=0 对照）=====")
loader0 = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
with torch.no_grad():
    for images, _, _ in loader0:
        images = images.to(device)
        loc_preds, conf_preds = model(images)
        for b in range(images.size(0)):
            scores = torch.softmax(conf_preds[b], dim=1)
            max_scores, pred_labels = scores.max(dim=1)
            nonbg = (pred_labels != 0).sum().item()
            print(f"  图{b}: 非背景={nonbg} (单张时={single_results[b]})")
        break
