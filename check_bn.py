"""对比 train 模式 vs eval 模式下模型对同一张图的预测
如果 eval 模式预测崩溃而 train 模式正常 → BatchNorm running stats 损坏
"""
import torch
import yaml
from core.model import SSD
from core.dataset import VOCDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config/default.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

model = SSD(backbone=cfg['backbone'], num_classes=cfg['num_classes']).to(device)
model.load_state_dict(torch.load('runs/ssd_epoch_100.pth', map_location=device))
priors = model.priors.to(device)

dataset = VOCDataset(root='data/VOCdevkit/VOC2007', image_set='train', transform=None)
img, boxes, labels = dataset[0]
img = img.unsqueeze(0).to(device)
print(f"train[0] GT: {labels.tolist()}")

def predict(mode_name):
    if mode_name == 'train':
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        loc, conf = model(img)
    scores = torch.softmax(conf[0], dim=1)
    max_scores, pred_labels = scores.max(dim=1)
    nonbg = pred_labels != 0
    print(f"\n[{mode_name} 模式]")
    print(f"  非背景预测数: {nonbg.sum().item()}")
    print(f"  background logit: mean={conf[0][:,0].mean().item():.2f}, max={conf[0][:,0].max().item():.2f}")
    if nonbg.sum() > 0:
        print(f"  非背景最高分数 top5: {sorted(max_scores[nonbg].tolist(), reverse=True)[:5]}")
        print(f"  对应类别: {pred_labels[nonbg].tolist()[:10]}")

predict('train')
predict('eval')

# 检查 BN running stats 是否异常
print("\n===== BatchNorm running stats 检查 =====")
for name, module in model.backbone.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        rm = module.running_mean
        rv = module.running_var
        nan_flag = torch.isnan(rm).any() or torch.isnan(rv).any() or (rv < 0).any()
        print(f"  {name}: mean_abs={rm.abs().mean().item():.3f}, var_mean={rv.mean().item():.3f}, 异常={nan_flag.item()}")
        break  # 只看第一个

print("\n===== BN bn1 详细统计 =====")
bn1 = model.backbone.bn1
print(f"  running_mean: {bn1.running_mean.tolist()}")
print(f"  running_var: {bn1.running_var.tolist()}")
print(f"  num_batches_tracked: {bn1.num_batches_tracked.item()}")
