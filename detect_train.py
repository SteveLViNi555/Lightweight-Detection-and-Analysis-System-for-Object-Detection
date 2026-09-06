import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import os
import argparse
from core.dataset import VOCDataset, Compose, RandomHorizontalFlip, RandomBrightness
from core.model import SSD
from core.loss import MultiBoxLoss

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='总训练轮数（从0或resume开始算）')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小（显存4G建议8）')
    parser.add_argument('--resume', type=str, default=None, help='加载的权重路径（继续训练）')
    args = parser.parse_args()

    with open('config/default.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 训练总轮数
    epochs = args.epochs
    batch_size = args.batch_size
    print(f"总训练轮数: {epochs}, Batch size: {batch_size}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据增强
    transform = Compose([
        RandomHorizontalFlip(),
        RandomBrightness(),
    ])

    # 数据集加载
    train_dataset = VOCDataset(root='data/VOCdevkit/VOC2007', image_set='train', transform=transform)
    val_dataset = VOCDataset(root='data/VOCdevkit/VOC2007', image_set='val', transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,          # Windows下设为2即可，过高可能出错
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 模型初始化
    model = SSD(backbone=cfg['backbone'], num_classes=cfg['num_classes']).to(device)

    # 加载已有权重（如果指定）
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        # 从文件名中提取已训练的epoch数（例如 ssd_epoch_80.pth -> 80）
        try:
            start_epoch = int(args.resume.split('_')[-1].split('.')[0])
            print(f"加载权重: {args.resume}，从 epoch {start_epoch} 继续训练")
        except:
            start_epoch = 0
            print(f"加载权重: {args.resume}，但无法解析epoch数，从头开始")
    else:
        print("未找到权重文件，从头开始训练")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 损失函数
    priors = model.priors.to(device)
    criterion = MultiBoxLoss(priors, num_classes=cfg['num_classes'])

    # 优化器：标准 SSD 学习率 0.001
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # 学习率调度：每25个epoch减半（50 epoch 内衰减两次）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # TensorBoard记录
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter('runs/experiment_1')

    # ---------- 训练循环 ----------
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, (images, boxes, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = [(b.to(device, non_blocking=True), l.to(device, non_blocking=True))
                       for b, l in zip(boxes, labels)]

            # 前向传播
            loc_preds, conf_preds = model(images)
            loss = criterion((loc_preds, conf_preds), targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Step {i}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        scheduler.step()
        print(f'Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}, Time: {time.time() - start_time:.2f}s')

        # 每5个epoch保存一次
        if (epoch + 1) % 5 == 0:
            save_path = f'runs/ssd_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  保存权重至 {save_path}，大小: {file_size:.2f} MB")

    writer.close()
    print("训练完成！")