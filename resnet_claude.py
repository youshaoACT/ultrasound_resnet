"""
使用torchvision预训练ResNet训练医学图像分类
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse



def create_model(num_classes, pretrained=True):
    """
    创建ResNet-50模型

    Args:
        num_classes: 分类数量
        pretrained: 是否使用ImageNet预训练权重
    """
    print(f"创建ResNet-50模型 (pretrained={pretrained})...")

    # 加载预训练模型
    model = models.resnet50(pretrained=pretrained)

    # 修改最后的全连接层
    num_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(num_features, num_classes)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


# ==================== 4. 训练函数 ====================
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


# ==================== 5. 验证函数 ====================
def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            # 获取预测
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    # 计算AUC（如果是二分类）
    if len(set(all_labels)) == 2:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    else:
        epoch_auc = 0

    return epoch_loss, epoch_acc, epoch_auc


# ==================== 6. 主训练流程 ====================
def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据变换
    train_transform = get_transforms(args.image_size, augment=True)
    val_transform = get_transforms(args.image_size, augment=False)

    # 创建数据集
    print("\n加载数据集...")
    train_dataset = MedicalImageDataset(
        args.train_csv, args.data_dir, transform=train_transform
    )
    val_dataset = MedicalImageDataset(
        args.val_csv, args.data_dir, transform=val_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 创建模型
    model = create_model(args.num_classes, pretrained=args.pretrained)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 优化器：只训练特定层或全部训练
    if args.freeze_backbone:
        # 冻结backbone，只训练分类头
        print("冻结backbone，只训练最后的FC层")
        for param in model.parameters():
            param.requires_grad = False
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    else:
        # 训练全部参数
        print("训练全部参数")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 混合精度训练
    scaler = GradScaler()

    # 训练循环
    best_acc = 0
    best_auc = 0

    print(f"\n开始训练，共{args.epochs}个epoch")
    print("=" * 70)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # 验证
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device
        )

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印结果
        print(f"\n结果:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_auc': best_auc,
            }, args.output_dir / 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (Acc: {val_acc:.4f})")

        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.output_dir / f'checkpoint_epoch_{epoch + 1}.pth')

    print("\n" + "=" * 70)
    print(f"训练完成！")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"最佳验证AUC: {best_auc:.4f}")
    print(f"模型保存在: {args.output_dir / 'best_model.pth'}")


# ==================== 7. 命令行参数 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet医学图像分类训练')

    # 数据相关
    parser.add_argument('--data_dir', type=str, required=True,
                        help='图像根目录')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='训练集CSV文件')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='验证集CSV文件')

    # 模型相关
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类数量')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='冻结backbone，只训练分类头')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--image_size', type=int, default=224,
                        help='图像大小')

    # 其他
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)