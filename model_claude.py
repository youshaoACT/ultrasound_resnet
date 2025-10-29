import torch
import yaml
from easydict import EasyDict
from torchvision import models
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pickle as pkl
import gc

# ========== 新增导入 ==========
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


class PKUDataset(Dataset):
    def __init__(self, data_list, train=True):
        with open(data_list, "rb") as f:
            self.data = pkl.load(f)
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_root = self.data[index]["image_root"]
        label = self.data[index]["label"]

        if self.train:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        images = Image.open(image_root).convert('RGB')
        images = transform(images)

        return images, label


class ResNet50(pl.LightningModule):

    def __init__(self, params):
        super(ResNet50, self).__init__()
        self.params = params
        self.save_hyperparameters()

        # 用于记录验证集的输出
        self.validation_step_outputs = []
        self.training_step_outputs = []

        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, self.params.model.num_target_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_hat = self(x_batch)
        loss = self.criterion(y_hat, y_batch)

        preds = y_hat.argmax(1)
        acc = (preds == y_batch).float().mean()

        # 记录训练指标（每个step）
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False, prog_bar=False)

        # 记录训练指标（每个epoch平均）
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # 保存用于epoch结束时的详细分析
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'acc': acc.detach(),
            'preds': preds.detach(),
            'targets': y_batch.detach()
        })

        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_hat = self(x_batch)
        loss = self.criterion(y_hat, y_batch)

        preds = y_hat.argmax(1)
        acc = (preds == y_batch).float().mean()

        # 记录验证指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        # 保存预测结果
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'acc': acc.detach(),
            'preds': preds.detach(),
            'targets': y_batch.detach()
        })

        return {"val_loss": loss, "val_acc": acc}

    def on_train_epoch_end(self):
        # 计算训练集的详细统计
        all_preds = torch.cat([x['preds'] for x in self.training_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.training_step_outputs])

        # 每个类别的准确率
        for label in range(self.params.model.num_target_classes):
            mask = all_targets == label
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_targets[mask]).float().mean()
                self.log(f'train_acc_class_{label}', class_acc)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        # 汇总验证结果
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])

        # 整体准确率
        correct = (all_preds == all_targets).sum().item()
        total = len(all_targets)
        acc = correct / total

        # 每个类别的准确率
        class_accs = []
        for label in range(self.params.model.num_target_classes):
            mask = all_targets == label
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_targets[mask]).float().mean()
                self.log(f'val_acc_class_{label}', class_acc)
                class_accs.append(class_acc.item())
            else:
                class_accs.append(0.0)

        # 打印详细信息
        print(f"\n{'=' * 70}")
        print(f"Epoch {self.current_epoch} 验证结果:")
        print(f"  总体准确率: {acc:.4f} ({correct}/{total})")
        print(f"  验证loss: {self.trainer.callback_metrics.get('val_loss', 0):.4f}")
        for label, class_acc in enumerate(class_accs):
            num_samples = (all_targets == label).sum().item()
            print(f"  类别 {label} 准确率: {class_acc:.4f} ({num_samples}个样本)")
        print(f"{'=' * 70}\n")

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.params.training.lr,
            weight_decay=1e-4
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def train_dataloader(self):
        train_dataset = PKUDataset(self.params.data_list.train, train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = PKUDataset(self.params.data_list.test, train=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        return val_loader


def main():
    # ========== 初始化 ==========
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"初始 GPU 内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    torch.set_float32_matmul_precision('medium')
    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ========== 加载配置 ==========
    config_path = r"/home/vipuser/ultrasound/resnet/configs/last_query_log_account.yaml"
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    # ========== 创建模型 ==========
    model = ResNet50(config)

    # ========== 配置 TensorBoard Logger ==========
    logger = TensorBoardLogger(
        save_dir='logs/',  # 日志保存目录
        name='resnet50_experiment',  # 实验名称
        version=None,  # 自动生成版本号（如 version_0, version_1...）
        log_graph=True,  # 记录模型结构
        default_hp_metric=True  # 记录超参数
    )

    # ========== 配置回调函数 ==========
    # 1. 模型检查点：保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='resnet50-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,  # 保存最好的3个模型
        save_last=True,  # 保存最后一个模型
        verbose=True
    )

    # 2. 早停：防止过拟合
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=15,  # 15个epoch没提升就停止
        mode='max',
        verbose=True,
        min_delta=0.001  # 最小改善阈值
    )

    # 3. 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ========== 创建 Trainer ==========
    trainer = pl.Trainer(
        max_epochs=config.training.n_epochs,
        accelerator='gpu',
        devices=1,
        precision=16,
        enable_progress_bar=True,
        logger=logger,  # 添加 logger
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor
        ],
        log_every_n_steps=10,  # 每10步记录一次
        strategy='auto'
    )

    # ========== 训练 ==========
    try:
        trainer.fit(model)
        print("\n✓ 训练成功完成！")
        print(f"✓ 最佳模型保存在: {checkpoint_callback.best_model_path}")
        print(f"✓ TensorBoard 日志保存在: {logger.log_dir}")
    except KeyboardInterrupt:
        print("\n⚠ 训练被用户中断")
    except Exception as e:
        print(f"\n✗ 训练出错: {e}")
        raise
    finally:
        # ========== 清理内存 ==========
        print("\n正在清理 GPU 内存...")
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            print(f"清理后 GPU 内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print("✓ 清理完成！")


if __name__ == '__main__':
    main()