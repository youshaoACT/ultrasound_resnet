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



class PKUDataset(Dataset):

    def __init__(self,data_list,train = True):
        self.trainsize = (512,512)
        self.train = train
        with open(data_list,'rb') as f:
            tr_dl = pkl.load(f)
        self.data_list = tr_dl
        self.size = len(self.data_list)

        if train:
            self.transform_center = transforms.Compose([
                #trans.CropCenterSquare(),
                transforms.Resize(size=self.trainsize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229,0.224,0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                #trans.CropCenterSquare(),
                transforms.Resize(size=self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229,0.224,0.225])
            ])

    def __getitem__(self,index):
        data_path = self.data_list[index]
        img_path = self.data_list[index]["image_root"]

        img = Image.open(img_path).convert('RGB')
        img_torch = self.transform_center(img)
        label = int(data_path["label"])

        return img_torch, label

    def __len__(self):
        return self.size





class ResNet50(pl.LightningModule):

    def __init__(self, hyparams):
        super(ResNet50, self).__init__()

        self.params = hyparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr
        self.pretrained = self.params.pretrained
        self.num_classes = self.params.num_classes
        self.model = models.resnet18(pretrained=self.pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.initlr,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # 学习率衰减系数
            patience=10,  # 10个epoch没改善就降低学习率
            min_lr=1e-7,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                "monitor": "val_loss",
                'interval': 'epoch',
                'frequency': 1
            },
        }

    #def init_weight(self,ckpt_path = None):

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        #self.model.train()
        x_batch, y_batch = batch
        #y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        #y_batch = y_batch.cuda()
        #x_batch = x_batch.cuda()
        y_hat = self(x_batch)#self(x)=self.__call__(x)=self.forward(x)
        loss = self.criterion(y_hat, y_batch)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        acc = (y_hat.argmax(1) == y_batch).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_hat = self(x_batch)
        loss = self.criterion(y_hat, y_batch)
        acc = (y_hat.argmax(1) == y_batch).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def train_dataloader(self):
        train_dataset = PKUDataset(self.params.data_list.train,train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = PKUDataset(self.params.data_list.test,train=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        return val_loader

def main():
    torch.set_float32_matmul_precision('medium')
    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
   # if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = True

    config_path = r"/home/vipuser/ultrasound/resnet/configs/last_query_log_account.yaml"
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    model = ResNet50(config)

   # checkpoint_callback = ModelCheckpoint(
    #    dirpath="./checkpoints",
     #   filename="{epoch:02d}-{val_loss:.2f}",
      #  monitor="val_acc",
       # mode="max",
        #save_last=True
    #)
    #lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
       # checkpoint_callback=lr_monitor_callback,
        max_epochs = config.training.n_epochs,
        accelerator='gpu',
        devices=1,
        precision=16,
        enable_progress_bar=True,
        strategy='auto',
        #log_every_n_train_steps=5
#        callbacks=[lr_monitor_callback]
    )

    trainer.fit(model)

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
