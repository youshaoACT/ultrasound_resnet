深度学习基本框架
-
数据加载
定义模型
定义训练过程train_one_step()
    每次处理1个batch
    定义损失函数、优化器
    损失函数反向传播loss.backward()
    优化器进行优化optimizer.step()
定义训练过程train_one_epoch()
    每次处理1个epoch
定义训练器trainer()
    完整的训练流程
    trainer()=n个train_one_epoch()的过程加上数据加载、模型保存、tensorboard()记录、早停机制等 
    如果使用pytorch_lightening框架可以直接使用它本身的trainer()

class model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ...
    def training_step(self,batch,batch_idx):
        loss =
        self.log("train_loss",loss)
        return loss
    def validate_step(self,batch,batch_idx):
        acc = ..
        self.log("val_acc",acc)
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),lr = 1e-4)
        scheduler = ...
        return [optimizer], [scheduler]
model = model()
trainer = pl.Trainer(
    max_epochs = 50,
    accelerator = "gpu",
    devices = 1
    precision = 16,
    callbacks = [...]
)


Dataset类&Dataloader类