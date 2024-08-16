import subprocess
from tqdm import tqdm

from net.HAIR import HAIR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
# from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class HAIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = HAIR()
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

if __name__=="__main__":
    path="./ckpt/hair3d.ckpt"
    model = HAIRModel()
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["state_dict"])
    print("number of parameters:",sum(p.numel() for p in model.parameters()))
    test_img=torch.randn(1,3,128,128)
    print(model(test_img).shape)
    print("If the output shape is torch.Size([1, 3, 128, 128]), your setting is correct")

