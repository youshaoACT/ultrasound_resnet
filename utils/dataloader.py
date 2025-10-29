import os, torch, cv2, random
import numpy as np
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms

from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import dataloader
import dataloader.transforms as trans
import json, numbers
from glob import glob
import pickle as pkl
import pickle
"""
这个文档应该起到的作用：
1. 读取文件:定义单个样本如何获取和处理（
2. 将文件格式变成(512,512)
"""

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
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229,0.224,0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                #trans.CropCenterSquare(),
                trans.Resize(size=self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229,0.224,0.225])
            ])

    def __getitems__(self,index):
        data_path = self.data_list[index]
        img_path = data_path["img_root"]

        img = Image.open(img_path).convert('RGB')
        img_torch = self.transform_center(img)
        label = int(data_path["label"])

        return img_torch, label

    def __len__(self):
        return self.size