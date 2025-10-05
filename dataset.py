# 该文件用于对数据进行加载与预处理，供后续进行训练
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class CustomDataset(Dataset):
    def __init__(self, labels_file, categories_file, mode='train', transform=None):
        """
        Custom dataset for loading images and their labels from given txt files.

        Args:
            labels_file (str): Path to the train_labels.txt file.
            categories_file (str): Path to the cate.txt file.
            mode (str): 'train' or 'eval' mode. Default is 'train'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # 读取类别种类
        with open(categories_file, 'r') as f:
            self.categories    = [line.strip() for line in f.readlines()]
        
        # 构造各类别标签对应表
        self.category_to_index = {category: index for index, category in enumerate(self.categories)}

        # 从标签读取图像地址以及对应的图像标签
        with open(labels_file, 'r') as f:
            lines              = f.readlines()
        self.data              = [(line.split()[0], line.split()[1]) for line in lines]  # (image_path, label)

        # 这里有两种模式，训练与测试。训练阶段会启用图像增强的手段
        self.mode = mode

        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # 'eval' mode
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")  
        
        if self.transform:
            img = self.transform(img)
        
        label_idx = self.category_to_index[label]
        label_onehot = torch.zeros(len(self.categories))
        label_onehot[label_idx] = 1

        return img, label_onehot

def get_dataloader(labels_file, categories_file, batch_size=32, mode='train'):
    dataset = CustomDataset(labels_file, categories_file, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=4)
    return dataloader


