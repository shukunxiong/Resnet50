# 该部分是用于训练的函数
# from scipy.io import loadmat
# data = loadmat('/workspace/xiongshukun/adam/dataset/caltech-101/Annotations/accordion/annotation_0001.mat')
# print(data)
from adam import Adam
from net import ResNet50
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import CustomDataset, get_dataloader
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


# 用于训练过程中对当前的精度进行验证，以便于实时评估当前的训练情况
def evaluate(model, val_dataloader):
    model.eval()
    all_preds              = []
    all_labels             = []

    # 评估模式不计算梯度信息
    with torch.no_grad():  
        for imgs, labels in val_dataloader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            labels_ = torch.argmax(labels, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return acc

def main():
    # 超参数设置
    batch_size             = 8
    epochs                 = 100
    learning_rate          = 0.0001
    num_classes            = 102  
    
    # 指定训练与测试集划分地址以及类别地址
    train_labels_file      = 'your_train_labels.txt'
    val_labels_file        = 'your_val_labels.txt'
    categories_file        = 'your_categories.txt'
    
    # 构建DataLoder
    train_dataloader       = get_dataloader(train_labels_file, categories_file, batch_size=batch_size, mode='train')
    val_dataloader         = get_dataloader(val_labels_file, categories_file, batch_size=batch_size, mode='eval')
    
    # 模型初始化
    model                  = ResNet50(num_classes=num_classes).cuda()  # 将模型移至GPU
    
    # 定义损失函数以及优化器
    criterion              = nn.CrossEntropyLoss()
    # optimizer              = Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 用于记录历史最高精度，便于保存模型的权重
    best_acc = 0.0
    
    # 开始训练
    for epoch in tqdm(range(epochs)):
        model.train()  # 切换到训练模式
        running_loss       = 0.0
        
    # 单个epoch的训练
        for imgs, labels in train_dataloader:
            
            imgs, labels   = imgs.cuda(), labels.cuda()
            
            # 清空当前累计的梯度
            optimizer.zero_grad()
            
            outputs        = model(imgs)
            
            # 计算损失，其中的output为(batch_size,num_categories);而后者为(batch_size,),主要为各样本的标签号
            loss           = criterion(outputs, torch.max(labels, 1)[1])  
            loss.backward()  # 反向传播
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # 更新权重

            # 累计当前epoch的损失
            running_loss  += loss.item()
            
            
        # 每个epoch结束后，计算训练集上的损失
        epoch_loss          = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        # 评估验证集上的损失
        val_acc             = evaluate(model, val_dataloader)
        print(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.4f}")
        
        # 保存当前模型权重以及最佳权重的模型
        if val_acc > best_acc:
            best_acc        = val_acc
            print(f"New best accuracy: {best_acc:.4f}")
            torch.save(model.state_dict(), 'best_model.pth')  # 保存当前