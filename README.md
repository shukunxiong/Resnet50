# Resnet50
This is a simple implementation of ResNet50, trained on the Caltech101 dataset for a basic classification task, and configured with a custom Adam optimizer. The advantage of this project is that all parts of the network and optimizer, along with their functions, are clearly annotated in Chinese, making it suitable for beginners.
## Installation
### 1. Clone the repository:
Follow the steps below to set up the project on your local machine:
```bash
git clone https://github.com:shukunxiong/Resnet50.git
cd Resnet50    
```

### 2.Create new conda environment:
```bash
conda create --name resnet python=3.12
conda activate resnet
```

### 3.Install dependencies
```bash
pip install -r requirements.txt
```

## Dataset
This project mainly supports training for classification tasks using the Caltech101 dataset, and it also supports training with your own dataset. Please note, when using your own dataset for training, the step 2 must be followed.

### Use caltech101
#### 1.First, you need to download the Caltech101 dataset from the website and unzip it into the 'dataset' folder:
```
https://www.kaggle.com/datasets/imbikramsaha/caltech-101/data
```

#### 2.Then, organize the directory in the following format:
```
/root/adam/dataset/caltech-101/
├── 101_ObjectCategories/
│ ├── accordion/
│ ├── airplanes/
│ ├── ... (other categories)
└── Annotations/
```
#### 3.Next, run `dataset_trans.py` to generate the annotation files, and place the generated files under the `dataset` directory.

#### 4.Finally, replace the file path in `train.py`:
```python
    train_labels_file      = 'your_train_labels.txt'
    val_labels_file        = 'your_val_labels.txt'
    categories_file        = 'your_categories.txt'
```

with the newly generated file path, and then run:
```python
python train.py
```
### Use your own dataset
#### 1.Make the split between the validation set and training set, and construct the annotation file in the following format:
```
dataset/
    ├── train_labels.txt  
    ├── val_labels.txt  
    ├── categories.txt               
    └── your_own_dataset          

```

Among them, `train_labels.txt`and `val_labels.txt` as follows:
```
image_0001_path class_0001 
image_0002_path class_0002
......          
```

Furthermore, the `categories.txt` file contains all the categories, and its content is as follows:
```
class1
class2
......
```
#### 2.Finally, replace the file path in `train.py`:
```python
    train_labels_file      = 'your_train_labels.txt'
    val_labels_file        = 'your_val_labels.txt'
    categories_file        = 'your_categories.txt'
```

with the newly file path, and then run:
```python
python train.py
```




     