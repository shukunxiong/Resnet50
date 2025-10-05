import os
import random

def split_dataset(data_dir, train_dir, val_dir, train_ratio=0.8):
    """
    Split dataset into training and validation sets, and create label files with category, image path and label.

    Args:
    - data_dir: The directory containing the 101_ObjectCategories folder.
    - train_dir: The destination directory for training images (just for labeling).
    - val_dir: The destination directory for validation images (just for labeling).
    - train_ratio: The proportion of images to use for training (default is 0.8).
    """
    categories = os.listdir(data_dir)
    
    # Create a file to store category names
    with open('categories.txt', 'w') as f:
        for category in categories:
            f.write(category + '\n')

    # Open the label files for writing
    with open('train_labels.txt', 'w') as train_labels, open('val_labels.txt', 'w') as val_labels:
        # Process each category folder
        for category in categories:
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                images = os.listdir(category_path)
                random.shuffle(images)  # Shuffle the images to get random splits
                num_train = int(len(images) * train_ratio)
                train_images = images[:num_train]
                val_images = images[num_train:]

                # Write image paths and labels to train and validation label files
                for image in train_images:
                    image_path = os.path.join(category_path, image)
                    train_labels.write(f'{image_path} {category}\n')
                for image in val_images:
                    image_path = os.path.join(category_path, image)
                    val_labels.write(f'{image_path} {category}\n')

if __name__ == "__main__":
    data_dir = 'your_caltech-101/101_ObjectCategories'
    train_dir = 'your_caltech-101/train'  # Not used but kept for consistency
    val_dir = 'your_caltech-101/val'  # Not used but kept for consistency
    split_dataset(data_dir, train_dir, val_dir)
