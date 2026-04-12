# preprocess.py — dataset inspection and preprocessing script
# run this first before training to make sure the data is loaded correctly
# also saves some sample images so you can visually verify the dataset

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dataset split paths — folder structure is train/valid/test with bird/ and drone/ inside each
TRAIN_DIR = 'dataset/classification/train'
VALID_DIR = 'dataset/classification/valid'
TEST_DIR  = 'dataset/classification/test'

# standard image size for the models
IMG_SIZE  = (224, 224)

# batch size — kept small to avoid memory issues
BATCH     = 16

# only two classes in this project
CLASSES   = ['bird', 'drone']


def get_generators():
    # augmentation on training data only — helps prevent overfitting
    train_gen = ImageDataGenerator(
        rescale=1./255,               # normalize to 0-1
        rotation_range=20,            # random rotation
        horizontal_flip=True,         # flip images horizontally
        zoom_range=0.2,               # slight zoom
        brightness_range=[0.8, 1.2],  # vary lighting
        shear_range=0.2
    )

    # validation and test just get rescaling — no augmentation
    valid_gen = ImageDataGenerator(rescale=1./255)
    test_gen  = ImageDataGenerator(rescale=1./255)

    # flow images directly from directory
    train = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=True   # shuffle training data
    )
    valid = valid_gen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False  # keep validation order consistent
    )
    test = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False  # same for test
    )
    return train, valid, test


def visualize_samples():
    # showing 5 images from each class in a grid — just to visually verify the data looks right
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images - Bird vs Drone', fontsize=16)

    for i, cls in enumerate(CLASSES):
        folder = os.path.join(TRAIN_DIR, cls)
        images = os.listdir(folder)[:5]  # just grab first 5 images
        for j, img_name in enumerate(images):
            img = mpimg.imread(os.path.join(folder, img_name))
            axes[i][j].imshow(img)
            axes[i][j].set_title(cls.upper())
            axes[i][j].axis('off')  # hide axis ticks — looks cleaner

    plt.tight_layout()
    plt.savefig('logs/sample_images.png')
    plt.close()
    print('✅ Sample images saved to logs/sample_images.png')


def check_distribution():
    # counting images in each class for each split
    # useful to check if dataset is balanced — bird and drone should be roughly equal
    for split in ['train', 'valid', 'test']:
        base = f'dataset/classification/{split}'
        bird  = len(os.listdir(f'{base}/bird'))
        drone = len(os.listdir(f'{base}/drone'))
        print(f'{split:6} -> bird: {bird:4} | drone: {drone:4} | total: {bird+drone}')


if __name__ == '__main__':
    print('=== Dataset Distribution ===')
    check_distribution()

    print('\n=== Loading Generators ===')
    train, valid, test = get_generators()
    print(f'Classes: {train.class_indices}')  # should print bird:0, drone:1

    print('\n=== Visualizing Samples ===')
    visualize_samples()

    print('\n✅ Preprocessing complete!')