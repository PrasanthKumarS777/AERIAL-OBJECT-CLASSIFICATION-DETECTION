import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Paths ──────────────────────────────────────────────
TRAIN_DIR = 'dataset/classification/train'
VALID_DIR = 'dataset/classification/valid'
TEST_DIR  = 'dataset/classification/test'
IMG_SIZE  = (224, 224)
BATCH     = 16

# ── 1. Data Generators ─────────────────────────────────
def get_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        shear_range=0.2
    )
    valid_gen = ImageDataGenerator(rescale=1./255)
    test_gen  = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary', shuffle=True
    )
    valid = valid_gen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary', shuffle=False
    )
    test = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary', shuffle=False
    )
    return train, valid, test

# ── 2. Visualize Sample Images ─────────────────────────
def visualize_samples():
    classes = ['bird', 'drone']
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images - Bird vs Drone', fontsize=16)
    for i, cls in enumerate(classes):
        folder = os.path.join(TRAIN_DIR, cls)
        images = os.listdir(folder)[:5]
        for j, img_name in enumerate(images):
            img = mpimg.imread(os.path.join(folder, img_name))
            axes[i][j].imshow(img)
            axes[i][j].set_title(cls.upper())
            axes[i][j].axis('off')
    plt.tight_layout()
    plt.savefig('logs/sample_images.png')
    print('✅ Sample images saved to logs/sample_images.png')
    plt.show()

# ── 3. Class Distribution ──────────────────────────────
def check_distribution():
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
    print(f'Classes: {train.class_indices}')
    print('\n=== Visualizing Samples ===')
    visualize_samples()
    print('\n✅ Preprocessing complete!')
