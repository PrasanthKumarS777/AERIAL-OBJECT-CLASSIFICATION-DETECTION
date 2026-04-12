# train_cnn.py — building and training a custom CNN from scratch
# this is the baseline model before i tried transfer learning
# doesnt perform as well as efficientnet but good to compare between the models

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# same paths as train_transfer.py
TRAIN_DIR = 'dataset/classification/train'
VALID_DIR = 'dataset/classification/valid'
TEST_DIR  = 'dataset/classification/test'

# 224x224 is standard for most cnn architectures
IMG_SIZE  = (224, 224)

# batch 16 to avoid memory issues on my pc
BATCH     = 16

# early stopping usually stops before 15 epochs anyway
EPOCHS    = 15

# only two classes — bird and drone
CLASSES   = ['bird', 'drone']


def get_generators():
    # augmenting training data to help the model generalize better
    # without augmentation it was overfitting badly
    train_gen = ImageDataGenerator(
        rescale=1./255,           # normalize pixel values to 0-1
        rotation_range=20,        # random rotation up to 20 degrees
        horizontal_flip=True,     # flip images horizontally
        zoom_range=0.2,           # slight zoom in/out
        brightness_range=[0.8, 1.2],  # vary brightness a bit
        shear_range=0.2
    )

    # only rescaling for validation and test — no augmentation
    valid_gen = ImageDataGenerator(rescale=1./255)
    test_gen  = ImageDataGenerator(rescale=1./255)

    # reading images from folder structure
    train = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=True   # shuffle each epoch
    )
    valid = valid_gen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False  # no shuffle for validation
    )
    test = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False  # no shuffle for test
    )
    return train, valid, test


def build_cnn():
    # 3 conv blocks with increasing filters — 32, 64, 128
    # each block has batchnorm and dropout to reduce overfitting
    # tried without batchnorm first and training was really unstable
    model = Sequential([
        # first conv block — catches basic edges and textures
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # second conv block — learns more complex patterns
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # third conv block — higher level features
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # flatten and pass to fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),  # higher dropout before final layer

        # single sigmoid output for binary classification
        Dense(1, activation='sigmoid')
    ])

    # adam optimizer with default lr works fine here
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history, name='cnn'):
    # side by side accuracy and loss plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # accuracy curves
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title(f'{name.upper()} - Accuracy')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')

    # loss curves
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{name.upper()} - Loss')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'logs/{name}_history.png')
    plt.close()
    print(f'Training plot saved to logs/{name}_history.png')


if __name__ == '__main__':
    print('=== Loading Data ===')
    train, valid, test = get_generators()

    print('\n=== Building Custom CNN ===')
    model = build_cnn()
    model.summary()

    # early stopping watches val_loss — stops if no improvement for 3 epochs
    # model checkpoint keeps the best weights based on val_accuracy
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint('models/custom_cnn.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print('\n=== Training Custom CNN ===')
    history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=valid,
        callbacks=callbacks,
        verbose=1
    )

    print('\n=== Evaluating on Test Set ===')
    loss, acc = model.evaluate(test, verbose=0)
    print(f'Test Accuracy : {acc:.4f}')
    print(f'Test Loss     : {loss:.4f}')

    # save training plots to logs folder
    plot_history(history, name='custom_cnn')

    # save metrics to json — app.py reads this for the dashboard
    metrics = {'test_accuracy': float(acc), 'test_loss': float(loss)}
    with open('logs/cnn_metrics.json', 'w') as f:
        json.dump(metrics, f)

    print('Custom CNN training complete!')
    print('Model saved to models/custom_cnn.h5')
