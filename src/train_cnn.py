import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

TRAIN_DIR = 'dataset/classification/train'
VALID_DIR = 'dataset/classification/valid'
TEST_DIR  = 'dataset/classification/test'
IMG_SIZE  = (224, 224)
BATCH     = 16
EPOCHS    = 15
CLASSES   = ['bird', 'drone']

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
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=True
    )
    valid = valid_gen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False
    )
    test = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False
    )
    return train, valid, test

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history, name='cnn'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title(f'{name.upper()} - Accuracy')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
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
    plot_history(history, name='custom_cnn')
    metrics = {'test_accuracy': float(acc), 'test_loss': float(loss)}
    with open('logs/cnn_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print('Custom CNN training complete!')
    print('Model saved to models/custom_cnn.h5')
