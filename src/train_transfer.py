import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
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
        preprocessing_function=preprocess_input,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        shear_range=0.2
    )
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen  = ImageDataGenerator(preprocessing_function=preprocess_input)

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

def build_transfer_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history, name='transfer'):
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
    print('\n=== Building EfficientNetB0 Transfer Learning Model ===')
    model = build_transfer_model()
    model.summary()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint('models/transfer_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    print('\n=== Training Transfer Learning Model ===')
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
    plot_history(history, name='transfer')
    metrics = {'test_accuracy': float(acc), 'test_loss': float(loss)}
    with open('logs/transfer_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print('Transfer Learning training complete!')
    print('Model saved to models/transfer_model.h5')
