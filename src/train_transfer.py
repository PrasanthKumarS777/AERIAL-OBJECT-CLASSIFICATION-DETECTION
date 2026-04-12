# train_transfer.py — fine-tuning EfficientNetB0 on the bird vs drone dataset
# used transfer learning here because training from scratch would take forever
# imagenet weights give a really good starting point for aerial images

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

# dataset paths — split into train/valid/test folders
TRAIN_DIR = 'dataset/classification/train'
VALID_DIR = 'dataset/classification/valid'
TEST_DIR  = 'dataset/classification/test'

# image size must be 224x224 for efficientnet
IMG_SIZE  = (224, 224)

# smaller batch size because my machine runs out of memory with larger ones
BATCH     = 16

# 15 epochs max but early stopping usually kicks in before that
EPOCHS    = 15

# binary classification — just bird and drone
CLASSES   = ['bird', 'drone']


def get_generators():
    # augmentation only on training data — validation and test stay clean
    # tried higher rotation range but it hurt accuracy so kept it at 20
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],  # simulates different lighting conditions
        shear_range=0.2
    )

    # no augmentation for validation and test — just preprocessing
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen  = ImageDataGenerator(preprocessing_function=preprocess_input)

    # flow from directory reads images directly from folder structure
    train = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=True  # shuffle training data each epoch
    )
    valid = valid_gen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False  # dont shuffle val — need consistent evaluation
    )
    test = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False  # dont shuffle test either
    )
    return train, valid, test


def build_transfer_model():
    # loading efficientnet without the top classification layer
    # include_top=False lets us add our own head for binary classification
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # freezing the base model — we dont want to mess up the imagenet weights
    base_model.trainable = False

    # adding custom head on top of efficientnet
    # global average pooling reduces spatial dimensions before the dense layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),         # helps with training stability
        Dense(256, activation='relu'),
        Dropout(0.5),                 # dropout to prevent overfitting
        Dense(1, activation='sigmoid')  # sigmoid for binary output
    ])

    # adam with 0.001 lr worked well — tried 0.0001 but converged too slowly
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history, name='transfer'):
    # plotting accuracy and loss curves side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title(f'{name.upper()} - Accuracy')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')

    # loss plot
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

    # early stopping prevents overfitting — stops if val_loss doesnt improve for 3 epochs
    # model checkpoint saves the best version based on val_accuracy
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

    # save training curves to logs folder
    plot_history(history, name='transfer')

    # save metrics to json so app.py can load them for the dashboard
    metrics = {'test_accuracy': float(acc), 'test_loss': float(loss)}
    with open('logs/transfer_metrics.json', 'w') as f:
        json.dump(metrics, f)

    print('Transfer Learning training complete!')
    print('Model saved to models/transfer_model.h5')