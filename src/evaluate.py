# evaluate.py — runs evaluation on both trained models and compares them
# run this after training is done to get confusion matrix and classification report
# saves all outputs to the logs/ folder

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.metrics import confusion_matrix, classification_report
import json

# only need test set here — training is already done at this point
TEST_DIR = 'dataset/classification/test'
IMG_SIZE = (224, 224)
BATCH    = 16
CLASSES  = ['bird', 'drone']


def get_test_generator(use_efficientnet=False):
    # efficientnet needs its own preprocessing function
    # regular cnn just needs rescaling to 0-1
    if use_efficientnet:
        test_gen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
    else:
        test_gen = ImageDataGenerator(rescale=1./255)

    # shuffle must be False here — we need labels in the same order as predictions
    test = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False
    )
    return test


def evaluate_model(model_path, model_name, use_efficientnet=False):
    print(f'\n=== Evaluating {model_name} ===')

    # load the saved model from disk
    model = load_model(model_path)
    test = get_test_generator(use_efficientnet)

    # get ground truth labels and model predictions
    y_true = test.classes
    y_pred_prob = model.predict(test, verbose=1)

    # threshold at 0.5 — above is drone, below is bird
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # confusion matrix — shows where the model is making mistakes
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'logs/{model_name}_confusion_matrix.png')
    plt.close()
    print(f'Confusion matrix saved to logs/{model_name}_confusion_matrix.png')

    # full classification report with precision recall f1 per class
    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    print(f'\nClassification Report:\n{report}')

    # save report to text file for reference
    with open(f'logs/{model_name}_report.txt', 'w') as f:
        f.write(f'{model_name} Classification Report\n')
        f.write('=' * 50 + '\n')
        f.write(report)


def compare_models():
    # reads the saved metrics json files from both training runs
    # and plots a bar chart comparing their test accuracies
    print('\n=== MODEL COMPARISON ===')
    results = {}

    file_map = {
        'Custom_CNN': 'logs/cnn_metrics.json',
        'Transfer_EfficientNetB0': 'logs/transfer_metrics.json'
    }

    for name, path in file_map.items():
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
            results[name] = metrics
            print(f'{name:30} -> Accuracy: {metrics["test_accuracy"]:.4f}')
        else:
            # skip if metrics file doesnt exist yet
            print(f'{name} metrics not found - skipping')

    if results:
        names = list(results.keys())
        accs  = [results[n]['test_accuracy'] for n in names]

        # bar chart comparing both models
        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, accs, color=['#2196F3', '#4CAF50'])
        plt.ylim(0, 1)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Test Accuracy')

        # adding accuracy percentage labels on top of each bar
        for bar, acc in zip(bars, accs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', fontweight='bold'
            )

        plt.tight_layout()
        plt.savefig('logs/model_comparison.png')
        plt.close()
        print('Model comparison chart saved to logs/model_comparison.png')


if __name__ == '__main__':
    # evaluate custom cnn first
    evaluate_model('models/custom_cnn.h5', 'Custom_CNN', use_efficientnet=False)

    # only evaluate transfer model if it exists — might not be trained yet
    if os.path.exists('models/transfer_model.h5'):
        evaluate_model('models/transfer_model.h5', 'Transfer_EfficientNetB0', use_efficientnet=True)

    # compare both models side by side
    compare_models()

    print('\nEvaluation complete!')