import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import json

TEST_DIR = 'dataset/classification/test'
IMG_SIZE = (224, 224)
BATCH    = 16
CLASSES  = ['bird', 'drone']

def get_test_generator():
    test_gen = ImageDataGenerator(rescale=1./255)
    test = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode='binary',
        classes=CLASSES, shuffle=False
    )
    return test

def evaluate_model(model_path, model_name):
    print(f'\n=== Evaluating {model_name} ===')
    model = load_model(model_path)
    test = get_test_generator()
    y_true = test.classes
    y_pred_prob = model.predict(test, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'logs/{model_name}_confusion_matrix.png')
    plt.close()
    print(f'Confusion matrix saved to logs/{model_name}_confusion_matrix.png')

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(f'\nClassification Report:\n{report}')

    # Save report
    with open(f'logs/{model_name}_report.txt', 'w') as f:
        f.write(f'{model_name} Classification Report\n')
        f.write('='*50 + '\n')
        f.write(report)

    return report

def compare_models():
    print('\n=== MODEL COMPARISON ===')
    results = {}
    for name, path in [('Custom_CNN', 'models/custom_cnn.h5'),
                        ('Transfer_EfficientNetB0', 'models/transfer_model.h5')]:
        if os.path.exists(path):
            with open(f'logs/{name.lower().replace(" ","_")}_metrics.json') as f:
                metrics = json.load(f)
            results[name] = metrics
            print(f'{name:30} -> Accuracy: {metrics["test_accuracy"]:.4f}')
        else:
            print(f'{name} model not found - skipping')

    # Comparison bar chart
    if results:
        names = list(results.keys())
        accs  = [results[n]['test_accuracy'] for n in names]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, accs, color=['#2196F3', '#4CAF50'])
        plt.ylim(0, 1)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Test Accuracy')
        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f'{acc:.2%}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig('logs/model_comparison.png')
        plt.close()
        print('Model comparison chart saved to logs/model_comparison.png')

if __name__ == '__main__':
    evaluate_model('models/custom_cnn.h5', 'Custom_CNN')
    if os.path.exists('models/transfer_model.h5'):
        evaluate_model('models/transfer_model.h5', 'Transfer_EfficientNetB0')
    compare_models()
    print('\nEvaluation complete!')
