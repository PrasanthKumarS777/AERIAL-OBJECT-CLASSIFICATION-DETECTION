from ultralytics import YOLO
import yaml
import os

DATA_YAML = 'yolo/data.yaml'
EPOCHS    = 10
IMG_SIZE  = 416
BATCH     = 8

if __name__ == '__main__':
    print('=== YOLOv8 Training ===')
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device='cpu',
        project='runs/detect',
        name='bird_drone',
        exist_ok=True,
        verbose=True
    )
    print('YOLOv8 training complete!')
    metrics = model.val()
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')
    model.export(format='onnx')
    print('Model exported!')
