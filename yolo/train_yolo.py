from ultralytics import YOLO
import os

DATA_YAML = 'yolo/data.yaml'
EPOCHS    = 3
IMG_SIZE  = 224
BATCH     = 32

if __name__ == '__main__':
    print('=== YOLOv8 Training ===')
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device='cpu',
        fraction=0.3,
        workers=0,
        cache=True,
        optimizer='SGD',
        lr0=0.01,
        warmup_epochs=1,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        project='runs/detect',
        name='bird_drone',
        exist_ok=True,
        verbose=False
    )
    print('YOLOv8 training complete!')

    metrics = model.val()
    print(f'mAP50:     {metrics.box.map50:.4f}')
    print(f'mAP50-95:  {metrics.box.map:.4f}')

    model.export(format='onnx')
    print('Model exported to ONNX!')