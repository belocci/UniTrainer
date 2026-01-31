# Training a Real YOLO Model for Rust Player Detection

## The Issue

The Uni Trainer app currently saves **training metadata as JSON**, not actual PyTorch model files. For live detection to work, you need a **real PyTorch .pt model file** from actual YOLO training.

## Solution: Train a Real YOLO Model

You need to train your model using Ultralytics YOLO directly. Here's how:

### Option 1: Quick Training Script

Create a file `train_rust_detector.py`:

```python
from ultralytics import YOLO
import os

# Initialize model (YOLOv11s for balanced speed/accuracy)
model = YOLO('yolo11s.pt')  # Downloads pre-trained weights automatically

# Train the model
results = model.train(
    data='path/to/your/rust_dataset/data.yaml',  # Your dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # 0 for GPU, 'cpu' for CPU
    project='rust_detection',
    name='yolov11s_rust',
    save=True
)

# Export the best model
best_model = model.export(format='pt')
print(f"Model saved to: {best_model}")
```

### Option 2: Using Command Line

```bash
# Install ultralytics if not already installed
pip install ultralytics

# Train the model
yolo detect train data=path/to/your/data.yaml model=yolo11s.pt epochs=100 imgsz=640 batch=16 device=0

# The trained model will be in: runs/detect/train/weights/best.pt
```

### Dataset Structure

Your dataset should be organized like this:

```
rust_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── val1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── val1.txt
        └── ...
```

### data.yaml Example

```yaml
path: C:/path/to/rust_dataset
train: images/train
val: images/val

names:
  0: player
```

### Label Format (YOLO format)

Each label file (image1.txt) should contain:
```
0 0.5 0.5 0.3 0.4
```
Format: `class_id center_x center_y width height` (all normalized 0-1)

## After Training

1. **Locate your trained model**:
   - Usually in: `runs/detect/train/weights/best.pt`
   - Or: `runs/detect/yolov11s_rust/weights/best.pt`

2. **Use this file in Uni Trainer**:
   - Copy the model file to: `C:\Users\vaugh\Documents\UniTrainer\Models\`
   - Or use the full path to the model file
   - This should be a **real PyTorch .pt file** (10-50MB+)

3. **Test the model**:
   ```bash
   python detector.py "path/to/your/real/model.pt"
   ```

## Quick Test with Pre-trained Model

To test the detection system works, you can use a pre-trained YOLO model:

```python
from ultralytics import YOLO

# Download and use pre-trained COCO model
model = YOLO('yolo11s.pt')  # Downloads automatically
results = model('path/to/test/image.jpg')
```

Then use this model path in Uni Trainer for testing.

## Next Steps

1. **If you have a real trained model**: Point Uni Trainer to that file
2. **If you need to train**: Use the scripts above with your Rust player dataset
3. **For testing**: Use a pre-trained YOLO model first to verify the detection system works
