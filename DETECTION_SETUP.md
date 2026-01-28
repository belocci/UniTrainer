# Live Detection Setup Guide

## Overview
This feature allows you to use your trained YOLOv11s model for real-time player detection in Rust using MSS (screen capture) and YOLO inference.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **PyTorch** and **Ultralytics YOLO** installed
3. **MSS** library for screen capture

## Installation Steps

### 1. Install Python Dependencies

Open a terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics torch torchvision mss Pillow numpy
```

### 2. Verify Installation

Test that YOLO can load your model:
```bash
python -c "from ultralytics import YOLO; model = YOLO('C:\\Users\\vaugh\\Documents\\UniTrainer\\Models\\cv_yolo_yolov11s_2026-01-12T02-11-49.pt'); print('Model loaded successfully!')"
```

### 3. Using Live Detection

1. **Open Uni Trainer**
2. **Navigate to "Live Detection" section** (in the left panel)
3. **Enter your model path**:
   - Default: `C:\Users\vaugh\Documents\UniTrainer\Models\cv_yolo_yolov11s_2026-01-12T02-11-49.pt`
   - Or click "Browse..." to select a different model
4. **Adjust settings**:
   - **Confidence Threshold**: 0.1-1.0 (higher = fewer but more confident detections)
   - **FPS**: 10-60 (frames per second for detection)
5. **Click "Start Detection"**
6. **Open Rust** and play - detections will appear in the console/logs

## How It Works

1. **Screen Capture**: MSS captures your screen in real-time
2. **YOLO Inference**: Your trained model processes each frame
3. **Detection Results**: Detected players are returned with bounding boxes and confidence scores
4. **Real-time Updates**: Results are displayed in the UI with detection count and FPS

## Detection Output Format

Each detection includes:
- **Class**: Object class name (e.g., "player")
- **Confidence**: Detection confidence (0.0-1.0)
- **Bounding Box**: [x1, y1, x2, y2] coordinates
- **Class ID**: Numeric class identifier

## Troubleshooting

### "detector.py not found"
- Make sure `detector.py` is in the same directory as `main.js`
- Check that the file wasn't excluded during build

### "Model file not found"
- Verify the model path is correct
- Use absolute paths (full Windows path)
- Check that the .pt file exists

### "Python not found" or "Module not found"
- Ensure Python is in your system PATH
- Install required packages: `pip install -r requirements.txt`
- Try using `python3` instead of `python` (update main.js if needed)

### Low FPS
- Reduce FPS setting in the UI
- Lower confidence threshold to reduce processing
- Close other applications to free up resources
- Consider using a smaller YOLO variant (yolov11n instead of yolov11s)

### No detections
- Lower confidence threshold (try 0.3-0.5)
- Ensure Rust is visible on screen
- Check that the model was trained correctly
- Verify model classes match your detection targets

## Performance Tips

1. **Optimize FPS**: Start with 30 FPS, adjust based on your system
2. **Confidence Tuning**: Start at 0.5, lower for more detections, higher for fewer false positives
3. **Monitor Resources**: Watch CPU/GPU usage in the Resources section
4. **Screen Region**: Future versions may support selecting specific screen regions

## Next Steps (Future Enhancements)

- Overlay detection boxes on screen
- Save detection screenshots
- Detection history/logging
- Customizable detection regions
- Multiple model support
- Detection alerts/notifications
