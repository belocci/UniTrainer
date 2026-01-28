# Quick Start: Cloud Training

## 🚀 Train Your AI Model on Cloud GPU in 5 Minutes

---

## Prerequisites

✅ CanopyWave account with API key  
✅ Dataset prepared (YOLO format for object detection)  
✅ Uni Trainer installed  

---

## Step 1: Get CanopyWave API Key

1. Go to https://canopywave.com
2. Sign up / Log in
3. Navigate to **Account → API Keys**
4. Create new API key
5. Copy the key (starts with `cw_...`)

---

## Step 2: Prepare Your Dataset

### For YOLO (Object Detection):

```
my-dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   └── val/
│       ├── img1.txt
│       └── ...
└── data.yaml
```

**data.yaml** example:
```yaml
path: /root/training/dataset
train: images/train
val: images/val

nc: 3  # number of classes
names: ['cat', 'dog', 'bird']
```

---

## Step 3: Launch Uni Trainer

```bash
cd "C:\Users\vaugh\transfer package - uni trainer"
npm start
```

Or run the built executable:
```
dist\Uni Trainer-win32-x64\Uni Trainer.exe
```

---

## Step 4: Configure Cloud Training

### In the Uni Trainer UI:

1. **Switch to Cloud Mode**
   - Toggle: `Local` → `Cloud`

2. **Enter API Key**
   - Paste your CanopyWave API key
   - Click "Validate"
   - ✅ Should show "API key validated"

3. **Select Cloud Resources**
   - **Project**: Select from dropdown
   - **Region**: Choose closest (e.g., `seq` for Seattle)
   - **GPU Type**: 
     - `H100-4` (fastest, most expensive)
     - `A100-8` (balanced)
     - `RTX-4090-2` (budget-friendly)
   - **Image**: `GPU-Ubuntu.22.04`
   - **Password**: Create secure password for SSH

4. **Configure Training**
   - **Model Purpose**: Computer Vision
   - **Framework**: YOLO
   - **Model Variant**: YOLOv11 Nano (fastest for testing)
   - **Epochs**: 10 (use 2-3 for first test)
   - **Batch Size**: 16
   - **Image Size**: 640

5. **Select Dataset**
   - Click "Select Folder"
   - Choose your dataset directory
   - Verify `data.yaml` is present

---

## Step 5: Start Training

1. Click **"Start Training"**
2. Watch the console for progress:

```
✅ Launching cloud GPU instance...
✅ Waiting for instance to be ready...
✅ Instance ready at 203.0.113.45
✅ Connecting to instance via SSH...
✅ SSH connection established
✅ Setting up training environment...
✅ Uploading dataset and training scripts...
✅ Starting training...

Epoch: 1/10  Loss: 0.543  mAP: 0.234
Epoch: 2/10  Loss: 0.421  mAP: 0.456
Epoch: 3/10  Loss: 0.356  mAP: 0.612
...
Epoch: 10/10 Loss: 0.123  mAP: 0.892

✅ Training completed successfully!
✅ Downloading trained model...
✅ Model downloaded: model_yolo_2026-01-16T14-30-45.pt
✅ Cleaning up cloud resources...
✅ Cloud instance terminated

🎉 Training complete!
```

---

## Step 6: Find Your Model

Your trained model is saved to:

```
C:\Users\vaugh\Documents\UniTrainer\models\
└── model_yolo_2026-01-16T14-30-45.pt
```

---

## Step 7: Use Your Model

### Test Inference:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('C:/Users/vaugh/Documents/UniTrainer/models/model_yolo_2026-01-16T14-30-45.pt')

# Run inference
results = model('path/to/test/image.jpg')

# Display results
results[0].show()
```

---

## Troubleshooting

### Issue: "API key validation failed"
**Solution**: 
- Check API key is correct
- Verify CanopyWave account is active
- Check internet connection

### Issue: "No GPU available"
**Solution**:
- Try different region
- Try different GPU type
- Check CanopyWave dashboard for availability

### Issue: "SSH connection timeout"
**Solution**:
- Wait longer (instance may still be starting)
- Check CanopyWave security groups allow SSH (port 22)
- Try again (sometimes instances are slow to start)

### Issue: "Dataset upload failed"
**Solution**:
- Check dataset format (needs `data.yaml`)
- Verify all images have corresponding labels
- Try smaller dataset first

### Issue: "Training failed"
**Solution**:
- Check console logs for error message
- Verify dataset format is correct
- Try with fewer epochs (2-3)
- Check CanopyWave dashboard for instance status

---

## Cost Estimation

### Example Training Session:

| Phase              | Time    | Cost (H100-4 @ $4/hr) |
|--------------------|---------|-----------------------|
| Setup              | 10 min  | $0.67                 |
| Training (10 epochs)| 15 min | $1.00                 |
| Download           | 5 min   | $0.33                 |
| **Total**          | **30 min** | **$2.00**         |

### Tips to Reduce Costs:
- Use smaller GPU types for testing
- Reduce epochs for initial tests
- Compress datasets before upload
- Use smaller model variants (YOLOv11n vs YOLOv11x)

---

## Advanced Configuration

### Use SSH Keys Instead of Password:

1. Generate SSH key pair:
```bash
ssh-keygen -t rsa -b 4096
```

2. Upload public key to CanopyWave

3. In Uni Trainer, select key instead of password

### Monitor Training Remotely:

1. Get instance IP from console logs
2. SSH into instance:
```bash
ssh ubuntu@203.0.113.45
```

3. Check training logs:
```bash
cd ~/training
tail -f train.log
```

### Download Additional Files:

After training, you can manually download:
- Training logs
- Validation images
- Confusion matrices
- All checkpoints (not just best)

---

## Best Practices

### 1. Test Locally First
- Verify dataset format
- Run 1-2 epochs locally
- Fix any errors before cloud training

### 2. Start Small
- First cloud run: 2-3 epochs
- Verify entire workflow works
- Then scale up to full training

### 3. Monitor Costs
- Check CanopyWave balance before starting
- Set max training hours
- Use cheaper GPUs for testing

### 4. Backup Models
- Download models immediately
- Keep multiple versions
- Test inference before deleting

### 5. Clean Up
- Verify instance terminated
- Check no orphaned instances
- Monitor CanopyWave dashboard

---

## Example: Complete YOLO Training

### Dataset: 1000 images, 3 classes (cat, dog, bird)

```javascript
// In Uni Trainer UI or via code:

const config = {
  // Cloud
  project: 'my-ml-project',
  region: 'seq',
  flavor: 'H100-4',
  image: 'GPU-Ubuntu.22.04',
  password: 'SecurePass123!',
  
  // Dataset
  datasetPath: 'C:\\datasets\\animals-yolo',
  
  // Training
  trainingSettings: {
    framework: 'yolo',
    modelVariant: 'yolov11n',
    epochs: 10,
    batchSize: 16,
    imageSize: 640
  }
};

// Start training
const result = await ipcRenderer.invoke('start-cloud-training', apiKey, config);

// Result:
// {
//   success: true,
//   modelPath: "C:\\Users\\vaugh\\Documents\\UniTrainer\\models\\model_yolo_2026-01-16T14-30-45.pt",
//   trainingResult: {
//     epochs: 10,
//     final_loss: 0.123,
//     final_mAP: 0.892
//   }
// }
```

---

## FAQ

**Q: How long does training take?**  
A: Depends on dataset size, model, and epochs. Typical: 10-60 minutes.

**Q: Can I stop training early?**  
A: Yes, click "Stop Training" button. Instance will terminate.

**Q: What happens if my internet disconnects?**  
A: Training continues on cloud. Reconnect and check CanopyWave dashboard.

**Q: Can I train multiple models simultaneously?**  
A: Yes, but each requires separate instance (separate cost).

**Q: What if I forget to terminate instance?**  
A: App auto-terminates after training. Manual check: CanopyWave dashboard.

**Q: Can I use my own cloud provider (AWS/GCP)?**  
A: Not currently. Only CanopyWave supported. (Future: AWS/Azure/GCP)

---

## Support

### Documentation:
- `CLOUD_TRAINING_IMPLEMENTATION.md` - Technical details
- `CLOUD_WORKFLOW.md` - Visual workflow
- `IMPLEMENTATION_SUMMARY.md` - Overview

### Logs:
- Console output in Uni Trainer
- Log file: `%TEMP%\uni-trainer-debug.log`

### CanopyWave Support:
- Dashboard: https://cloud.canopywave.io
- Docs: https://canopywave.com/docs
- Support: support@canopywave.com

---

## Summary

✅ **5-minute setup**  
✅ **Automatic everything** (launch, train, download, cleanup)  
✅ **Real-time progress** updates  
✅ **Cost-effective** (pay only for training time)  
✅ **Production-ready** models  

**You're ready to train AI models on cloud GPUs!** 🚀
