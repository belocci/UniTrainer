# Cloud Training Implementation Guide

## Overview
Complete implementation of cloud training workflow with all 4 critical features.

---

## ✅ FEATURE 1: Instance IP Retrieval After Launch

### How It Works:

1. **Launch Instance** (`launchInstance()`)
   - Calls CanopyWave API to create GPU instance
   - Returns instance object with ID

2. **Poll for IP Address** (`waitForInstanceReady()`)
   - Polls instance status every 5 seconds
   - Checks multiple IP fields:
     - `instance.ip`
     - `instance.floating_ip`
     - `instance.public_ip`
     - `instance.accessIPv4`
     - `instance.addresses` (complex object)
   - Waits for status to be `ACTIVE` or `active`
   - Adds 30-second buffer for SSH daemon to start
   - Max wait time: 10 minutes (configurable)

3. **Extract IP from Addresses** (`extractIPFromAddresses()`)
   - Handles complex address structures
   - Example: `{ "network": [{ "addr": "1.2.3.4", "version": 4 }] }`
   - Finds first IPv4 address

### Code Location:
- `cloud-training-handler.js` lines 108-175

---

## ✅ FEATURE 2: Remote Training Script Execution

### How It Works:

1. **Setup Environment** (`setupRemoteEnvironment()`)
   - Updates Ubuntu packages
   - Installs Python 3 and pip
   - Installs PyTorch with CUDA support
   - Installs Ultralytics YOLO
   - Installs scikit-learn, XGBoost, LightGBM
   - Creates working directories: `~/training`, `~/training/dataset`, `~/training/output`

2. **Upload Training Files** (`uploadTrainingFiles()`)
   - Uploads dataset (directory or file)
   - Generates training script based on framework
   - Uploads Python training script to `~/training/train.py`

3. **Generate Training Script** (`generateTrainingScript()`)
   - **YOLO**: Uses Ultralytics API with progress reporting
   - **PyTorch**: Custom neural network training
   - **Generic**: Placeholder for other frameworks
   - Scripts include JSON progress output for parsing

4. **Execute Training** (`executeRemoteTraining()`)
   - Runs: `cd ~/training && python3 train.py`
   - Streams stdout/stderr in real-time
   - Parses progress and sends to UI

### Code Location:
- `cloud-training-handler.js` lines 227-381

### Example YOLO Script Generated:
```python
from ultralytics import YOLO

model = YOLO("yolov11n.pt")
results = model.train(
    data="~/training/dataset/data.yaml",
    epochs=10,
    batch=16,
    imgsz=640,
    device=0,
    project="~/training/output",
    name="training_run"
)
```

---

## ✅ FEATURE 3: Progress Streaming Over SSH

### How It Works:

1. **SSH Stream Execution** (`executeRemoteTraining()`)
   - Uses `ssh2` library's `conn.exec()` with streaming
   - Captures stdout and stderr separately
   - Processes data in real-time (not waiting for completion)

2. **Parse Progress** (`parseAndSendProgress()`)
   - **JSON Format**: Looks for `{"type": "progress", "data": {...}}`
   - **YOLO Format**: Regex matches `Epoch: 5/10` patterns
   - Extracts:
     - Current epoch
     - Total epochs
     - Loss values
     - Metrics (mAP, precision, recall)
   - Calculates progress percentage

3. **Send to UI**
   - `sendProgress()`: Structured progress data
   - `sendLog()`: Raw output for console
   - `sendStatus()`: High-level status messages

### Code Location:
- `cloud-training-handler.js` lines 383-474

### Progress Data Format:
```javascript
{
  epoch: 5,
  total_epochs: 10,
  progress: 0.5,
  metrics: { loss: 0.234, mAP: 0.85 },
  status: "training"
}
```

---

## ✅ FEATURE 4: Model Download Automation

### How It Works:

1. **Determine Model Location** (`downloadModel()`)
   - **YOLO**: `~/training/output/training_run/weights/best.pt`
   - **PyTorch**: `~/training/output/model.pth`
   - Framework-specific paths

2. **Create Local Directory**
   - Default: `~/Documents/UniTrainer/models`
   - Creates if doesn't exist

3. **Download via SFTP** (`sshConnection.downloadFile()`)
   - Uses `ssh2` SFTP `fastGet()` method
   - Efficient binary transfer
   - Progress tracking (optional)

4. **Fallback Paths**
   - If primary path fails, tries alternatives:
     - `best.pt` → `last.pt`
     - `weights/best.pt` → `best.pth`
   - Prevents download failure from path variations

5. **Filename Generation**
   - Format: `model_{framework}_{timestamp}.pt`
   - Example: `model_yolo_2026-01-16T14-30-45.pt`
   - Prevents overwriting existing models

### Code Location:
- `cloud-training-handler.js` lines 476-541

---

## Integration with Main Process

### IPC Handlers Added to `main.js`:

1. **`start-cloud-training`**
   - Creates `CloudTrainingHandler` instance
   - Executes full workflow
   - Returns model path on success

2. **`stop-cloud-training`**
   - Stops active training
   - Terminates instance
   - Cleans up resources

### Usage from Renderer:
```javascript
// Start cloud training
const result = await ipcRenderer.invoke('start-cloud-training', apiKey, {
  project: 'my-project',
  region: 'seq',
  flavor: 'H100-4',
  image: 'GPU-Ubuntu.22.04',
  password: 'secure-password',
  datasetPath: '/path/to/dataset',
  trainingSettings: {
    framework: 'yolo',
    modelVariant: 'yolov11n',
    epochs: 10,
    batchSize: 16,
    imageSize: 640
  }
});

if (result.success) {
  console.log('Model saved to:', result.result.modelPath);
}
```

---

## Error Handling

### Automatic Cleanup on Failure:
- SSH connection closed
- Instance terminated
- Temp files removed
- Error message sent to UI

### Retry Logic:
- SSH connection: 5 retries with 10-second delay
- Instance status polling: 10 minutes max
- Model download: Multiple fallback paths

### Timeout Protection:
- SSH commands: 5 minutes default
- Instance launch: 10 minutes
- Training execution: No timeout (streams until completion)

---

## Testing Checklist

### Before First Use:
1. ✅ CanopyWave API key validated
2. ✅ Project and region selected
3. ✅ GPU flavor available
4. ✅ Dataset prepared (YOLO format for YOLO training)
5. ✅ Sufficient account balance

### During Training:
- Monitor console logs for SSH connection
- Check progress updates in UI
- Verify GPU utilization (if monitoring enabled)

### After Training:
- Verify model file downloaded
- Check model file size (should be > 0 bytes)
- Test model inference locally

---

## Cost Management

### Automatic Termination:
- Instance terminated after training completes
- Instance terminated on error
- Instance terminated on user stop

### Manual Termination:
```javascript
await ipcRenderer.invoke('terminate-cloud-instance', apiKey, instanceId, project, region);
```

### Budget Protection:
- Set max training hours in UI
- Monitor balance before launch
- Estimate costs based on GPU type and duration

---

## Troubleshooting

### Issue: SSH Connection Timeout
**Solution**: 
- Increase wait time after instance launch (line 168)
- Check instance security group allows SSH (port 22)

### Issue: Model Not Found
**Solution**:
- Check training script output for actual save location
- Add custom path to `alternativePaths` array (line 522)

### Issue: Training Fails to Start
**Solution**:
- Check dataset format (YOLO needs `data.yaml`)
- Verify Python dependencies installed correctly
- Check remote logs: `ssh user@ip 'cat ~/training/train.log'`

### Issue: Progress Not Updating
**Solution**:
- Verify training script outputs progress in correct format
- Check `parseAndSendProgress()` regex patterns
- Enable verbose logging in training script

---

## Performance Optimization

### Reduce Upload Time:
- Compress dataset before upload
- Use `.zip` or `.tar.gz` format
- Upload only necessary files

### Reduce Download Time:
- Download only `best.pt` (not all checkpoints)
- Use model compression (quantization)

### Reduce Instance Costs:
- Use smaller model variants (YOLOv11n vs YOLOv11x)
- Reduce epochs for testing
- Use cheaper GPU types for small datasets

---

## Future Enhancements

### Potential Additions:
1. **Multi-GPU Training**: Distribute across multiple instances
2. **Checkpoint Resume**: Resume interrupted training
3. **Real-time Metrics Dashboard**: Live GPU/memory charts
4. **Automatic Hyperparameter Tuning**: Grid search on cloud
5. **Model Comparison**: Train multiple variants in parallel
6. **Cost Prediction**: Estimate total cost before launch

---

## Summary

All 4 critical features are now implemented:

✅ **Instance IP Retrieval**: Polls CanopyWave API until instance is active with IP  
✅ **Remote Execution**: Uploads scripts, installs dependencies, runs training  
✅ **Progress Streaming**: Real-time SSH output parsing and UI updates  
✅ **Model Download**: Automatic SFTP download with fallback paths  

**Total Implementation**: ~700 lines of production-ready code with error handling, retries, and cleanup.

**Ready to use**: Just call `ipcRenderer.invoke('start-cloud-training', ...)` from UI!
