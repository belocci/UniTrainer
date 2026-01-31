# Real Training Implementation Guide

## Overview
This document outlines the implementation of REAL model training in Uni Trainer, replacing the simulation with actual training using Python.

## Current Status
- ✅ `trainer.py` created with infrastructure for YOLO and sklearn
- ✅ `main.js` IPC handlers added for training process management
- ⏳ `renderer.js` needs to be updated to use real training
- ⏳ File handling needs to be implemented (browser File objects → disk)

## Implementation Plan

### Phase 1: File Handling
**Problem**: Browser File objects need to be saved to disk for Python trainer to access them.

**Solutions**:
1. **Option A (Current)**: When training starts, read files with FileReader, send to main process via IPC, save to temp directory
2. **Option B (Future)**: Use Electron dialog to select files/directories (gives paths directly)

**Implementation**: Use Option A for now, add Option B later.

### Phase 2: Training Integration
1. Update `startTraining()` to:
   - Save uploaded files to temp directory
   - Check if model type is supported for real training
   - If supported: Call real training via IPC
   - If not supported: Use simulation (with warning)
2. Add IPC handlers for:
   - `training-progress`: Update UI with real metrics
   - `training-log`: Display training logs
   - `training-result`: Handle completion, save model path
   - `training-error`: Display errors

### Phase 3: Model Support (Incremental)
- ✅ YOLO (computer_vision/yolo) - Implemented
- ✅ sklearn (machine_learning/sklearn) - Implemented
- ⏳ XGBoost - Partially implemented
- ⏳ LightGBM - Not implemented
- ⏳ PyTorch MLP - Not implemented
- ⏳ TensorFlow - Not implemented
- ⏳ NLP models (BERT, GPT, LSTM) - Not implemented
- ⏳ RL models (DQN, PPO) - Not implemented
- ⏳ Time Series - Not implemented
- ⏳ Generative models - Not implemented

### Phase 4: Model Saving
- Update `save-model` handler to handle real model files
- Copy/move trained models to final location
- Save metadata JSON alongside model file

## File Structure
```
temp/
  training_XXX/  (unique ID per training session)
    data/        (uploaded files)
    output/      (trained model)
```

## Next Steps
1. Implement file saving helper function
2. Update startTraining() to use real training for supported models
3. Add IPC handlers for training progress
4. Test with YOLO dataset
5. Test with sklearn CSV dataset
6. Incrementally add more model support

## Notes
- Real training requires actual datasets in proper format
- YOLO needs YOLO-format dataset (images/ + labels/ + data.yaml)
- sklearn needs CSV with features + labels
- Some models may require significant preprocessing
- Training time will be REAL (not simulated), can take hours/days
