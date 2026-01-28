# Debug Commands for Uni Trainer

## Browser Console (F12 → Console Tab)

### Check Neural Network State
```javascript
// Check if neural network is initialized
console.log('Neural Network:', window.neuralNetwork);
console.log('Is Training:', window.neuralNetwork?.isTraining);
console.log('Is Validating:', window.neuralNetwork?.isValidating);
console.log('Training Progress:', window.neuralNetwork?.trainingProgress);
console.log('Layers:', window.neuralNetwork?.layers?.length || 0);
console.log('Connections:', window.neuralNetwork?.connections?.length || 0);
```

### Check Canvas
```javascript
// Check canvas element
const canvas = document.getElementById('neuralCanvas');
console.log('Canvas:', canvas);
console.log('Canvas dimensions:', canvas?.width, 'x', canvas?.height);
console.log('Canvas CSS:', window.getComputedStyle(canvas));
console.log('Canvas visible:', canvas?.offsetWidth > 0 && canvas?.offsetHeight > 0);
```

### Force Start Visualization
```javascript
// Force start neural network visualization
if (window.neuralNetwork) {
    window.neuralNetwork.startTraining();
    window.neuralNetwork.updateTrainingMetrics(0.7, 10.0, 0, 10, 0.1);
    console.log('Visualization started!');
} else {
    console.error('Neural network not found. Check if canvas exists.');
}
```

### Check Training State
```javascript
// Check training state variables
console.log('isRealTraining:', typeof isRealTraining !== 'undefined' ? isRealTraining : 'undefined');
console.log('currentEpoch:', typeof currentEpoch !== 'undefined' ? currentEpoch : 'undefined');
console.log('totalEpochs:', typeof totalEpochs !== 'undefined' ? totalEpochs : 'undefined');
console.log('displayedProgress:', typeof displayedProgress !== 'undefined' ? displayedProgress : 'undefined');
console.log('trainingStartTime:', typeof trainingStartTime !== 'undefined' ? trainingStartTime : 'undefined');
```

### Check Status Indicator
```javascript
// Check status indicator
const statusIndicator = document.getElementById('trainingStatusIndicator');
console.log('Status Indicator:', statusIndicator);
console.log('Status Text:', statusIndicator?.querySelector('.status-text')?.textContent);
console.log('Is Active:', statusIndicator?.classList.contains('active'));
```

### Check Progress Elements
```javascript
// Check all progress-related elements
console.log('Progress Elements:');
console.log('  northStarValue:', document.getElementById('northStarValue')?.textContent);
console.log('  currentEpoch:', document.getElementById('currentEpoch')?.textContent);
console.log('  boxLoss:', document.getElementById('boxLoss')?.textContent);
console.log('  clsLoss:', document.getElementById('clsLoss')?.textContent);
console.log('  dflLoss:', document.getElementById('dflLoss')?.textContent);
console.log('  gpuMem:', document.getElementById('gpuMem')?.textContent);
console.log('  instances:', document.getElementById('instances')?.textContent);
console.log('  processingSpeed:', document.getElementById('processingSpeed')?.textContent);
console.log('  map50:', document.getElementById('map50')?.textContent);
console.log('  map5095:', document.getElementById('map5095')?.textContent);
```

### Full Diagnostic
```javascript
// Complete diagnostic check
console.log('=== FULL DIAGNOSTIC ===');
console.log('1. Neural Network:', {
    exists: !!window.neuralNetwork,
    isTraining: window.neuralNetwork?.isTraining,
    isValidating: window.neuralNetwork?.isValidating,
    trainingProgress: window.neuralNetwork?.trainingProgress,
    layers: window.neuralNetwork?.layers?.length || 0,
    connections: window.neuralNetwork?.connections?.length || 0
});
console.log('2. Canvas:', {
    exists: !!document.getElementById('neuralCanvas'),
    width: document.getElementById('neuralCanvas')?.width,
    height: document.getElementById('neuralCanvas')?.height,
    visible: document.getElementById('neuralCanvas')?.offsetWidth > 0
});
console.log('3. Training State:', {
    isRealTraining: typeof isRealTraining !== 'undefined' ? isRealTraining : 'undefined',
    currentEpoch: typeof currentEpoch !== 'undefined' ? currentEpoch : 'undefined',
    totalEpochs: typeof totalEpochs !== 'undefined' ? totalEpochs : 'undefined',
    displayedProgress: typeof displayedProgress !== 'undefined' ? displayedProgress : 'undefined'
});
console.log('4. Status Indicator:', {
    exists: !!document.getElementById('trainingStatusIndicator'),
    text: document.getElementById('trainingStatusIndicator')?.querySelector('.status-text')?.textContent,
    isActive: document.getElementById('trainingStatusIndicator')?.classList.contains('active')
});
console.log('======================');
```

## PowerShell Commands

### Check if App is Running
```powershell
# Check if Uni Trainer process is running
Get-Process | Where-Object {$_.ProcessName -like "*Uni Trainer*" -or $_.MainWindowTitle -like "*Uni Trainer*"} | Select-Object ProcessName, Id, MainWindowTitle
```

### Check Python Process
```powershell
# Check if Python training process is running
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Select-Object ProcessName, Id, StartTime, CPU
```

### Check GPU Usage (if nvidia-smi is available)
```powershell
# Check GPU usage (requires NVIDIA GPU and nvidia-smi)
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
```

### Check Log Files
```powershell
# Check debug log file location
$logFile = "$env:TEMP\uni-trainer-debug.log"
if (Test-Path $logFile) {
    Write-Host "Log file exists: $logFile"
    Get-Content $logFile -Tail 50
} else {
    Write-Host "Log file not found: $logFile"
}
```

### Kill All Training Processes
```powershell
# Stop all Python processes (use with caution!)
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force
```

### Check Port Usage (if needed)
```powershell
# Check if any ports are in use (for debugging IPC)
Get-NetTCPConnection | Where-Object {$_.State -eq "Listen"} | Select-Object LocalAddress, LocalPort, State | Format-Table
```

## Quick Fix Commands (Browser Console)

### Reset Neural Network
```javascript
// Reset and reinitialize neural network
if (window.neuralNetwork) {
    window.neuralNetwork.stopTraining();
    window.neuralNetwork = null;
}
const canvas = document.getElementById('neuralCanvas');
if (canvas && typeof NeuralNetworkVisualization !== 'undefined') {
    window.neuralNetwork = new NeuralNetworkVisualization('neuralCanvas');
    console.log('Neural network reinitialized');
}
```

### Force Update Status Indicator
```javascript
// Manually update status indicator
const indicator = document.getElementById('trainingStatusIndicator');
if (indicator) {
    indicator.classList.add('active');
    indicator.querySelector('.status-text').textContent = 'Training';
    console.log('Status indicator updated');
}
```

### Clear All Progress
```javascript
// Reset all progress displays
document.getElementById('northStarValue').textContent = '0%';
document.getElementById('currentEpoch').textContent = '0/10';
document.getElementById('boxLoss').textContent = '--';
document.getElementById('clsLoss').textContent = '--';
document.getElementById('dflLoss').textContent = '--';
console.log('Progress cleared');
```
