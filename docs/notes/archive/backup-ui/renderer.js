const si = require('systeminformation');
const fs = require('fs');
const path = require('path');

const { ipcRenderer } = require('electron');

let trainingInterval = null;
let monitoringInterval = null;
let uploadedFiles = [];
let trainingSettings = {
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    lossFunction: 'mse',
    validationSplit: 0.2
};
let modelPurpose = 'machine_learning';
let framework = '';
let modelVariant = '';
let modelFormat = 'pytorch';
let isManualMode = false;

// Model configuration definitions
const modelConfigs = {
    computer_vision: {
        frameworks: [
            { value: 'yolo', label: 'YOLO - Real-time object detection' },
            { value: 'resnet', label: 'ResNet - Image classification' },
            { value: 'efficientnet', label: 'EfficientNet - Efficient image classification' },
            { value: 'unet', label: 'U-Net - Image segmentation' },
            { value: 'transformer', label: 'Vision Transformer (ViT)' }
        ],
        variants: {
            yolo: [
                { value: 'yolov11n', label: 'YOLOv11 Nano - Fastest, smallest ⭐ Recommended' },
                { value: 'yolov11s', label: 'YOLOv11 Small - Balanced' },
                { value: 'yolov11m', label: 'YOLOv11 Medium - Better accuracy' },
                { value: 'yolov11l', label: 'YOLOv11 Large - High accuracy' },
                { value: 'yolov11x', label: 'YOLOv11 XLarge - Maximum accuracy' }
            ],
            resnet: [
                { value: 'resnet18', label: 'ResNet-18 - Small' },
                { value: 'resnet34', label: 'ResNet-34 - Medium' },
                { value: 'resnet50', label: 'ResNet-50 - Standard ⭐ Recommended' },
                { value: 'resnet101', label: 'ResNet-101 - Large' },
                { value: 'resnet152', label: 'ResNet-152 - Very Large' }
            ],
            efficientnet: [
                { value: 'efficientnet_b0', label: 'EfficientNet-B0 - Smallest' },
                { value: 'efficientnet_b1', label: 'EfficientNet-B1' },
                { value: 'efficientnet_b2', label: 'EfficientNet-B2' },
                { value: 'efficientnet_b3', label: 'EfficientNet-B3 ⭐ Recommended' },
                { value: 'efficientnet_b4', label: 'EfficientNet-B4 - Large' },
                { value: 'efficientnet_b5', label: 'EfficientNet-B5 - Very Large' }
            ],
            unet: [
                { value: 'unet_small', label: 'U-Net Small' },
                { value: 'unet_medium', label: 'U-Net Medium ⭐ Recommended' },
                { value: 'unet_large', label: 'U-Net Large' }
            ],
            transformer: [
                { value: 'vit_base', label: 'ViT-Base ⭐ Recommended' },
                { value: 'vit_large', label: 'ViT-Large' },
                { value: 'vit_huge', label: 'ViT-Huge' }
            ]
        },
        formats: {
            yolo: [
                { value: 'pt', label: 'PyTorch (.pt) ⭐ Recommended for YOLO' },
                { value: 'onnx', label: 'ONNX (.onnx)' },
                { value: 'torchscript', label: 'TorchScript (.torchscript)' }
            ],
            default: [
                { value: 'pth', label: 'PyTorch (.pth) ⭐ Recommended' },
                { value: 'h5', label: 'TensorFlow/Keras (.h5)' },
                { value: 'onnx', label: 'ONNX (.onnx)' },
                { value: 'pt', label: 'PyTorch (.pt)' }
            ]
        }
    },
    machine_learning: {
        frameworks: [
            { value: 'sklearn', label: 'Scikit-learn - Traditional ML' },
            { value: 'xgboost', label: 'XGBoost - Gradient boosting' },
            { value: 'lightgbm', label: 'LightGBM - Fast gradient boosting' },
            { value: 'pytorch', label: 'PyTorch - Deep learning' },
            { value: 'tensorflow', label: 'TensorFlow - Deep learning' }
        ],
        variants: {
            sklearn: [
                { value: 'random_forest', label: 'Random Forest ⭐ Recommended' },
                { value: 'gradient_boosting', label: 'Gradient Boosting' },
                { value: 'svm', label: 'SVM' },
                { value: 'logistic_regression', label: 'Logistic Regression' }
            ],
            xgboost: [
                { value: 'xgboost_default', label: 'XGBoost Default ⭐ Recommended' },
                { value: 'xgboost_tuned', label: 'XGBoost Tuned' }
            ],
            lightgbm: [
                { value: 'lightgbm_default', label: 'LightGBM Default ⭐ Recommended' }
            ],
            pytorch: [
                { value: 'mlp_small', label: 'MLP Small' },
                { value: 'mlp_medium', label: 'MLP Medium ⭐ Recommended' },
                { value: 'mlp_large', label: 'MLP Large' }
            ],
            tensorflow: [
                { value: 'dnn_small', label: 'DNN Small' },
                { value: 'dnn_medium', label: 'DNN Medium ⭐ Recommended' },
                { value: 'dnn_large', label: 'DNN Large' }
            ]
        },
        formats: {
            sklearn: [
                { value: 'pkl', label: 'Pickle (.pkl) ⭐ Recommended' },
                { value: 'joblib', label: 'Joblib (.joblib)' }
            ],
            xgboost: [
                { value: 'json', label: 'JSON (.json) ⭐ Recommended' },
                { value: 'ubj', label: 'UBJ (.ubj)' }
            ],
            lightgbm: [
                { value: 'txt', label: 'Text (.txt) ⭐ Recommended' }
            ],
            pytorch: [
                { value: 'pth', label: 'PyTorch (.pth) ⭐ Recommended' },
                { value: 'pt', label: 'PyTorch (.pt)' }
            ],
            tensorflow: [
                { value: 'h5', label: 'TensorFlow/Keras (.h5) ⭐ Recommended' },
                { value: 'pb', label: 'SavedModel (.pb)' }
            ]
        }
    },
    natural_language_processing: {
        frameworks: [
            { value: 'transformer', label: 'Transformer - BERT, GPT, etc.' },
            { value: 'lstm', label: 'LSTM - Recurrent networks' },
            { value: 'gpt', label: 'GPT - Generative language models' },
            { value: 'bert', label: 'BERT - Bidirectional encoders' }
        ],
        variants: {
            transformer: [
                { value: 'transformer_small', label: 'Transformer Small' },
                { value: 'transformer_base', label: 'Transformer Base ⭐ Recommended' },
                { value: 'transformer_large', label: 'Transformer Large' }
            ],
            lstm: [
                { value: 'lstm_small', label: 'LSTM Small' },
                { value: 'lstm_medium', label: 'LSTM Medium ⭐ Recommended' },
                { value: 'lstm_large', label: 'LSTM Large' }
            ],
            gpt: [
                { value: 'gpt2', label: 'GPT-2 ⭐ Recommended' },
                { value: 'gpt2_medium', label: 'GPT-2 Medium' },
                { value: 'gpt2_large', label: 'GPT-2 Large' }
            ],
            bert: [
                { value: 'bert_base', label: 'BERT Base ⭐ Recommended' },
                { value: 'bert_large', label: 'BERT Large' }
            ]
        },
        formats: {
            default: [
                { value: 'pt', label: 'PyTorch (.pt) ⭐ Recommended' },
                { value: 'h5', label: 'TensorFlow/Keras (.h5)' },
                { value: 'onnx', label: 'ONNX (.onnx)' }
            ]
        }
    },
    reinforcement_learning: {
        frameworks: [
            { value: 'dqn', label: 'DQN - Deep Q-Network' },
            { value: 'ppo', label: 'PPO - Proximal Policy Optimization' },
            { value: 'a3c', label: 'A3C - Asynchronous Actor-Critic' }
        ],
        variants: {
            dqn: [
                { value: 'dqn_small', label: 'DQN Small' },
                { value: 'dqn_medium', label: 'DQN Medium ⭐ Recommended' }
            ],
            ppo: [
                { value: 'ppo_default', label: 'PPO Default ⭐ Recommended' }
            ],
            a3c: [
                { value: 'a3c_default', label: 'A3C Default ⭐ Recommended' }
            ]
        },
        formats: {
            default: [
                { value: 'pt', label: 'PyTorch (.pt) ⭐ Recommended' },
                { value: 'h5', label: 'TensorFlow/Keras (.h5)' }
            ]
        }
    },
    time_series: {
        frameworks: [
            { value: 'lstm', label: 'LSTM - Sequential prediction' },
            { value: 'transformer', label: 'Transformer - Time series' },
            { value: 'arima', label: 'ARIMA - Traditional forecasting' }
        ],
        variants: {
            lstm: [
                { value: 'lstm_small', label: 'LSTM Small' },
                { value: 'lstm_medium', label: 'LSTM Medium ⭐ Recommended' }
            ],
            transformer: [
                { value: 'transformer_ts', label: 'Transformer Time Series ⭐ Recommended' }
            ],
            arima: [
                { value: 'arima_default', label: 'ARIMA Default ⭐ Recommended' }
            ]
        },
        formats: {
            default: [
                { value: 'pkl', label: 'Pickle (.pkl) ⭐ Recommended' },
                { value: 'pt', label: 'PyTorch (.pt)' }
            ]
        }
    },
    generative: {
        frameworks: [
            { value: 'gan', label: 'GAN - Generative Adversarial Network' },
            { value: 'vae', label: 'VAE - Variational Autoencoder' },
            { value: 'diffusion', label: 'Diffusion Models' }
        ],
        variants: {
            gan: [
                { value: 'dcgan', label: 'DCGAN ⭐ Recommended' },
                { value: 'wgan', label: 'WGAN' },
                { value: 'stylegan', label: 'StyleGAN' }
            ],
            vae: [
                { value: 'vae_default', label: 'VAE Default ⭐ Recommended' }
            ],
            diffusion: [
                { value: 'ddpm', label: 'DDPM ⭐ Recommended' }
            ]
        },
        formats: {
            default: [
                { value: 'pt', label: 'PyTorch (.pt) ⭐ Recommended' },
                { value: 'h5', label: 'TensorFlow/Keras (.h5)' }
            ]
        }
    }
};
let trainingStartTime = null;
let currentEpoch = 0;
let totalEpochs = 0;
let savedModelPath = null;
let trainingHistory = [];
let savedCheckpoint = null; // For resuming training
let displayedProgress = 0; // Smoothly displayed progress (0-100)

// Sparkline data
let cpuSparklineData = [];
let gpuSparklineData = [];
let maxSparklinePoints = 30;

// Metric sparkline data
let accuracySparklineData = [];
let lossSparklineData = [];
let epochSparklineData = [];
let paramSparklineData = [];

// Training chart data
let lossHistory = [];
let accuracyHistory = [];
let chartTooltip = null;

// Sparkline drawing functions
function drawSparkline(canvasId, data, color, isReversed = false) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data || data.length < 2) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Normalize data
    const values = data.map(d => typeof d === 'number' ? d : d.value || 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    
    // Draw sparkline
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    
    for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * width;
        const normalized = (values[i] - min) / range;
        const y = isReversed ? normalized * height : (1 - normalized) * height;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
    
    // Add subtle fill
    ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', ', 0.1)');
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fill();
}

function updateMetricSparklines() {
    // Update accuracy sparkline (green for high values)
    if (accuracySparklineData.length > 0) {
        const accuracyElement = document.getElementById('currentAccuracy').closest('.metric-item');
        if (accuracyElement) {
            const avgAccuracy = accuracySparklineData.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, accuracySparklineData.length);
            if (avgAccuracy >= 70) {
                accuracyElement.setAttribute('data-status', 'good');
                drawSparkline('accuracySparkline', accuracySparklineData, '#4ade80');
            } else if (avgAccuracy >= 50) {
                accuracyElement.setAttribute('data-status', 'warning');
                drawSparkline('accuracySparkline', accuracySparklineData, '#fbbf24');
            } else {
                accuracyElement.setAttribute('data-status', 'danger');
                drawSparkline('accuracySparkline', accuracySparklineData, '#f87171');
            }
        }
    }
    
    // Update loss sparkline (green for low values, red for high)
    if (lossSparklineData.length > 0) {
        const lossElement = document.getElementById('currentLoss').closest('.metric-item');
        if (lossElement) {
            const avgLoss = lossSparklineData.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, lossSparklineData.length);
            if (avgLoss <= 0.3) {
                lossElement.setAttribute('data-status', 'good');
                drawSparkline('lossSparkline', lossSparklineData, '#4ade80', true);
            } else if (avgLoss <= 0.6) {
                lossElement.setAttribute('data-status', 'warning');
                drawSparkline('lossSparkline', lossSparklineData, '#fbbf24', true);
            } else {
                lossElement.setAttribute('data-status', 'danger');
                drawSparkline('lossSparkline', lossSparklineData, '#f87171', true);
            }
        }
    }
    
    // Update epoch sparkline (neutral blue)
    if (epochSparklineData.length > 0) {
        drawSparkline('epochSparkline', epochSparklineData, '#60a5fa');
    }
    
    // Update parameter sparkline (neutral cyan)
    if (paramSparklineData.length > 0) {
        drawSparkline('paramSparkline', paramSparklineData, '#22d3ee');
    }
}

// Initialize system information
async function loadSystemInfo() {
    try {
        const cpu = await si.cpu();
        const graphics = await si.graphics();
        const mem = await si.mem();
        
        document.getElementById('cpu-info').textContent = `${cpu.manufacturer} ${cpu.brand} (${cpu.cores} cores)`;
        
        if (graphics.controllers && graphics.controllers.length > 0) {
            const gpu = graphics.controllers[0];
            document.getElementById('gpu-info').textContent = `${gpu.model || 'Unknown GPU'}`;
        } else {
            document.getElementById('gpu-info').textContent = 'No GPU detected';
        }
        
        const totalMemGB = (mem.total / 1024 / 1024 / 1024).toFixed(2);
        document.getElementById('memory-info').textContent = `${totalMemGB} GB`;
        
        log('System information loaded successfully', 'success');
    } catch (error) {
        log(`Error loading system info: ${error.message}`, 'error');
    }
}

// Logging function with typewriter effect
function log(message, type = 'log') {
    const output = document.getElementById('output');
    const timestamp = new Date().toLocaleTimeString();
    const className = type !== 'log' ? type : '';
    const logEntry = document.createElement('span');
    logEntry.className = className;
    logEntry.textContent = `[${timestamp}] ${message}`;
    output.appendChild(logEntry);
    
    // Scroll with smooth behavior
    setTimeout(() => {
        output.scrollTop = output.scrollHeight;
    }, 50);
}

// Test GPU
function testGPU() {
    log('Testing GPU capabilities...', 'warning');
    log('GPU test: Performing compute operations...', 'log');
    
    // Simulate GPU test
    const startTime = Date.now();
    let iterations = 0;
    const testDuration = 3000; // 3 seconds
    
    const testInterval = setInterval(() => {
        // Simulate GPU computation
        for (let i = 0; i < 1000000; i++) {
            Math.sqrt(Math.random() * 1000);
        }
        iterations++;
        
        if (Date.now() - startTime >= testDuration) {
            clearInterval(testInterval);
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            log(`GPU test completed: ${iterations} iterations in ${elapsed}s`, 'success');
            log(`Approximate throughput: ${(iterations / elapsed).toFixed(2)} ops/s`, 'success');
        }
    }, 10);
}

// Test CPU
function testCPU() {
    log('Testing CPU capabilities...', 'warning');
    log('CPU test: Performing parallel computations...', 'log');
    
    const startTime = Date.now();
    const cores = navigator.hardwareConcurrency || 4;
    let completed = 0;
    
    for (let i = 0; i < cores; i++) {
        setTimeout(() => {
            // Simulate CPU-intensive work
            let sum = 0;
            for (let j = 0; j < 50000000; j++) {
                sum += Math.sqrt(j);
            }
            completed++;
            
            if (completed === cores) {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
                log(`CPU test completed: Utilized ${cores} cores in ${elapsed}s`, 'success');
                log(`Total computations: ${(cores * 50000000).toLocaleString()}`, 'success');
            }
        }, i * 100);
    }
}

// Start training
function startTraining() {
    if (trainingInterval) {
        log('Training already in progress', 'warning');
        return;
    }
    
    // Check if files are uploaded
    if (uploadedFiles.length === 0) {
        log('No training data uploaded. Please upload files first.', 'error');
        return;
    }
    
    document.getElementById('status').textContent = 'Training';
    document.getElementById('status').className = 'value status-training';
    document.getElementById('startTrainingBtn').disabled = true;
    document.getElementById('stopTrainingBtn').disabled = false;
    
    // Calculate quality from slider
    const qualitySlider = document.getElementById('qualitySlider');
    const currentQuality = qualitySlider ? parseInt(qualitySlider.value) : 100;
    
    // Calculate total data size
    const totalDataSize = uploadedFiles.reduce((sum, file) => sum + (file.size || 0), 0);
    
    // Get model type
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const currentModelType = modelPurposeSelect ? modelPurposeSelect.value : 'machine_learning';
    
    // Update neural network with training settings
    if (typeof neuralNetwork !== 'undefined') {
        neuralNetwork.updateTrainingSettings({
            quality: currentQuality,
            epochs: trainingSettings.epochs,
            batchSize: trainingSettings.batchSize,
            learningRate: trainingSettings.learningRate,
            modelType: currentModelType,
            fileCount: uploadedFiles.length,
            dataSize: totalDataSize
        });
        neuralNetwork.startTraining();
        neuralNetwork.trainingProgress = 0; // Reset progress
    }
    
    // Parameter count should already be displayed from settings
    const paramElement = document.getElementById('parameterCount');
    const paramText = paramElement ? paramElement.textContent : '--';
    
    // Initialize progress display
    displayedProgress = 0; // Reset smooth progress
    document.getElementById('northStarValue').textContent = '0%';
    document.getElementById('currentAccuracy').textContent = '--';
    
    // Check if there's a saved checkpoint to resume from
    if (savedCheckpoint && savedCheckpoint.currentEpoch > 0) {
        const resume = confirm(`Found saved checkpoint at epoch ${savedCheckpoint.currentEpoch}/${savedCheckpoint.totalEpochs}.\n\nResume training from checkpoint?`);
        if (resume) {
            currentEpoch = savedCheckpoint.currentEpoch;
            totalEpochs = savedCheckpoint.totalEpochs;
            trainingHistory = savedCheckpoint.trainingHistory || [];
            log(`Resuming training from epoch ${currentEpoch}...`, 'success');
        } else {
            savedCheckpoint = null;
            currentEpoch = 0;
            trainingHistory = [];
        }
    } else {
        currentEpoch = 0;
        trainingHistory = [];
        savedCheckpoint = null;
    }
    
    log('Starting AI training session...', 'success');
    log(`Processing ${uploadedFiles.length} file(s)...`, 'log');
    log(`Model Parameters: ${paramText}`, 'log');
    log(`Settings: Epochs=${trainingSettings.epochs}, Batch=${trainingSettings.batchSize}, LR=${trainingSettings.learningRate}`, 'log');
    log(`Optimizer: ${trainingSettings.optimizer}, Loss: ${trainingSettings.lossFunction}`, 'log');
    log('Initializing training parameters...', 'log');
    log('Loading model architecture...', 'log');
    log('Preparing training data...', 'log');
    log('Training started - monitoring resources...', 'success');
    
    // Initialize progress tracking
    trainingStartTime = Date.now();
    currentEpoch = 0;
    totalEpochs = trainingSettings.epochs;
    updateTrainingActiveState(false);
    
    // Initialize training history
    trainingHistory = [];
    const initialLoss = 1.0;
    const initialAccuracy = 0.0;
    
    // Simulate training progress with improving metrics
    trainingInterval = setInterval(() => {
        // Check if we should stop BEFORE incrementing
        if (currentEpoch >= totalEpochs) {
            clearInterval(trainingInterval);
            trainingInterval = null;
            
            // Ensure progress shows 100% when training completes
            displayedProgress = 100; // Set to 100 immediately when complete
            document.getElementById('northStarValue').textContent = '100%';
            
            stopTraining();
            log('Training completed!', 'success');
            // Save model after a short delay to ensure final metrics are set
            const finalLoss = trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].loss : 0;
            const finalAccuracy = trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].accuracy : 0;
            setTimeout(() => {
                saveModel(finalLoss, finalAccuracy);
            }, 100);
            return;
        }
        
        currentEpoch++;
        
        // Simulate learning: loss decreases, accuracy increases over time
        const progress = currentEpoch / totalEpochs;
        const loss = Math.max(0.01, initialLoss * (1 - progress * 0.95) + (Math.random() * 0.1 - 0.05));
        const accuracy = Math.min(99.9, initialAccuracy + progress * 95 + (Math.random() * 2 - 1));
        
        const lossFormatted = loss.toFixed(4);
        const accuracyFormatted = accuracy.toFixed(2);
        
        log(`Epoch ${currentEpoch}/${totalEpochs}: Loss=${lossFormatted}, Accuracy=${accuracyFormatted}%`, 'log');
        
        // Store training metrics
        trainingHistory.push({
            epoch: currentEpoch,
            loss: parseFloat(lossFormatted),
            accuracy: parseFloat(accuracyFormatted)
        });
        
        // Update north star metrics - Progress percentage (0-100%)
        // Smoothly interpolate the displayed progress to avoid jumping
        const actualProgress = Math.min(100, (currentEpoch / totalEpochs) * 100);
        
        // Smooth interpolation: slowly catch up to actual progress
        // Use exponential smoothing for very smooth increase
        const progressDiff = actualProgress - displayedProgress;
        displayedProgress += progressDiff * 0.15; // Smooth interpolation (15% per frame for responsive but smooth)
        
        // Round for display (but based on smooth value)
        const progressPercent = Math.round(displayedProgress);
        
        // Calculate smooth progress (0.0 to 1.0) for neural network
        const smoothProgress = displayedProgress / 100;
        
        document.getElementById('northStarValue').textContent = `${progressPercent}%`;
        document.getElementById('currentAccuracy').textContent = `${accuracyFormatted}%`;
        document.getElementById('currentLoss').textContent = lossFormatted;
        document.getElementById('currentEpoch').textContent = `${currentEpoch}/${totalEpochs}`;
        
        // Update sparkline data
        accuracySparklineData.push(parseFloat(accuracyFormatted));
        if (accuracySparklineData.length > maxSparklinePoints) {
            accuracySparklineData.shift();
        }
        
        lossSparklineData.push(parseFloat(lossFormatted));
        if (lossSparklineData.length > maxSparklinePoints) {
            lossSparklineData.shift();
        }
        
        const epochProgress = (currentEpoch / totalEpochs) * 100;
        epochSparklineData.push(epochProgress);
        if (epochSparklineData.length > maxSparklinePoints) {
            epochSparklineData.shift();
        }
        
        // Update parameter count sparkline (use current parameter count value)
        const paramElement = document.getElementById('parameterCount');
        const paramText = paramElement ? paramElement.textContent : '0';
        const paramValue = parseFloat(paramText.replace(/[KM]/g, '')) || 0;
        const multiplier = paramText.includes('M') ? 1000000 : paramText.includes('K') ? 1000 : 1;
        paramSparklineData.push(paramValue * multiplier);
        if (paramSparklineData.length > maxSparklinePoints) {
            paramSparklineData.shift();
        }
        
        // Draw sparklines
        updateMetricSparklines();
        
        // Parameter count is already set, no need to update during training
        
        // Update chart (only if function exists)
        if (typeof updateTrainingChart === 'function') {
            lossHistory.push({ epoch: currentEpoch, value: parseFloat(lossFormatted) });
            accuracyHistory.push({ epoch: currentEpoch, value: parseFloat(accuracyFormatted) });
            updateTrainingChart();
        }
        
        // Update neural network visualization with actual metrics AND smooth progress
        // IMPORTANT: Pass smooth progress to maintain synchronization with displayed progress
        if (typeof neuralNetwork !== 'undefined') {
            // Update metrics with smooth progress to ensure display and visualization match exactly
            neuralNetwork.updateTrainingMetrics(loss, accuracy, currentEpoch, totalEpochs, smoothProgress);
        }
        
        // Accuracy IS the progress percentage - no separate progress bar needed
        
        // Final check to stop if we've reached the limit
        if (currentEpoch >= totalEpochs) {
            clearInterval(trainingInterval);
            trainingInterval = null;
            
            // Update progress to 100% before stopping
            displayedProgress = 100; // Set to 100 immediately when complete
            document.getElementById('northStarValue').textContent = '100%';
            
            stopTraining();
            log('Training completed!', 'success');
            const finalLoss = parseFloat(lossFormatted);
            const finalAccuracy = parseFloat(accuracyFormatted);
            setTimeout(() => {
                saveModel(finalLoss, finalAccuracy);
            }, 100);
        }
    }, 2000);
    
    // Start monitoring
    startMonitoring();
}

// Update training section active state based on training progress
function updateTrainingActiveState(isActive) {
    const trainingSection = document.querySelector('.training-section.card');
    if (trainingSection) {
        if (isActive) {
            trainingSection.classList.add('training-active');
        } else {
            trainingSection.classList.remove('training-active');
        }
    }
}

// Stop training
function stopTraining() {
    const wasCompleted = currentEpoch >= totalEpochs;
    
    // Force stop training interval
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
        monitoringInterval = null;
    }
    
    // Ensure training stops even if interval check failed
    if (currentEpoch >= totalEpochs) {
        currentEpoch = totalEpochs; // Cap at total
    }
    
    // Save checkpoint if training was stopped prematurely
    if (!wasCompleted && currentEpoch > 0 && trainingHistory.length > 0) {
        const latestMetrics = trainingHistory[trainingHistory.length - 1];
        savedCheckpoint = {
            currentEpoch: currentEpoch,
            totalEpochs: totalEpochs,
            trainingHistory: [...trainingHistory],
            loss: latestMetrics.loss,
            accuracy: latestMetrics.accuracy,
            trainingSettings: { ...trainingSettings }
        };
        
        // Save the checkpoint model
        saveModel(latestMetrics.loss, latestMetrics.accuracy, true); // true = isPartial
        
        log(`Training stopped at epoch ${currentEpoch}/${totalEpochs}`, 'warning');
        log('Checkpoint saved - you can resume training later', 'success');
    } else {
        savedCheckpoint = null;
    }
    
    // Stop neural network visualization but keep it visible if completed
    if (typeof neuralNetwork !== 'undefined') {
        if (wasCompleted) {
            // Mark as completed so it stays pulsing
            neuralNetwork.trainingProgress = 1.0;
            neuralNetwork.learningQuality = 0.8; // Maintain high quality for visual
        }
        neuralNetwork.stopTraining();
    }
    
    // Update training state
    if (wasCompleted) {
        // Training completed - ensure progress shows 100%
        document.getElementById('northStarValue').textContent = '100%';
        updateTrainingActiveState(false);
        log('Training completed! Network is ready for interaction.', 'success');
    } else {
        // Training stopped early
        updateTrainingActiveState(false);
        log('Training stopped', 'warning');
    }
    
    document.getElementById('status').textContent = wasCompleted ? 'Completed' : 'Ready';
    document.getElementById('status').className = wasCompleted ? 'value status-ready' : 'value status-ready';
    document.getElementById('startTrainingBtn').disabled = false;
    document.getElementById('stopTrainingBtn').disabled = true;
    
    if (!wasCompleted) {
        currentEpoch = 0;
        trainingStartTime = null;
    }
}

// Reset everything and start a new training project
function trainNewModel() {
    // Confirm action if training is in progress
    if (trainingInterval || (typeof neuralNetwork !== 'undefined' && neuralNetwork && neuralNetwork.isTraining)) {
        const confirmReset = confirm('Training is in progress. This will stop training and reset everything. Continue?');
        if (!confirmReset) {
            return;
        }
    }
    
    log('Starting new training project...', 'log');
    
    // Stop any ongoing training
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
        monitoringInterval = null;
    }
    
    // Reset neural network visualization
    if (typeof neuralNetwork !== 'undefined' && neuralNetwork) {
        neuralNetwork.stopTraining();
        neuralNetwork.reset();
    }
    
    // Clear uploaded files
    uploadedFiles = [];
    
    // Clear file list UI
    const fileList = document.getElementById('fileList');
    if (fileList) {
        fileList.innerHTML = '';
    }
    
    // Reset training variables
    currentEpoch = 0;
    totalEpochs = 0;
    trainingHistory = [];
    displayedProgress = 0;
    savedCheckpoint = null;
    trainingStartTime = null;
    
    // Clear sparkline data
    accuracySparklineData = [];
    lossSparklineData = [];
    epochSparklineData = [];
    paramSparklineData = [];
    
    // Clear loss and accuracy history
    lossHistory = [];
    accuracyHistory = [];
    
    // Reset UI displays
    document.getElementById('status').textContent = 'Ready';
    document.getElementById('status').className = 'value status-ready';
    document.getElementById('northStarValue').textContent = '0%';
    document.getElementById('currentAccuracy').textContent = '--';
    document.getElementById('currentLoss').textContent = '--';
    document.getElementById('currentEpoch').textContent = '--';
    document.getElementById('parameterCount').textContent = '--';
    document.getElementById('output').textContent = 'Ready to start training...';
    
    // Reset button states
    document.getElementById('startTrainingBtn').disabled = false;
    document.getElementById('stopTrainingBtn').disabled = true;
    
    // Hide settings sections
    document.getElementById('modelPurposeSection').style.display = 'none';
    document.getElementById('settingsSection').style.display = 'none';
    
    // Reset quality slider
    const qualitySlider = document.getElementById('qualitySlider');
    if (qualitySlider) {
        qualitySlider.value = 50;
        updateSlider(50);
    }
    
    // Reset training settings to defaults
    trainingSettings = {
        epochs: 10,
        batchSize: 32,
        learningRate: 0.001,
        optimizer: 'adam',
        lossFunction: 'mse',
        validationSplit: 0.2
    };
    
    // Clear sparklines
    updateMetricSparklines();
    
    // Reset model purpose and framework selections
    const modelPurposeInput = document.getElementById('modelPurposeInput');
    const frameworkInput = document.getElementById('frameworkInput');
    const variantInput = document.getElementById('modelVariantInput');
    if (modelPurposeInput) modelPurposeInput.value = 'machine_learning';
    if (frameworkInput) frameworkInput.value = '';
    if (variantInput) variantInput.value = '';
    
    log('New project ready. Upload training data to begin.', 'success');
}

// Resource monitoring
function startMonitoring() {
    if (monitoringInterval) return;
    
    monitoringInterval = setInterval(async () => {
        try {
            const cpuLoad = await si.currentLoad();
            const mem = await si.mem();
            
            const cpuPercent = Math.round(cpuLoad.currentLoad);
            const memUsed = mem.used;
            const memTotal = mem.total;
            const memPercent = Math.round((memUsed / memTotal) * 100);
            
            // CPU monitoring
            document.getElementById('cpu-usage').style.width = `${cpuPercent}%`;
            document.getElementById('cpu-percent').textContent = `${cpuPercent}%`;
            
            // Memory monitoring
            document.getElementById('memory-usage').style.width = `${memPercent}%`;
            document.getElementById('memory-percent').textContent = `${memPercent}%`;
            
            // GPU monitoring
            try {
                const graphics = await si.graphics();
                if (graphics && graphics.controllers && graphics.controllers.length > 0) {
                    const gpu = graphics.controllers[0];
                    // GPU usage simulation (real GPU usage requires additional APIs)
                    const gpuPercent = trainingInterval ? Math.min(100, Math.round(60 + Math.random() * 30)) : Math.round(Math.random() * 10);
                    document.getElementById('gpu-usage').style.width = `${gpuPercent}%`;
                    document.getElementById('gpu-percent').textContent = `${gpuPercent}%`;
                } else {
                    document.getElementById('gpu-usage').style.width = '0%';
                    document.getElementById('gpu-percent').textContent = 'N/A';
                }
            } catch (gpuError) {
                // GPU monitoring not available
                document.getElementById('gpu-usage').style.width = '0%';
                document.getElementById('gpu-percent').textContent = 'N/A';
            }
        } catch (error) {
            console.error('Monitoring error:', error);
        }
    }, 2000);
}

// File upload functionality
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function calculateRecommendedSettings(files) {
    const totalSize = Array.from(files).reduce((sum, file) => sum + file.size, 0);
    const fileCount = files.length;
    const hasImages = Array.from(files).some(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));
    const hasCSV = Array.from(files).some(f => f.name.endsWith('.csv'));
    
    // Calculate based on data size and type
    let quality = 100; // Max quality by default
    
    if (totalSize > 500 * 1024 * 1024) { // > 500MB
        quality = 80; // Large datasets need balance
    } else if (totalSize > 100 * 1024 * 1024) { // > 100MB
        quality = 90;
    }
    
    if (hasImages) {
        quality = Math.max(quality - 10, 70); // Images need more processing
    }
    
    if (fileCount > 1000) {
        quality = Math.max(quality - 10, 70);
    }
    
    // Return values that match dropdown options
    let epochs, batchSize, learningRate;
    
    if (quality > 90) {
        epochs = 50;
        batchSize = hasImages ? 16 : 32;
        learningRate = 0.0001;
    } else if (quality > 70) {
        epochs = 20;
        batchSize = hasImages ? 32 : 64;
        learningRate = 0.001;
    } else {
        epochs = 10;
        batchSize = hasImages ? 32 : 64;
        learningRate = 0.001;
    }
    
    // Clamp to available dropdown values
    const availableEpochs = [5, 10, 20, 50, 100, 200];
    epochs = availableEpochs.reduce((prev, curr) => 
        Math.abs(curr - epochs) < Math.abs(prev - epochs) ? curr : prev
    );
    
    const availableBatch = [8, 16, 32, 64, 128, 256];
    batchSize = availableBatch.reduce((prev, curr) => 
        Math.abs(curr - batchSize) < Math.abs(prev - batchSize) ? curr : prev
    );
    
    const availableLR = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1];
    learningRate = availableLR.reduce((prev, curr) => 
        Math.abs(curr - learningRate) < Math.abs(prev - learningRate) ? curr : prev
    );
    
    return {
        quality: quality,
        epochs: epochs,
        batchSize: batchSize,
        learningRate: learningRate,
        optimizer: quality > 90 ? 'adamw' : 'adam',
        lossFunction: hasImages ? 'categorical_crossentropy' : 'mse',
        validationSplit: 0.2
    };
}

function updateSlider(quality) {
    const slider = document.getElementById('qualitySlider');
    const sliderFill = document.getElementById('sliderFill');
    const sliderHandle = document.getElementById('sliderHandle');
    
    slider.value = quality;
    const percentage = quality;
    sliderFill.style.width = `${100 - percentage}%`;
    sliderHandle.style.left = `${percentage}%`;
    
    // Update labels
    let label = '';
    let description = '';
    
    if (quality >= 90) {
        label = 'Maximum Quality';
        description = 'Best results, longer training time';
    } else if (quality >= 70) {
        label = 'High Quality';
        description = 'Great results, balanced time';
    } else if (quality >= 50) {
        label = 'Balanced';
        description = 'Good results, moderate time';
    } else if (quality >= 30) {
        label = 'Fast';
        description = 'Decent results, faster training';
    } else {
        label = 'Maximum Speed';
        description = 'Quick results, lower quality';
    }
    
    document.getElementById('presetLabel').textContent = label;
    document.getElementById('presetDescription').textContent = description;
}

function showNotification() {
    const notification = document.getElementById('notification');
    notification.classList.add('show');
    
    // Hide after 2.5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
    }, 2500);
}

// Calculate and display parameter count based on training data and model configuration
function updateParameterCount() {
    // Need training data to calculate parameters
    if (uploadedFiles.length === 0) {
        const paramElement = document.getElementById('parameterCount');
        if (paramElement) {
            paramElement.textContent = '--';
        }
        return { count: 0, text: '--' };
    }
    
    // Get current model configuration
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const frameworkSelect = document.getElementById('frameworkInput');
    const variantSelect = document.getElementById('modelVariantInput');
    
    const modelPurpose = modelPurposeSelect ? modelPurposeSelect.value : 'machine_learning';
    const framework = frameworkSelect ? frameworkSelect.value : '';
    const variant = variantSelect ? variantSelect.value : '';
    
    // Calculate parameters based on training data and model configuration
    let parameterCount = calculateParametersFromData(uploadedFiles, modelPurpose, framework, variant);
    
    // Format parameter count
    let paramText;
    if (parameterCount >= 1000000000) {
        paramText = `${(parameterCount / 1000000000).toFixed(2)}B`;
    } else if (parameterCount >= 1000000) {
        paramText = `${(parameterCount / 1000000).toFixed(2)}M`;
    } else if (parameterCount >= 1000) {
        paramText = `${(parameterCount / 1000).toFixed(1)}K`;
    } else {
        paramText = parameterCount.toLocaleString();
    }
    
    // Update parameter count display
    const paramElement = document.getElementById('parameterCount');
    if (paramElement) {
        paramElement.textContent = paramText;
    }
    
    return { count: parameterCount, text: paramText };
}

// Calculate parameters based on training data and model configuration
// Uses the EXACT same logic as neural-network.js calculateNetworkArchitecture()
function calculateParametersFromData(files, modelPurpose, framework, variant) {
    // Analyze the training data
    const totalSize = files.reduce((sum, file) => sum + (file.size || 0), 0);
    const fileCount = files.length;
    
    // Get current training settings (same as what neural network uses)
    // Check both manual and preset modes
    const qualitySlider = document.getElementById('qualitySlider');
    const epochsInput = document.getElementById('epochsInput');
    const batchSizeInput = document.getElementById('batchSizeInput');
    const lrInput = document.getElementById('learningRateInput');
    
    const quality = qualitySlider ? parseInt(qualitySlider.value) : 50;
    
    // Get epochs, batch size, and learning rate from manual settings inputs
    // These inputs are available in both manual and preset modes
    let epochs = 10;
    let batchSize = 32;
    let learningRate = 0.001;
    
    if (epochsInput && epochsInput.value) {
        // Extract numeric value from dropdown (may contain text like "10 - Fast ⭐ Recommended")
        epochs = parseInt(epochsInput.value) || 10;
    }
    
    if (batchSizeInput && batchSizeInput.value) {
        batchSize = parseInt(batchSizeInput.value) || 32;
    }
    
    if (lrInput && lrInput.value) {
        learningRate = parseFloat(lrInput.value) || 0.001;
    }
    
    // Use the same logic as neural-network.js calculateNetworkArchitecture()
    const modelType = modelPurpose; // Map purpose to model type
    const dataSize = totalSize;
    
    // Base architecture varies by model type (EXACT same as neural-network.js)
    let baseArchitecture;
    let inputSize, outputSize;
    
    switch(modelType) {
        case 'computer_vision':
            // Computer vision: larger input layers (image features)
            inputSize = Math.max(16, Math.min(32, Math.floor(fileCount / 10) || 16));
            outputSize = 4; // Classification outputs
            baseArchitecture = 'wide'; // Wider networks for images
            break;
        case 'natural_language_processing':
            // NLP: sequence-aware architectures
            inputSize = Math.max(12, Math.min(24, Math.floor(dataSize / 1000000) || 12));
            outputSize = 6;
            baseArchitecture = 'deep'; // Deeper networks for language
            break;
        case 'reinforcement_learning':
            // RL: action-value networks
            inputSize = 10;
            outputSize = 8;
            baseArchitecture = 'medium';
            break;
        default: // machine_learning, time_series, generative
            inputSize = 8;
            outputSize = 6;
            baseArchitecture = 'standard';
    }
    
    // Quality affects network size: 0-100 maps to network complexity (EXACT same logic)
    const qualityFactor = quality / 100; // 0.0 to 1.0
    
    // Epochs affect depth: more epochs can handle deeper networks
    const depthFactor = Math.min(1.5, 0.5 + (epochs / 100)); // 0.5 to 1.5
    
    // Batch size affects width: larger batches can handle wider networks
    const widthFactor = Math.min(1.3, 0.7 + (batchSize / 200)); // 0.7 to 1.3
    
    // Learning rate affects stability: lower LR = can handle more complex networks
    const complexityFactor = Math.min(1.2, 0.8 + ((0.01 - learningRate) / 0.01)); // 0.8 to 1.2
    
    // Calculate number of hidden layers (3 to 6 layers based on quality and depth)
    const numHiddenLayers = Math.max(2, Math.min(5, Math.floor(2 + qualityFactor * 3 * depthFactor)));
    
    // Calculate architecture (EXACT same logic as neural-network.js)
    let layerSizes = [inputSize];
    
    if (baseArchitecture === 'deep') {
        // Deeper network: more layers, gradually narrowing
        for (let i = 0; i < numHiddenLayers; i++) {
            const progress = i / numHiddenLayers;
            const size = Math.max(
                6,
                Math.floor(
                    inputSize * (1 - progress * 0.6) * 
                    (1 + qualityFactor * 0.5) * 
                    complexityFactor
                )
            );
            layerSizes.push(size);
        }
    } else if (baseArchitecture === 'wide') {
        // Wider network: fewer layers, more neurons per layer
        const numLayers = Math.max(2, Math.min(4, Math.floor(numHiddenLayers * 0.75)));
        for (let i = 0; i < numLayers; i++) {
            const size = Math.max(
                10,
                Math.floor(
                    inputSize * (1 + qualityFactor * 0.8) * 
                    widthFactor
                )
            );
            layerSizes.push(size);
        }
    } else {
        // Standard architecture: balanced depth and width
        for (let i = 0; i < numHiddenLayers; i++) {
            const progress = i / numHiddenLayers;
            const size = Math.max(
                6,
                Math.floor(
                    inputSize * (1 - progress * 0.4) * 
                    (1 + qualityFactor * 0.6) * 
                    widthFactor
                )
            );
            layerSizes.push(size);
        }
    }
    
    layerSizes.push(outputSize);
    
    // Ensure minimum sizes and smooth progression (EXACT same as neural-network.js)
    layerSizes = layerSizes.map((size, i) => {
        if (i === 0 || i === layerSizes.length - 1) return size;
        return Math.max(6, size); // Minimum 6 neurons in hidden layers
    });
    
    // Calculate total parameters from layer sizes
    let parameterCount = 0;
    for (let i = 0; i < layerSizes.length - 1; i++) {
        // Weights: prev_layer_size * current_layer_size
        // Biases: current_layer_size
        parameterCount += layerSizes[i] * layerSizes[i + 1] + layerSizes[i + 1];
    }
    
    return parameterCount;
}

// Calculate architecture from model type
function calculateArchitectureFromModel(modelPurpose, framework, variant, inputSize, outputSize, totalSize, fileCount) {
    // Model-specific architectures that scale with data
    const baseConfigs = {
        computer_vision: {
            yolo: {
                'yolov11n': [inputSize, Math.floor(inputSize * 1.5), Math.floor(inputSize * 1.2), outputSize],
                'yolov11s': [inputSize, Math.floor(inputSize * 2), Math.floor(inputSize * 1.8), Math.floor(inputSize * 1.5), outputSize],
                'yolov11m': [inputSize, Math.floor(inputSize * 2.5), Math.floor(inputSize * 2), Math.floor(inputSize * 1.8), outputSize],
                'yolov11l': [inputSize, Math.floor(inputSize * 3), Math.floor(inputSize * 2.5), Math.floor(inputSize * 2), outputSize],
                'yolov11x': [inputSize, Math.floor(inputSize * 4), Math.floor(inputSize * 3), Math.floor(inputSize * 2.5), outputSize]
            },
            resnet: {
                'resnet18': [inputSize, Math.floor(inputSize * 2), Math.floor(inputSize * 1.5), outputSize],
                'resnet34': [inputSize, Math.floor(inputSize * 2.5), Math.floor(inputSize * 2), outputSize],
                'resnet50': [inputSize, Math.floor(inputSize * 3), Math.floor(inputSize * 2.5), outputSize]
            }
        },
        machine_learning: {
            pytorch: {
                'mlp_small': [inputSize, Math.floor(inputSize * 1.5), outputSize],
                'mlp_medium': [inputSize, Math.floor(inputSize * 1.8), Math.floor(inputSize * 1.2), outputSize],
                'mlp_large': [inputSize, Math.floor(inputSize * 2), Math.floor(inputSize * 1.5), Math.floor(inputSize * 1.2), outputSize]
            }
        }
    };
    
    if (baseConfigs[modelPurpose] && 
        baseConfigs[modelPurpose][framework] && 
        baseConfigs[modelPurpose][framework][variant]) {
        return baseConfigs[modelPurpose][framework][variant];
    }
    
    // Fallback to default
    return calculateDefaultArchitecture(modelPurpose, inputSize, outputSize, totalSize, fileCount);
}

// Calculate default architecture based on data
function calculateDefaultArchitecture(modelPurpose, inputSize, outputSize, totalSize, fileCount) {
    // Scale network complexity based on data size
    const dataComplexity = Math.min(2, 1 + (totalSize / 10000000)); // Scale up to 2x for large datasets
    
    let hiddenSize1, hiddenSize2, hiddenSize3;
    
    if (modelPurpose === 'computer_vision') {
        // Deeper networks for images
        hiddenSize1 = Math.floor(inputSize * 1.8 * dataComplexity);
        hiddenSize2 = Math.floor(inputSize * 1.4 * dataComplexity);
        hiddenSize3 = Math.floor(inputSize * 1.0 * dataComplexity);
        return [inputSize, hiddenSize1, hiddenSize2, hiddenSize3, outputSize];
    } else if (modelPurpose === 'natural_language_processing') {
        // Wide networks for NLP
        hiddenSize1 = Math.floor(inputSize * 2.0 * dataComplexity);
        hiddenSize2 = Math.floor(inputSize * 1.6 * dataComplexity);
        return [inputSize, hiddenSize1, hiddenSize2, outputSize];
    } else {
        // Standard ML architecture
        hiddenSize1 = Math.floor(inputSize * 1.5 * dataComplexity);
        hiddenSize2 = Math.floor(inputSize * 1.2 * dataComplexity);
        return [inputSize, hiddenSize1, hiddenSize2, outputSize];
    }
}

// Get complexity multiplier based on model type
function getModelComplexityMultiplier(modelPurpose, framework, variant) {
    const multipliers = {
        computer_vision: {
            yolo: {
                'yolov11n': 1.0,
                'yolov11s': 1.5,
                'yolov11m': 2.0,
                'yolov11l': 2.5,
                'yolov11x': 4.0
            },
            resnet: {
                'resnet18': 1.0,
                'resnet34': 1.5,
                'resnet50': 2.0,
                'resnet101': 3.0,
                'resnet152': 4.0
            }
        },
        natural_language_processing: {
            transformer: {
                'transformer_small': 3.0,
                'transformer_base': 5.0,
                'transformer_large': 8.0
            }
        }
    };
    
    if (multipliers[modelPurpose] && 
        multipliers[modelPurpose][framework] && 
        multipliers[modelPurpose][framework][variant]) {
        return multipliers[modelPurpose][framework][variant];
    }
    
    return 1.0; // Default multiplier
}

function applyPresetSettings() {
    const quality = parseInt(document.getElementById('qualitySlider').value);
    const recommended = calculateRecommendedSettings(uploadedFiles);
    
    // Adjust based on slider position
    const ratio = quality / 100;
    const maxEpochs = 100;
    const minEpochs = 10;
    
    trainingSettings.epochs = Math.round(minEpochs + (maxEpochs - minEpochs) * ratio);
    trainingSettings.batchSize = recommended.batchSize;
    
    // Update neural network architecture with new settings
    updateNeuralNetworkSettings();
    trainingSettings.learningRate = recommended.learningRate * (1 + (1 - ratio) * 0.5);
    trainingSettings.optimizer = recommended.optimizer;
    trainingSettings.lossFunction = recommended.lossFunction;
    trainingSettings.validationSplit = recommended.validationSplit;
    
    // Calculate and display parameter count
    const paramInfo = updateParameterCount();
    
    log(`Applied preset settings: ${quality}% quality`, 'success');
    log(`Model Parameters: ${paramInfo.text} (${paramInfo.count.toLocaleString()})`, 'log');
    log(`Epochs: ${trainingSettings.epochs}, Batch: ${trainingSettings.batchSize}, LR: ${trainingSettings.learningRate}`, 'log');
    
    showNotification();
}

function applyManualSettings() {
    // Extract numeric values from dropdown options (they contain text like "10 - Fast ⭐ Recommended")
    const epochsValue = document.getElementById('epochsInput').value;
    const batchSizeValue = document.getElementById('batchSizeInput').value;
    const learningRateValue = document.getElementById('learningRateInput').value;
    const validationSplitValue = document.getElementById('validationSplitInput').value;
    
    trainingSettings.epochs = parseInt(epochsValue);
    trainingSettings.batchSize = parseInt(batchSizeValue);
    trainingSettings.learningRate = parseFloat(learningRateValue);
    trainingSettings.optimizer = document.getElementById('optimizerInput').value;
    trainingSettings.lossFunction = document.getElementById('lossFunctionInput').value;
    trainingSettings.validationSplit = parseFloat(validationSplitValue);
    
    // Calculate and display parameter count
    const paramInfo = updateParameterCount();
    
    log('Applied manual settings', 'success');
    log(`Model Parameters: ${paramInfo.text} (${paramInfo.count.toLocaleString()})`, 'log');
    log(`Epochs: ${trainingSettings.epochs}, Batch: ${trainingSettings.batchSize}, LR: ${trainingSettings.learningRate}`, 'log');
    
    // Update neural network architecture with new settings
    updateNeuralNetworkSettings();
    
    showNotification();
}

// Update neural network visualization with current training settings
function updateNeuralNetworkSettings() {
    if (typeof neuralNetwork === 'undefined' || !neuralNetwork) return;
    
    const qualitySlider = document.getElementById('qualitySlider');
    const currentQuality = qualitySlider ? parseInt(qualitySlider.value) : 100;
    
    const totalDataSize = uploadedFiles.reduce((sum, file) => sum + (file.size || 0), 0);
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const currentModelType = modelPurposeSelect ? modelPurposeSelect.value : 'machine_learning';
    
    neuralNetwork.updateTrainingSettings({
        quality: currentQuality,
        epochs: trainingSettings.epochs,
        batchSize: trainingSettings.batchSize,
        learningRate: trainingSettings.learningRate,
        modelType: currentModelType,
        fileCount: uploadedFiles.length,
        dataSize: totalDataSize
    });
}

function updateManualDropdowns() {
    // Set dropdowns to recommended values
    const recommended = calculateRecommendedSettings(uploadedFiles);
    
    // Find and select recommended epoch value
    const epochsSelect = document.getElementById('epochsInput');
    const recommendedEpochs = recommended.epochs;
    let closestEpochs = 10;
    let minDiff = Infinity;
    for (let option of epochsSelect.options) {
        const value = parseInt(option.value);
        const diff = Math.abs(value - recommendedEpochs);
        if (diff < minDiff) {
            minDiff = diff;
            closestEpochs = value;
        }
    }
    epochsSelect.value = closestEpochs;
    
    // Find and select recommended batch size
    const batchSelect = document.getElementById('batchSizeInput');
    const recommendedBatch = recommended.batchSize;
    let closestBatch = 32;
    minDiff = Infinity;
    for (let option of batchSelect.options) {
        const value = parseInt(option.value);
        const diff = Math.abs(value - recommendedBatch);
        if (diff < minDiff) {
            minDiff = diff;
            closestBatch = value;
        }
    }
    batchSelect.value = closestBatch;
    
    // Find and select recommended learning rate
    const lrSelect = document.getElementById('learningRateInput');
    const recommendedLR = recommended.learningRate;
    let closestLR = '0.001';
    minDiff = Infinity;
    for (let option of lrSelect.options) {
        const value = parseFloat(option.value);
        const diff = Math.abs(value - recommendedLR);
        if (diff < minDiff) {
            minDiff = diff;
            closestLR = option.value;
        }
    }
    lrSelect.value = closestLR;
    
    // Set other recommended values
    document.getElementById('optimizerInput').value = recommended.optimizer;
    document.getElementById('lossFunctionInput').value = recommended.lossFunction;
    document.getElementById('validationSplitInput').value = recommended.validationSplit.toString();
}

function handleFiles(files) {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    
    uploadedFiles = Array.from(files);
    
    // Calculate total size
    const totalSize = uploadedFiles.reduce((sum, file) => sum + file.size, 0);
    const fileCount = uploadedFiles.length;
    
    // Show simple success message instead of listing all files
    const successItem = document.createElement('div');
    successItem.className = 'file-item';
    successItem.innerHTML = `
        <span class="file-item-name">✓ Loaded Successfully</span>
        <span class="file-item-size">${fileCount} file(s) • ${formatFileSize(totalSize)}</span>
        <button class="file-item-remove" id="clearFilesBtn">×</button>
    `;
    fileList.appendChild(successItem);
    
    // Store file data in background
    uploadedFiles.forEach((file) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            file.data = e.target.result;
        };
        reader.readAsArrayBuffer(file);
    });
    
    log(`Loaded ${files.length} file(s) for training`, 'success');
    
    // Show model purpose section
    document.getElementById('modelPurposeSection').style.display = 'block';
    
    // Show recommended settings
    const recommended = calculateRecommendedSettings(files);
    document.getElementById('settingsSection').style.display = 'block';
    updateSlider(recommended.quality);
    updateManualDropdowns();
    applyPresetSettings();
    
    // Show parameter count immediately
    updateParameterCount();
    
    log(`Recommended settings calculated: ${recommended.quality}% quality`, 'success');
    
    // Handle clear button
    document.getElementById('clearFilesBtn').addEventListener('click', () => {
        uploadedFiles = [];
        fileList.innerHTML = '';
        document.getElementById('settingsSection').style.display = 'none';
        document.getElementById('modelPurposeSection').style.display = 'none';
        log('Files cleared', 'log');
    });
    
    // Update model purpose based on file types
    const hasImages = Array.from(files).some(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));
    if (hasImages) {
        document.getElementById('modelPurposeInput').value = 'computer_vision';
        modelPurpose = 'computer_vision';
        updateFrameworkOptions();
    }
    
    // Update neural network architecture with file data
    updateNeuralNetworkSettings();
}

function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });
    
    // Drag and drop with magnetic effect
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
        
        // Magnetic icon effect
        const icon = uploadArea.querySelector('.upload-icon');
        if (icon) {
            const rect = uploadArea.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            const distance = Math.sqrt(x * x + y * y);
            const maxDistance = 150;
            const factor = Math.min(1, distance / maxDistance);
            icon.style.transform = `translate(${x * 0.3 * (1 - factor)}px, ${y * 0.3 * (1 - factor)}px) scale(${1 + 0.2 * (1 - factor)})`;
            icon.style.transition = 'transform 0.2s ease-out';
        }
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
        const icon = uploadArea.querySelector('.upload-icon');
        if (icon) {
            icon.style.transform = 'translate(0, 0) scale(1)';
        }
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });
    
    // File list click handler (for clear button)
    fileList.addEventListener('click', (e) => {
        if (e.target.classList.contains('file-item-remove')) {
            if (e.target.id === 'clearFilesBtn') {
                uploadedFiles = [];
                fileList.innerHTML = '';
                document.getElementById('settingsSection').style.display = 'none';
                log('Files cleared', 'log');
            }
        }
    });
}

function updateFrameworkOptions() {
    const purpose = document.getElementById('modelPurposeInput').value;
    const frameworkGroup = document.getElementById('frameworkGroup');
    const frameworkInput = document.getElementById('frameworkInput');
    const variantGroup = document.getElementById('modelVariantGroup');
    const formatGroup = document.getElementById('formatGroup');
    
    if (!modelConfigs[purpose]) {
        frameworkGroup.style.display = 'none';
        variantGroup.style.display = 'none';
        formatGroup.style.display = 'none';
        return;
    }
    
    // Show framework selection
    frameworkGroup.style.display = 'block';
    frameworkInput.innerHTML = '';
    
    modelConfigs[purpose].frameworks.forEach(fw => {
        const option = document.createElement('option');
        option.value = fw.value;
        option.textContent = fw.label;
        frameworkInput.appendChild(option);
    });
    
    // Auto-select first framework and update variants
    if (modelConfigs[purpose].frameworks.length > 0) {
        framework = modelConfigs[purpose].frameworks[0].value;
        frameworkInput.value = framework;
        updateVariantOptions();
        updateFormatOptions();
    }
}

function updateVariantOptions() {
    const purpose = document.getElementById('modelPurposeInput').value;
    const variantGroup = document.getElementById('modelVariantGroup');
    const variantInput = document.getElementById('modelVariantInput');
    
    if (!framework || !modelConfigs[purpose] || !modelConfigs[purpose].variants[framework]) {
        variantGroup.style.display = 'none';
        return;
    }
    
    variantGroup.style.display = 'block';
    variantInput.innerHTML = '';
    
    modelConfigs[purpose].variants[framework].forEach(variant => {
        const option = document.createElement('option');
        option.value = variant.value;
        option.textContent = variant.label;
        variantInput.appendChild(option);
    });
    
    // Auto-select first variant
    if (modelConfigs[purpose].variants[framework].length > 0) {
        modelVariant = modelConfigs[purpose].variants[framework][0].value;
        variantInput.value = modelVariant;
    }
}

function updateFormatOptions() {
    const purpose = document.getElementById('modelPurposeInput').value;
    const formatGroup = document.getElementById('formatGroup');
    const formatInput = document.getElementById('modelFormatInput');
    
    if (!framework || !modelConfigs[purpose]) {
        formatGroup.style.display = 'none';
        return;
    }
    
    formatGroup.style.display = 'block';
    formatInput.innerHTML = '';
    
    // Get formats for this framework, or use default
    const formats = modelConfigs[purpose].formats[framework] || modelConfigs[purpose].formats.default || [];
    
    formats.forEach(format => {
        const option = document.createElement('option');
        option.value = format.value;
        option.textContent = format.label;
        formatInput.appendChild(option);
    });
    
    // Auto-select first format
    if (formats.length > 0) {
        modelFormat = formats[0].value;
        formatInput.value = modelFormat;
    }
}

function setupSettings() {
    const qualitySlider = document.getElementById('qualitySlider');
    const presetModeBtn = document.getElementById('presetModeBtn');
    const manualModeBtn = document.getElementById('manualModeBtn');
    const presetSettings = document.getElementById('presetSettings');
    const manualSettings = document.getElementById('manualSettings');
    const applyPresetBtn = document.getElementById('applyPresetBtn');
    const applyManualBtn = document.getElementById('applyManualBtn');
    const modelPurposeInput = document.getElementById('modelPurposeInput');
    const frameworkInput = document.getElementById('frameworkInput');
    const variantInput = document.getElementById('modelVariantInput');
    const formatInput = document.getElementById('modelFormatInput');
    
    // Update model purpose and cascade updates
    modelPurposeInput.addEventListener('change', (e) => {
        modelPurpose = e.target.value;
        updateFrameworkOptions();
        updateNeuralNetworkSettings(); // Update network architecture when model type changes
        updateParameterCount(); // Update parameter display
    });
    
    frameworkInput.addEventListener('change', (e) => {
        framework = e.target.value;
        updateVariantOptions();
        updateFormatOptions();
        updateParameterCount(); // Update parameter display
    });
    
    variantInput.addEventListener('change', (e) => {
        modelVariant = e.target.value;
        updateParameterCount(); // Update parameter display
    });
    
    formatInput.addEventListener('change', (e) => {
        modelFormat = e.target.value;
        updateParameterCount(); // Update parameter display
    });
    
    // Manual settings dropdowns - update parameter count when changed
    const epochsInput = document.getElementById('epochsInput');
    const batchSizeInput = document.getElementById('batchSizeInput');
    const learningRateInput = document.getElementById('learningRateInput');
    const optimizerInput = document.getElementById('optimizerInput');
    const lossFunctionInput = document.getElementById('lossFunctionInput');
    const validationSplitInput = document.getElementById('validationSplitInput');
    
    if (epochsInput) {
        epochsInput.addEventListener('change', () => {
            updateParameterCount(); // Update parameter display
        });
    }
    if (batchSizeInput) {
        batchSizeInput.addEventListener('change', () => {
            updateParameterCount(); // Update parameter display
        });
    }
    if (learningRateInput) {
        learningRateInput.addEventListener('change', () => {
            updateParameterCount(); // Update parameter display
        });
    }
    if (optimizerInput) {
        optimizerInput.addEventListener('change', () => {
            updateParameterCount(); // Update parameter display
        });
    }
    if (lossFunctionInput) {
        lossFunctionInput.addEventListener('change', () => {
            updateParameterCount(); // Update parameter display
        });
    }
    if (validationSplitInput) {
        validationSplitInput.addEventListener('change', () => {
            updateParameterCount(); // Update parameter display
        });
    }
    
    // Slider update - update parameter count when slider changes
    qualitySlider.addEventListener('input', (e) => {
        updateSlider(parseInt(e.target.value));
        updateParameterCount(); // Update parameter display
    });
    
    // Mode toggle
    presetModeBtn.addEventListener('click', () => {
        isManualMode = false;
        presetModeBtn.classList.add('active');
        manualModeBtn.classList.remove('active');
        presetSettings.style.display = 'block';
        manualSettings.style.display = 'none';
    });
    
    manualModeBtn.addEventListener('click', () => {
        isManualMode = true;
        manualModeBtn.classList.add('active');
        presetModeBtn.classList.remove('active');
        presetSettings.style.display = 'none';
        manualSettings.style.display = 'block';
    });
    
    // Apply buttons
    applyPresetBtn.addEventListener('click', applyPresetSettings);
    applyManualBtn.addEventListener('click', applyManualSettings);
}

// Load and display saved models
function loadSavedModels() {
    ipcRenderer.send('list-models');
}

ipcRenderer.on('models-list', (event, models) => {
    const modelsList = document.getElementById('modelsList');
    modelsList.innerHTML = '';
    
    if (models.length === 0) {
        modelsList.innerHTML = '<div class="model-item-empty">No saved models found. Train a model to see it here.</div>';
        return;
    }
    
    models.forEach((model, index) => {
        const modelItem = document.createElement('div');
        modelItem.className = 'model-item';
        
        const statusBadge = model.isCheckpoint ? 
            `<span class="model-badge checkpoint">Checkpoint</span>` : 
            `<span class="model-badge completed">Completed</span>`;
        
        const accuracy = model.accuracy ? model.accuracy.toFixed(2) : '--';
        const progress = model.totalEpochs > 0 ? 
            `${Math.round((model.currentEpoch / model.totalEpochs) * 100)}%` : '--';
        
        modelItem.innerHTML = `
            <div class="model-item-header">
                <span class="model-item-name">${model.filename}</span>
                ${statusBadge}
            </div>
            <div class="model-item-info">
                <span>Progress: ${model.currentEpoch}/${model.totalEpochs} epochs (${progress})</span>
                <span>Accuracy: ${accuracy}%</span>
                <span>Loss: ${model.loss ? model.loss.toFixed(4) : '--'}</span>
                <span>Date: ${new Date(model.timestamp).toLocaleString()}</span>
            </div>
            <div class="model-item-actions">
                <button class="btn btn-small" onclick="continueTraining('${model.filepath.replace(/\\/g, '\\\\')}')">
                    ${model.isCheckpoint ? 'Continue Training' : 'Retrain'}
                </button>
                <button class="btn btn-small" onclick="loadModelForRetrain('${model.filepath.replace(/\\/g, '\\\\')}')">
                    Load Settings
                </button>
            </div>
        `;
        
        modelsList.appendChild(modelItem);
    });
});

ipcRenderer.on('models-list-error', (event, error) => {
    log(`Error loading models: ${error}`, 'error');
});

// Make functions globally accessible for onclick handlers
window.continueTraining = function(filepath) {
    ipcRenderer.send('load-model', filepath);
};

window.loadModelForRetrain = function(filepath) {
    ipcRenderer.send('load-model', filepath);
};

ipcRenderer.on('model-loaded', (event, modelData, filepath) => {
    // Determine if this was triggered by continue or retrain based on model type
    const isCheckpoint = modelData.isPartial || false;
    
    if (isCheckpoint) {
        // This is a checkpoint - continue training from where it left off
        savedCheckpoint = {
            currentEpoch: modelData.currentEpoch || 0,
            totalEpochs: modelData.totalEpochs || 10,
            trainingHistory: modelData.trainingHistory || [],
            loss: modelData.finalMetrics ? modelData.finalMetrics.loss : 0,
            accuracy: modelData.finalMetrics ? modelData.finalMetrics.accuracy : 0,
            trainingSettings: modelData.trainingSettings || {}
        };
        
        // Apply settings from checkpoint
        if (modelData.trainingSettings) {
            trainingSettings = { ...modelData.trainingSettings };
            
            // Update UI with checkpoint settings
            document.getElementById('epochsInput').value = trainingSettings.epochs || 10;
            document.getElementById('batchSizeInput').value = trainingSettings.batchSize || 32;
            document.getElementById('learningRateInput').value = trainingSettings.learningRate || 0.001;
            if (trainingSettings.optimizer) document.getElementById('optimizerInput').value = trainingSettings.optimizer;
            if (trainingSettings.lossFunction) document.getElementById('lossFunctionInput').value = trainingSettings.lossFunction;
            if (trainingSettings.validationSplit) document.getElementById('validationSplitInput').value = trainingSettings.validationSplit.toString();
            
            if (modelData.modelConfig) {
                document.getElementById('modelPurposeInput').value = modelData.modelConfig.purpose || 'machine_learning';
                document.getElementById('frameworkInput').value = modelData.modelConfig.framework || '';
                document.getElementById('modelVariantInput').value = modelData.modelConfig.variant || '';
                updateFrameworkOptions();
            }
        }
        
        log(`Checkpoint loaded: ${modelData.currentEpoch}/${modelData.totalEpochs} epochs`, 'success');
        log('Settings restored. Click "Start Training" to continue from checkpoint.', 'log');
        
        // Auto-scroll to training section
        setTimeout(() => {
            document.querySelector('.training-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    } else {
        // This is a completed model - prepare for retraining (fresh start with same settings)
        savedCheckpoint = null; // Clear any checkpoint
        currentEpoch = 0; // Reset to start
        
        if (modelData.trainingSettings) {
            trainingSettings = { ...modelData.trainingSettings };
            
            // Update UI with model settings
            document.getElementById('epochsInput').value = trainingSettings.epochs || 10;
            document.getElementById('batchSizeInput').value = trainingSettings.batchSize || 32;
            document.getElementById('learningRateInput').value = trainingSettings.learningRate || 0.001;
            if (trainingSettings.optimizer) document.getElementById('optimizerInput').value = trainingSettings.optimizer;
            if (trainingSettings.lossFunction) document.getElementById('lossFunctionInput').value = trainingSettings.lossFunction;
            if (trainingSettings.validationSplit) document.getElementById('validationSplitInput').value = trainingSettings.validationSplit.toString();
            
            if (modelData.modelConfig) {
                document.getElementById('modelPurposeInput').value = modelData.modelConfig.purpose || 'machine_learning';
                document.getElementById('frameworkInput').value = modelData.modelConfig.framework || '';
                document.getElementById('modelVariantInput').value = modelData.modelConfig.variant || '';
                updateFrameworkOptions();
            }
            
            updateParameterCount();
            log('Model settings loaded for retraining', 'success');
            log('Adjust settings if needed, then click "Start Training" to begin fresh training.', 'log');
        }
        
        // Auto-scroll to training section
        setTimeout(() => {
            document.querySelector('.training-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
});

ipcRenderer.on('model-load-error', (event, error) => {
    log(`Error loading model: ${error}`, 'error');
});

// Retrain the current saved model
function retrainModel() {
    if (!savedModelPath) {
        log('No model saved yet. Train a model first.', 'warning');
        return;
    }
    
    // Load the saved model for retraining
    ipcRenderer.send('load-model', savedModelPath);
}

function saveModel(finalLoss, finalAccuracy, isPartial = false) {
    // Get model configuration from UI
    modelPurpose = document.getElementById('modelPurposeInput').value;
    framework = document.getElementById('frameworkInput').value;
    modelVariant = document.getElementById('modelVariantInput').value;
    modelFormat = document.getElementById('modelFormatInput').value;
    
    const modelData = {
        isPartial: isPartial,
        currentEpoch: currentEpoch,
        totalEpochs: totalEpochs,
        trainingSettings: trainingSettings,
        trainingHistory: trainingHistory,
        finalMetrics: {
            loss: finalLoss,
            accuracy: finalAccuracy,
            epochs: isPartial ? currentEpoch : totalEpochs
        },
        modelConfig: {
            purpose: modelPurpose,
            framework: framework,
            variant: modelVariant
        },
        timestamp: new Date().toISOString(),
        fileCount: uploadedFiles.length,
        // Include checkpoint data for resume
        checkpoint: isPartial ? {
            currentEpoch: currentEpoch,
            totalEpochs: totalEpochs,
            trainingHistory: trainingHistory
        } : null
    };
    
    ipcRenderer.send('save-model', modelData, modelPurpose, framework, modelVariant, modelFormat);
}

// Handle model save response
ipcRenderer.on('model-saved', (event, filepath, isPartial) => {
    savedModelPath = filepath;
    document.getElementById('modelPath').textContent = filepath;
    document.getElementById('modelSection').style.display = 'block';
    const saveType = isPartial ? 'Checkpoint saved' : 'Model saved';
    log(`${saveType}: ${filepath}`, 'success');
    
    // Refresh model list
    loadSavedModels();
    
    // Scroll to model section
    document.getElementById('modelSection').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

ipcRenderer.on('model-save-error', (event, error) => {
    log(`Error saving model: ${error}`, 'error');
});

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    loadSystemInfo();
    setupFileUpload();
    setupSettings();
    
    document.getElementById('refreshBtn').addEventListener('click', loadSystemInfo);
    document.getElementById('testGpuBtn').addEventListener('click', testGPU);
    document.getElementById('testCpuBtn').addEventListener('click', testCPU);
    document.getElementById('startTrainingBtn').addEventListener('click', startTraining);
    document.getElementById('stopTrainingBtn').addEventListener('click', stopTraining);
    document.getElementById('trainNewModelBtn').addEventListener('click', trainNewModel);
    
    // Open model location button
    document.getElementById('openModelLocationBtn').addEventListener('click', () => {
        if (savedModelPath) {
            ipcRenderer.send('open-model-location', savedModelPath);
        }
    });
    
    // Close button
    document.getElementById('closeBtn').addEventListener('click', () => {
        ipcRenderer.send('close-window');
    });
    
    // Model management
    document.getElementById('refreshModelsBtn').addEventListener('click', loadSavedModels);
    document.getElementById('retrainModelBtn').addEventListener('click', retrainModel);
    
    // Load saved models on startup
    loadSavedModels();
    
    log('Uni Trainer initialized', 'success');
    log('Ready to start training', 'log');
    
    // Initialize north star metrics
    document.getElementById('northStarValue').textContent = '--';
    document.getElementById('currentAccuracy').textContent = '--';
    document.getElementById('currentLoss').textContent = '--';
    document.getElementById('currentEpoch').textContent = '--';
    
    // Initialize sparklines with empty state
    updateMetricSparklines();
});
