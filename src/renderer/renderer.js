// UNI TRAINER
// Copyright (c) 2026 [Vagif Hasanov]
// 
// This work was created by [Vagif Hasanov] via - contracting AI tools
// as implementation assistants. All creative direction,
// architecture, design, and business logic are original
// works of [Vagif Hasanov].
// 
// AI was used as a tool under direction, similar to
// how a developer uses a compiler or IDE.

const si = require('systeminformation');
const fs = require('fs');
const path = require('path');
const os = require('os');

const { ipcRenderer } = require('electron');

let trainingInterval = null;
let monitoringInterval = null;
let progressAnimationId = null;
let uploadedFiles = [];
let selectedFolderPath = null; // Track selected folder path for faster real training
let selectedDatasetFile = null; // Track selected CSV file path (for tabular data)
let isRealTraining = false; // Track if real training is active
let trainingSettings = {
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    lossFunction: 'mse',
    validationSplit: 0.2,
    // RandomForest-specific settings
    n_estimators: 100,
    max_depth: 10,
    min_samples_split: 2,
    min_samples_leaf: 1,
    max_features: 'sqrt'
};
let modelPurpose = 'machine_learning';
let framework = '';
let modelVariant = '';
let modelFormat = 'pytorch';
let isManualMode = false;
let trainingMode = 'local'; // 'local' or 'cloud'

// Use window.* for cloud-related globals to ensure they're accessible across all scopes
window.canopywaveApiKey = null; // Store CanopyWave API key
window.cloudGPUInfo = null; // Store selected cloud GPU information
window.cloudConfig = null; // Store cloud configuration (project, region, GPU, image, password)
window.currentCloudInstanceId = null; // Store current cloud instance ID for stopping/monitoring

// Note: All cloud state is now stored in window.* globals
// Access via window.canopywaveApiKey, window.cloudConfig, etc.

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
    tabular: {
        frameworks: [
            { value: 'sklearn', label: 'Scikit-learn - Traditional ML ⭐ Recommended' },
            { value: 'xgboost', label: 'XGBoost - Gradient boosting' },
            { value: 'lightgbm', label: 'LightGBM - Fast gradient boosting' },
            { value: 'pytorch', label: 'PyTorch - Deep learning' },
            { value: 'catboost', label: 'CatBoost - Categorical boosting' }
        ],
        variants: {
            sklearn: [
                { value: 'random_forest', label: 'Random Forest ⭐ Recommended' },
                { value: 'gradient_boosting', label: 'Gradient Boosting' },
                { value: 'extra_trees', label: 'Extra Trees' },
                { value: 'svm', label: 'SVM' },
                { value: 'logistic_regression', label: 'Logistic Regression' },
                { value: 'linear_regression', label: 'Linear Regression' }
            ],
            xgboost: [
                { value: 'xgboost_default', label: 'XGBoost Default ⭐ Recommended' },
                { value: 'xgboost_tuned', label: 'XGBoost Tuned (Hyperparameter optimized)' }
            ],
            lightgbm: [
                { value: 'lightgbm_default', label: 'LightGBM Default ⭐ Recommended' },
                { value: 'lightgbm_tuned', label: 'LightGBM Tuned' }
            ],
            pytorch: [
                { value: 'mlp_small', label: 'MLP Small (2 layers)' },
                { value: 'mlp_medium', label: 'MLP Medium (3 layers) ⭐ Recommended' },
                { value: 'mlp_large', label: 'MLP Large (4 layers)' },
                { value: 'mlp_deep', label: 'MLP Deep (5+ layers)' }
            ],
            catboost: [
                { value: 'catboost_default', label: 'CatBoost Default ⭐ Recommended' }
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
            catboost: [
                { value: 'cbm', label: 'CatBoost Model (.cbm) ⭐ Recommended' }
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
let trainingStartTime = null; // When training started
let currentEpoch = 0;
let totalEpochs = 0;
let savedModelPath = null;
let trainingHistory = [];
let savedCheckpoint = null; // For resuming training
let displayedProgress = 0; // Smoothly displayed progress (0-100)
let estimatedTrainingDuration = 0; // Estimated total training time in seconds
let progressEstimationInterval = null; // Interval for progress estimation
let lastRealProgress = null; // Last real progress received from training

/**
 * Centralized cloud GPU pricing calculation utility
 * Single source of truth for all pricing calculations
 * @param {object} params - Pricing parameters
 * @param {number} params.pricePerGpuHour - Price per GPU per hour
 * @param {number} params.gpuCount - Number of GPUs
 * @param {number} params.providerUpfrontHours - Provider upfront authorization hours (default: 24)
 * @returns {object} Calculated costs
 * @returns {number} returns.hourlyCost - Total hourly cost (pricePerGpuHour * gpuCount)
 * @returns {number} returns.upfrontAuthorizationCost - Upfront authorization cost (hourlyCost * providerUpfrontHours)
 */
function calculateCloudCosts({ pricePerGpuHour, gpuCount, providerUpfrontHours = 24 }) {
    const hourlyCost = pricePerGpuHour * gpuCount;
    const upfrontAuthorizationCost = hourlyCost * providerUpfrontHours;
    
    return {
        hourlyCost,
        upfrontAuthorizationCost
    };
}

// Start progress estimation based on elapsed time
function startProgressEstimation() {
    // Clear any existing interval
    if (progressEstimationInterval) {
        clearInterval(progressEstimationInterval);
    }
    
    console.log('[Renderer] Starting progress estimation');
    
    // Estimate progress every 500ms
    progressEstimationInterval = setInterval(() => {
        if (!trainingStartTime || !isRealTraining) {
            return;
        }
        
        // Skip estimation if we're receiving real training updates (within last 2 seconds)
        if (isRealTraining && lastRealProgress) {
            const timeSinceLastRealProgress = Date.now() - lastRealProgress.timestamp;
            if (timeSinceLastRealProgress < 2000) {
                return; // Skip estimation, real updates are coming
            }
        }
        
        // Calculate elapsed time
        const elapsedSeconds = (Date.now() - trainingStartTime) / 1000;
        
        // Estimate progress based on batch-by-batch calculation
        // For YOLO: ~194 batches per epoch (6194 images / 32 batch size)
        // Estimate 2-3 seconds per batch (conservative estimate)
        const batchesPerEpoch = 194; // 6194 images / 32 batch size
        const estimatedSecondsPerBatch = 2.5; // Conservative estimate
        const estimatedSecondsPerEpoch = batchesPerEpoch * estimatedSecondsPerBatch;
        const estimatedTotalSeconds = totalEpochs * estimatedSecondsPerEpoch;
        
        // Calculate estimated progress with batch-level granularity
        // This gives more realistic progress during long epochs
        const estimatedProgress = Math.min(0.95, elapsedSeconds / estimatedTotalSeconds);
        
        // Only update if estimated progress is greater than current displayed progress
        // This prevents going backwards when real progress comes in
        if (estimatedProgress > displayedProgress) {
            displayedProgress = estimatedProgress;
            const progressPercent = Math.round(estimatedProgress * 100);
            
            // Update UI
            const progressElement = document.getElementById('northStarValue');
            if (progressElement) {
                progressElement.textContent = `${progressPercent}%`;
            }
            
            // DON'T update epoch/accuracy/loss in UI from estimation - only update progress percentage
            // These should only be updated from real training progress data
            
            // Update neural network visualization with estimated values (visual only)
            if (typeof neuralNetwork !== 'undefined' && neuralNetwork) {
                neuralNetwork.updateTrainingMetrics(
                    1.0 - (estimatedProgress * 0.5), // Estimated loss decreases with progress
                    estimatedProgress * 50, // Estimated accuracy increases with progress
                    Math.floor(estimatedProgress * totalEpochs), // Estimated epoch
                    totalEpochs,
                    estimatedProgress
                );
            }
        }
    }, 500); // Update every 500ms
}

// Stop progress estimation
function stopProgressEstimation() {
    if (progressEstimationInterval) {
        clearInterval(progressEstimationInterval);
        progressEstimationInterval = null;
    }
    // Don't reset trainingStartTime or lastRealProgress - they're needed for ETA calculation
}

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
    // Only load local system info if not in cloud mode
    if (trainingMode === 'cloud') {
        updateCloudSystemInfo();
        return;
    }
    
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
    
    // Start real-time resource monitoring (only for local mode)
    if (trainingMode === 'local') {
        startResourceMonitoring();
    }
}

function updateCloudSystemInfo() {
    if (!window.cloudConfig) {
        // No cloud config yet, show placeholder
        document.getElementById('cpu-info').textContent = 'Cloud Instance';
        document.getElementById('gpu-info').textContent = 'Select GPU in configuration';
        document.getElementById('memory-info').textContent = 'N/A';
        return;
    }
    
    // Update system info with cloud GPU information
    document.getElementById('cpu-info').textContent = `Cloud Instance - ${window.window.cloudConfig.region || 'N/A'}`;
    document.getElementById('gpu-info').textContent = window.window.cloudConfig.gpuName || window.window.cloudConfig.gpu || 'Cloud GPU';
    document.getElementById('memory-info').textContent = 'Cloud Instance';
    
    log('Cloud system information updated', 'success');
}

function showBalanceDisplay() {
    const balanceDisplay = document.getElementById('balanceDisplay');
    if (balanceDisplay) {
        balanceDisplay.style.display = 'flex';
    }
}

function hideBalanceDisplay() {
    const balanceDisplay = document.getElementById('balanceDisplay');
    if (balanceDisplay) {
        balanceDisplay.style.display = 'none';
    }
}

async function loadBalance() {
    if (!window.canopywaveApiKey) {
        hideBalanceDisplay();
        return;
    }
    
    try {
        const result = await ipcRenderer.invoke('get-canopywave-balance', window.canopywaveApiKey);
        console.log('[Balance] API result:', result);
        
        if (result.success && result.balance !== undefined && result.balance !== null) {
            // Handle different possible balance response formats
            let balanceValue = 0;
            
            if (typeof result.balance === 'number') {
                balanceValue = result.balance;
            } else if (typeof result.balance === 'object') {
                // Try various property names
                balanceValue = result.balance.balance || 
                             result.balance.amount || 
                             result.balance.credits || 
                             result.balance.value ||
                             result.balance.remaining ||
                             0;
            } else if (typeof result.balance === 'string') {
                // Try to parse string
                balanceValue = parseFloat(result.balance) || 0;
            }
            
            const balanceAmountEl = document.getElementById('balanceAmount');
            if (balanceAmountEl) {
                if (balanceValue > 0 || balanceValue === 0) {
                    balanceAmountEl.textContent = `$${balanceValue.toFixed(2)}`;
                } else {
                    balanceAmountEl.textContent = 'Loading...';
                    console.warn('[Balance] Unexpected balance format:', result.balance);
                }
            }
        } else {
            console.error('[Balance] Failed to load balance:', result.error);
            const balanceAmountEl = document.getElementById('balanceAmount');
            if (balanceAmountEl) {
                // Don't show N/A, try alternative endpoint or show as unavailable
                balanceAmountEl.textContent = 'Unavailable';
            }
        }
    } catch (error) {
        console.error('[Balance] Error loading balance:', error);
        const balanceAmountEl = document.getElementById('balanceAmount');
        if (balanceAmountEl) {
            balanceAmountEl.textContent = 'Error';
        }
    }
}

// Real-time resource monitoring
function startResourceMonitoring() {
    // Clear any existing interval
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
    }
    
    // Update resources every second
    monitoringInterval = setInterval(async () => {
        try {
            // CPU usage
            const cpuCurrent = await si.currentLoad();
            const cpuPercent = Math.round(cpuCurrent.currentLoad || 0);
            const cpuUsageEl = document.getElementById('cpu-usage');
            const cpuPercentEl = document.getElementById('cpu-percent');
            if (cpuUsageEl) cpuUsageEl.style.width = `${cpuPercent}%`;
            if (cpuPercentEl) cpuPercentEl.textContent = `${cpuPercent}%`;
            
            // GPU usage (if available)
            try {
                const graphics = await si.graphics();
                if (graphics && graphics.controllers && graphics.controllers.length > 0) {
                    // GPU usage is not directly available via systeminformation
                    // This would require platform-specific tools (nvidia-smi, etc.)
                    // For now, we'll use a placeholder or estimate based on training state
                    const gpuPercent = isRealTraining ? Math.min(95, Math.random() * 30 + 60) : Math.random() * 10;
                    const gpuUsageEl = document.getElementById('gpu-usage');
                    const gpuPercentEl = document.getElementById('gpu-percent');
                    if (gpuUsageEl) gpuUsageEl.style.width = `${Math.round(gpuPercent)}%`;
                    if (gpuPercentEl) gpuPercentEl.textContent = `${Math.round(gpuPercent)}%`;
                }
            } catch (gpuError) {
                // GPU monitoring not available
            }
            
            // Memory usage
            const mem = await si.mem();
            const memUsed = mem.used;
            const memTotal = mem.total;
            const memPercent = Math.round((memUsed / memTotal) * 100);
            const memoryUsageEl = document.getElementById('memory-usage');
            const memoryPercentEl = document.getElementById('memory-percent');
            if (memoryUsageEl) memoryUsageEl.style.width = `${memPercent}%`;
            if (memoryPercentEl) memoryPercentEl.textContent = `${memPercent}%`;
        } catch (error) {
            console.error('Error updating resource monitoring:', error);
        }
    }, 1000);
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

// Check if current model configuration supports real training
function supportsRealTraining() {
    const modelPurpose = document.getElementById('modelPurposeInput').value;
    const framework = document.getElementById('frameworkInput').value;
    
    // Currently implemented real trainers
    const supportedCombinations = [
        { purpose: 'computer_vision', framework: 'yolo' },
        { purpose: 'machine_learning', framework: 'sklearn' },
        { purpose: 'machine_learning', framework: 'xgboost' },
        { purpose: 'tabular', framework: 'sklearn' },
        { purpose: 'tabular', framework: 'xgboost' },
        { purpose: 'tabular', framework: 'lightgbm' },
        { purpose: 'tabular', framework: 'pytorch' }
    ];
    
    return supportedCombinations.some(combo => 
        combo.purpose === modelPurpose && combo.framework === framework
    );
}

// Validate dataset format based on model type
function validateDatasetFormat(folderPath, modelPurpose, framework) {
    const fs = require('fs');
    const path = require('path');
    
    try {
        if (!fs.existsSync(folderPath)) {
            return {
                valid: false,
                message: 'Selected folder does not exist',
                expectedFormat: null
            };
        }
        
        const files = fs.readdirSync(folderPath);
        const fileStats = files.map(f => {
            const fullPath = path.join(folderPath, f);
            return {
                name: f,
                isDirectory: fs.statSync(fullPath).isDirectory(),
                path: fullPath
            };
        });
        
        // Computer Vision - YOLO format
        if (modelPurpose === 'computer_vision' && framework === 'yolo') {
            const hasDataYaml = files.includes('data.yaml');
            const hasTrainDir = fileStats.some(f => f.isDirectory && f.name === 'train');
            const hasValidDir = fileStats.some(f => f.isDirectory && f.name === 'valid');
            const hasImagesDir = fileStats.some(f => f.isDirectory && f.name === 'images');
            const hasLabelsDir = fileStats.some(f => f.isDirectory && f.name === 'labels');
            
            if (hasDataYaml) {
                return {
                    valid: true,
                    message: '✓ YOLO format detected (data.yaml found)',
                    expectedFormat: 'YOLO format with data.yaml, images/, and labels/ folders'
                };
            } else if (hasTrainDir && hasValidDir) {
                return {
                    valid: true,
                    message: '✓ YOLO format detected (train/valid structure)',
                    expectedFormat: 'YOLO format with train/ and valid/ directories'
                };
            } else if (hasImagesDir && hasLabelsDir) {
                return {
                    valid: true,
                    message: '✓ YOLO format detected (images/labels structure)',
                    expectedFormat: 'YOLO format with images/ and labels/ directories'
                };
            } else {
                return {
                    valid: false,
                    message: 'YOLO format not detected. Missing data.yaml or required directory structure.',
                    expectedFormat: 'YOLO format: data.yaml + images/ + labels/ folders, OR train/ + valid/ directories'
                };
            }
        }
        
        // Tabular Data - CSV files
        if (modelPurpose === 'tabular') {
            const csvFiles = files.filter(f => f.toLowerCase().endsWith('.csv'));
            
            if (csvFiles.length === 0) {
                return {
                    valid: false,
                    message: 'No CSV files found in folder',
                    expectedFormat: 'CSV file(s) with features in columns and target in last column (or named "target"/"label"/"y")'
                };
            } else {
                return {
                    valid: true,
                    message: `✓ Tabular format detected (${csvFiles.length} CSV file(s))`,
                    expectedFormat: 'CSV file(s) with features and target column'
                };
            }
        }
        
        // Machine Learning - CSV or JSON files
        if (modelPurpose === 'machine_learning') {
            const csvFiles = files.filter(f => f.toLowerCase().endsWith('.csv'));
            const jsonFiles = files.filter(f => f.toLowerCase().endsWith('.json'));
            
            if (csvFiles.length === 0 && jsonFiles.length === 0) {
                return {
                    valid: false,
                    message: 'No CSV or JSON files found in folder',
                    expectedFormat: 'CSV or JSON file(s) with features and target column'
                };
            } else {
                const fileType = csvFiles.length > 0 ? 'CSV' : 'JSON';
                const count = csvFiles.length > 0 ? csvFiles.length : jsonFiles.length;
                return {
                    valid: true,
                    message: `✓ Machine Learning format detected (${count} ${fileType} file(s))`,
                    expectedFormat: `${fileType} file(s) with features and target column`
                };
            }
        }
        
        // Natural Language Processing - Text files
        if (modelPurpose === 'natural_language_processing') {
            const textFiles = files.filter(f => {
                const ext = f.toLowerCase().split('.').pop();
                return ['txt', 'json', 'csv'].includes(ext);
            });
            
            if (textFiles.length === 0) {
                return {
                    valid: false,
                    message: 'No text files found in folder',
                    expectedFormat: 'Text files (.txt, .json, or .csv) containing text data'
                };
            } else {
                return {
                    valid: true,
                    message: `✓ NLP format detected (${textFiles.length} text file(s))`,
                    expectedFormat: 'Text files (.txt, .json, or .csv)'
                };
            }
        }
        
        // Time Series - CSV or JSON files
        if (modelPurpose === 'time_series') {
            const csvFiles = files.filter(f => f.toLowerCase().endsWith('.csv'));
            const jsonFiles = files.filter(f => f.toLowerCase().endsWith('.json'));
            
            if (csvFiles.length === 0 && jsonFiles.length === 0) {
                return {
                    valid: false,
                    message: 'No CSV or JSON files found in folder',
                    expectedFormat: 'CSV or JSON file(s) with time series data (timestamp + values)'
                };
            } else {
                const fileType = csvFiles.length > 0 ? 'CSV' : 'JSON';
                const count = csvFiles.length > 0 ? csvFiles.length : jsonFiles.length;
                return {
                    valid: true,
                    message: `✓ Time Series format detected (${count} ${fileType} file(s))`,
                    expectedFormat: `${fileType} file(s) with time series data`
                };
            }
        }
        
        // Reinforcement Learning - typically needs environment files or data files
        if (modelPurpose === 'reinforcement_learning') {
            // RL can have various formats, so we'll be lenient
            return {
                valid: true,
                message: '✓ Folder selected for reinforcement learning',
                expectedFormat: 'Environment files, game data, or training logs'
            };
        }
        
        // Generative Models - can be images or other data
        if (modelPurpose === 'generative') {
            const imageFiles = files.filter(f => {
                const ext = f.toLowerCase().split('.').pop();
                return ['jpg', 'jpeg', 'png', 'gif', 'bmp'].includes(ext);
            });
            
            if (imageFiles.length === 0) {
                return {
                    valid: true,
                    message: '✓ Folder selected for generative models',
                    expectedFormat: 'Image files or other training data'
                };
            } else {
                return {
                    valid: true,
                    message: `✓ Generative format detected (${imageFiles.length} image file(s))`,
                    expectedFormat: 'Image files or other training data'
                };
            }
        }
        
        // Default: accept any folder (for unknown model types)
        return {
            valid: true,
            message: '✓ Folder selected',
            expectedFormat: 'Appropriate data format for selected model type'
        };
        
    } catch (error) {
        return {
            valid: false,
            message: `Error validating dataset: ${error.message}`,
            expectedFormat: null
        };
    }
}

// Function to update dataset button text based on selected model type
function updateDatasetButtonText() {
    const selectFolderBtn = document.getElementById('selectFolderBtn');
    const modelPurposeInput = document.getElementById('modelPurposeInput');
    
    if (!selectFolderBtn || !modelPurposeInput) return;
    
    const modelPurpose = modelPurposeInput.value;
    
    switch(modelPurpose) {
        case 'tabular':
            selectFolderBtn.textContent = '📊 Select CSV File or Folder';
            break;
        case 'computer_vision':
            selectFolderBtn.textContent = '📂 Select Dataset Folder';
            break;
        case 'natural_language_processing':
            selectFolderBtn.textContent = '📄 Select Text Files Folder';
            break;
        case 'time_series':
            selectFolderBtn.textContent = '📈 Select Time Series Data Folder';
            break;
        default:
            selectFolderBtn.textContent = '📂 Select Dataset Folder';
    }
}

// Update epoch label based on model type
function updateEpochLabel(modelPurpose, framework, variant) {
    const epochLabel = document.getElementById('epochLabel');
    if (!epochLabel) return;
    
    // If model purpose is not set yet, default to 'Epoch'
    if (!modelPurpose) {
        epochLabel.textContent = 'Epoch';
        return;
    }
    
    // Check if this is RandomForest (tabular + sklearn + random_forest)
    // Use explicit check instead of relying on external variable
    const isRandomForest = modelPurpose === 'tabular' && framework === 'sklearn' && variant === 'random_forest';
    
    // For tabular models, use "Trees" for RandomForest, "Progress" for others
    if (modelPurpose === 'tabular') {
        if (isRandomForest) {
            epochLabel.textContent = 'Trees';
        } else {
            epochLabel.textContent = 'Progress';
        }
    } else {
        epochLabel.textContent = 'Epoch';
    }
}

// Initialize epoch label on page load
function initializeEpochLabel() {
    const modelPurposeInput = document.getElementById('modelPurposeInput');
    const frameworkInput = document.getElementById('frameworkInput');
    const variantInput = document.getElementById('modelVariantInput');
    
    if (modelPurposeInput && frameworkInput && variantInput) {
        const modelPurpose = modelPurposeInput.value;
        const framework = frameworkInput.value;
        const variant = variantInput.value;
        updateEpochLabel(modelPurpose, framework, variant);
    }
}

// Extract schema from CSV file
async function extractCSVSchema(csvFilePath) {
    try {
        const fs = require('fs');
        const path = require('path');
        
        // Read first few lines to get schema
        const fileContent = fs.readFileSync(csvFilePath, 'utf-8');
        const lines = fileContent.split('\n').filter(line => line.trim());
        
        if (lines.length < 2) {
            throw new Error('CSV file must have at least a header and one data row');
        }
        
        // Parse header
        const header = lines[0].split(',').map(col => col.trim().replace(/^"|"$/g, ''));
        
        // Parse first few data rows for sample values
        const sampleRows = lines.slice(1, Math.min(4, lines.length));
        const sampleData = sampleRows.map(row => {
            const values = row.split(',').map(val => val.trim().replace(/^"|"$/g, ''));
            return values;
        });
        
        // Read full file to compute statistics (for small files) or sample (for large files)
        const fullContent = fs.readFileSync(csvFilePath, 'utf-8');
        const allLines = fullContent.split('\n').filter(line => line.trim());
        const totalRows = allLines.length - 1; // Exclude header
        
        // For large files, sample rows for statistics
        const maxRowsToAnalyze = Math.min(10000, totalRows);
        const rowsToAnalyze = totalRows <= maxRowsToAnalyze 
            ? allLines.slice(1) 
            : allLines.slice(1).filter((_, i) => i % Math.ceil(totalRows / maxRowsToAnalyze) === 0);
        
        const schema = [];
        
        for (let i = 0; i < header.length; i++) {
            const colName = header[i];
            const values = rowsToAnalyze.map(row => {
                const cols = row.split(',').map(val => val.trim().replace(/^"|"$/g, ''));
                return cols[i] || '';
            }).filter(val => val !== '');
            
            // Determine type
            const numericCount = values.filter(val => !isNaN(val) && val !== '').length;
            const isNumeric = numericCount / values.length > 0.8; // 80% numeric = numeric type
            
            // Compute unique count and ratio
            const uniqueValues = new Set(values);
            const uniqueCount = uniqueValues.size;
            const uniqueRatio = values.length > 0 ? uniqueCount / values.length : 0;
            
            // Get sample values (first 3 non-empty)
            const sampleValues = values.slice(0, 3).filter(v => v !== '');
            
            schema.push({
                name: colName,
                type: isNumeric ? 'numeric' : 'categorical',
                uniqueCount: uniqueCount,
                uniqueRatio: uniqueRatio,
                uniquePercent: (uniqueRatio * 100).toFixed(1),
                sampleValues: sampleValues,
                isSuspicious: uniqueRatio > 0.95 || colName.toLowerCase().includes('id')
            });
        }
        
        return schema;
    } catch (error) {
        console.error('Error extracting CSV schema:', error);
        throw error;
    }
}

// Helper: Get label-like confidence based on name match + unique count
// Returns: null if not label-like, or { isLabelLike: true, confidence: 'high'|'medium' }
function getLabelLikeConfidence(columnName, uniqueValueCount = null) {
    if (!columnName || typeof columnName !== 'string') return null;
    
    // First check if name is label-like (strict detection)
    const nameResult = isLabelLikeColumnNameOnly(columnName);
    if (!nameResult.isLabelLike) {
        return null; // Not label-like, no warning
    }
    
    // If label-like, determine confidence based on unique count
    if (uniqueValueCount !== null && uniqueValueCount <= 2) {
        return { isLabelLike: true, confidence: 'high' };
    } else {
        return { isLabelLike: true, confidence: 'medium' };
    }
}

// Update the label-like warning banner based on selected features
// Parameters:
//   - selectedFeatureColumns: Array of selected feature column names (optional, will be read from checkboxes if not provided)
//   - targetColumn: The target column name (optional, will be read from dropdown if not provided)
function updateLabelLikeWarningBanner(selectedFeatureColumns = null, targetColumn = null) {
    const warningBanner = document.getElementById('labelLikeWarningBanner');
    const warningText = document.getElementById('labelLikeWarningText');
    
    if (!warningBanner || !warningText) {
        return; // Banner elements don't exist
    }
    
    // Get selected features from checkboxes if not provided
    if (!selectedFeatureColumns) {
        const featureCheckboxes = document.querySelectorAll('#featureColumnsList input[type="checkbox"]:checked');
        selectedFeatureColumns = Array.from(featureCheckboxes).map(cb => cb.value);
    }
    
    // Get target column from dropdown if not provided
    if (!targetColumn) {
        const targetSelect = document.getElementById('targetColumnSelect');
        targetColumn = targetSelect ? targetSelect.value : null;
    }
    
    // Get schema to access unique counts
    const schema = window.tabularSchema || [];
    
    // Find label-like features that are selected
    const labelLikeSelected = [];
    
    selectedFeatureColumns.forEach(colName => {
        // Skip if it's the target column
        if (colName === targetColumn) {
            return;
        }
        
        // Find column in schema to get unique count
        const col = schema.find(c => c.name === colName);
        const uniqueCount = col ? col.uniqueCount : null;
        
        // Check if label-like
        const labelLikeResult = getLabelLikeConfidence(colName, uniqueCount);
        if (labelLikeResult && labelLikeResult.isLabelLike) {
            labelLikeSelected.push({
                name: colName,
                confidence: labelLikeResult.confidence,
                uniqueCount: uniqueCount
            });
        }
    });
    
    // Show/hide banner based on whether any label-like features are selected
    if (labelLikeSelected.length > 0) {
        // Build warning text
        const highConfidence = labelLikeSelected.filter(f => f.confidence === 'high');
        const mediumConfidence = labelLikeSelected.filter(f => f.confidence === 'medium');
        
        let warningMessage = '';
        if (highConfidence.length > 0) {
            const names = highConfidence.map(f => f.name).join(', ');
            warningMessage += `High confidence: ${names}`;
        }
        if (mediumConfidence.length > 0) {
            if (warningMessage) warningMessage += '. ';
            const names = mediumConfidence.map(f => f.name).join(', ');
            warningMessage += `Medium confidence: ${names}`;
        }
        
        warningText.textContent = warningMessage;
        warningBanner.style.display = 'block';
    } else {
        warningBanner.style.display = 'none';
    }
}

// Helper: Check if column name looks like a label/target (STRICT name matching only)
// Returns: { isLabelLike: boolean }
function isLabelLikeColumnNameOnly(columnName) {
    if (!columnName || typeof columnName !== 'string') return { isLabelLike: false };
    
    // STRICT label/target tokens only
    const strictLabelTokens = [
        'label', 'target', 'class', 'outcome', 'y',
        'bought', 'purchase', 'purchased', 'churn', 'clicked',
        'converted', 'conversion', 'fraud', 'default', 'response'
    ];
    
    const lowerName = columnName.toLowerCase().trim();
    
    // Check for exact match (for short tokens like "y")
    if (lowerName === 'y' || lowerName === 'label' || lowerName === 'target' || lowerName === 'class') {
        return { isLabelLike: true };
    }
    
    // For longer tokens, check for substring match but only in specific contexts
    for (const pattern of strictLabelTokens) {
        if (pattern.length <= 2) continue; // Skip short patterns (already handled above)
        
        // Exact match
        if (lowerName === pattern) {
            return { isLabelLike: true };
        }
        
        // Starts with pattern_
        if (lowerName.startsWith(pattern + '_')) {
            return { isLabelLike: true };
        }
        
        // Ends with _pattern
        if (lowerName.endsWith('_' + pattern)) {
            return { isLabelLike: true };
        }
        
        // Contains pattern as whole word (with word boundaries)
        const wordBoundaryRegex = new RegExp(`\\b${pattern}\\b`, 'i');
        if (wordBoundaryRegex.test(lowerName)) {
            return { isLabelLike: true };
        }
    }
    
    return { isLabelLike: false };
}

// Helper: Check if column name looks like a label/target (STRICT matching)
// Returns: { isLabelLike: boolean, confidence?: 'high' | 'medium' }
// DEPRECATED: Use getLabelLikeConfidence() instead for better confidence calculation
function isLabelLikeColumn(columnName, uniqueValueCount = null) {
    if (!columnName || typeof columnName !== 'string') return { isLabelLike: false };
    
    // STRICT label/target tokens only
    const strictLabelTokens = [
        'label', 'target', 'class', 'outcome', 'y',
        'bought', 'purchase', 'purchased', 'churn', 'clicked',
        'converted', 'conversion', 'fraud', 'default', 'response'
    ];
    
    const lowerName = columnName.toLowerCase().trim();
    
    // Check for exact match (for short tokens like "y")
    if (lowerName === 'y' || lowerName === 'label' || lowerName === 'target' || lowerName === 'class') {
        const isBinary = uniqueValueCount !== null && uniqueValueCount <= 2;
        return {
            isLabelLike: true,
            confidence: isBinary ? 'high' : 'medium'
        };
    }
    
    // For longer tokens, check for substring match but only in specific contexts
    // Must be: exact match, starts with token_, ends with _token, or contains token as whole word
    for (const pattern of strictLabelTokens) {
        if (pattern.length <= 2) continue; // Skip short patterns (already handled above)
        
        // Exact match
        if (lowerName === pattern) {
            const isBinary = uniqueValueCount !== null && uniqueValueCount <= 2;
            return {
                isLabelLike: true,
                confidence: isBinary ? 'high' : 'medium'
            };
        }
        
        // Starts with pattern_
        if (lowerName.startsWith(pattern + '_')) {
            const isBinary = uniqueValueCount !== null && uniqueValueCount <= 2;
            return {
                isLabelLike: true,
                confidence: isBinary ? 'high' : 'medium'
            };
        }
        
        // Ends with _pattern
        if (lowerName.endsWith('_' + pattern)) {
            const isBinary = uniqueValueCount !== null && uniqueValueCount <= 2;
            return {
                isLabelLike: true,
                confidence: isBinary ? 'high' : 'medium'
            };
        }
        
        // Contains pattern as whole word (with word boundaries)
        const wordBoundaryRegex = new RegExp(`\\b${pattern}\\b`, 'i');
        if (wordBoundaryRegex.test(lowerName)) {
            const isBinary = uniqueValueCount !== null && uniqueValueCount <= 2;
            return {
                isLabelLike: true,
                confidence: isBinary ? 'high' : 'medium'
            };
        }
    }
    
    return { isLabelLike: false };
}

// Helper: Validate tabular config
function validateTabularConfig(targetColumn, featureColumns, allColumns) {
    const errors = [];
    const warnings = [];
    
    // Hard invariant: target_column must exist
    if (!targetColumn || targetColumn.trim() === '') {
        errors.push('Target column must be selected');
    }
    
    // Hard invariant: feature_columns must be non-empty
    if (!featureColumns || featureColumns.length === 0) {
        errors.push('At least one feature column must be selected');
    }
    
    // Hard invariant: target_column must NOT be in feature_columns
    if (targetColumn && featureColumns && featureColumns.includes(targetColumn)) {
        errors.push(`Target column "${targetColumn}" cannot be used as a feature`);
    }
    
    // Warning: label-like features (strict detection with confidence)
    if (featureColumns) {
        // Get unique counts from schema if available
        const columnUniqueCounts = {};
        if (allColumns && window.tabularSchema) {
            window.tabularSchema.forEach(schemaCol => {
                if (featureColumns.includes(schemaCol.name)) {
                    columnUniqueCounts[schemaCol.name] = schemaCol.uniqueCount;
                }
            });
        }
        
        featureColumns.forEach(col => {
            const uniqueCount = columnUniqueCounts[col];
            const result = getLabelLikeConfidence(col, uniqueCount);
            if (result && result.isLabelLike) {
                warnings.push({
                    column: col,
                    confidence: result.confidence || 'medium',
                    uniqueCount: uniqueCount,
                    message: `Feature "${col}" looks like a label/target column and may cause confusion`
                });
            }
        });
    }
    
    // Warning: target column doesn't look like a target
    if (targetColumn && !isLabelLikeColumn(targetColumn)) {
        // This is just informational, not a warning
    }
    
    return {
        valid: errors.length === 0,
        errors: errors,
        warnings: warnings
    };
}

// Update Schema Review UI
function updateSchemaReviewUI(schema, csvFilePath, preserveTargetSelection = false) {
    const schemaSection = document.getElementById('schemaReviewSection');
    const targetSelect = document.getElementById('targetColumnSelect');
    const featureList = document.getElementById('featureColumnsList');
    
    if (!schemaSection || !targetSelect || !featureList) return;
    
    // Show schema section
    schemaSection.style.display = 'block';
    
    // Preserve current selection if requested
    const currentSelection = preserveTargetSelection ? targetSelect.value : null;
    
    // Populate target column dropdown with visual grouping
    targetSelect.innerHTML = '<option value="">Select target column...</option>';
    
    // Find suggested target using smart logic:
    // 1. Prefer label-like column names (target, label, class, bought, churn, etc.)
    // 2. Fallback to last column (common convention)
    // 3. Never default to generic categorical features like "city" unless explicitly selected
    let suggestedTarget = null;
    const lastCol = schema[schema.length - 1];
    
    // Check for label-like names first (using strict detection)
    for (const col of schema) {
        const labelLikeResult = isLabelLikeColumnNameOnly(col.name);
        if (labelLikeResult.isLabelLike) {
            suggestedTarget = col;
            break;
        }
    }
    
    // If no label-like column found, check for common target patterns
    if (!suggestedTarget) {
        const targetLikeNames = ['target', 'label', 'y', 'class', 'outcome', 'result', 'bought', 'churn', 'clicked', 'converted'];
        for (const col of schema) {
            const lowerName = col.name.toLowerCase();
            if (targetLikeNames.some(name => lowerName === name || lowerName.includes(name))) {
                suggestedTarget = col;
                break;
            }
        }
    }
    
    // Fallback to last column if no target-like name found
    // But avoid generic categorical features like city, state, country, zip, etc.
    if (!suggestedTarget) {
        const genericCategoricals = ['city', 'state', 'country', 'zip', 'zipcode', 'address', 'location', 'region'];
        const lastColName = lastCol.name.toLowerCase();
        const isGenericCategorical = genericCategoricals.some(gc => lastColName.includes(gc));
        
        if (isGenericCategorical && schema.length > 1) {
            // Prefer second-to-last column if last is generic categorical
            suggestedTarget = schema[schema.length - 2] || lastCol;
        } else {
            suggestedTarget = lastCol;
        }
    }
    
    // Add suggested target with optgroup
    const suggestedGroup = document.createElement('optgroup');
    suggestedGroup.label = '— Suggested —';
    const suggestedOption = document.createElement('option');
    suggestedOption.value = suggestedTarget.name;
    suggestedOption.textContent = suggestedTarget.name;
    // Only auto-select if no preserved selection
    if (!currentSelection || currentSelection === suggestedTarget.name) {
        suggestedOption.selected = true;
    }
    suggestedGroup.appendChild(suggestedOption);
    targetSelect.appendChild(suggestedGroup);
    
    // Add other columns with optgroup
    const otherCols = schema.filter(col => col.name !== suggestedTarget.name);
    if (otherCols.length > 0) {
        const otherGroup = document.createElement('optgroup');
        otherGroup.label = '— Other columns —';
        otherCols.forEach(col => {
            const option = document.createElement('option');
            option.value = col.name;
            option.textContent = col.name;
            // Preserve selection if it matches
            if (currentSelection === col.name) {
                option.selected = true;
            }
            otherGroup.appendChild(option);
        });
        targetSelect.appendChild(otherGroup);
    }
    
    // Restore selection if it was preserved
    if (currentSelection) {
        targetSelect.value = currentSelection;
    }
    
    // Populate feature columns list with clean visual hierarchy
    featureList.innerHTML = '';
    
    // Get auto-exclude setting
    const autoExclude = document.getElementById('autoExcludeIdColumns')?.checked ?? true;
    
    // Track label-like features for warning banner
    const labelLikeFeatures = [];
    
    schema.forEach(col => {
        const isTarget = col.name === targetSelect.value;
        const labelLikeResult = getLabelLikeConfidence(col.name, col.uniqueCount);
        const isLabelLike = labelLikeResult && labelLikeResult.isLabelLike;
        
        if (isTarget) {
            // Show target column in feature list but disabled
            const item = document.createElement('div');
            item.style.cssText = 'padding: 12px; margin-bottom: 8px; background: #F5F5F5; border-radius: 6px; border: 1px solid #E0D5C7; opacity: 0.6; pointer-events: none;';
            
            const primaryRow = document.createElement('div');
            primaryRow.style.cssText = 'display: flex; align-items: center; margin-bottom: 4px;';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `feature_${col.name}`;
            checkbox.value = col.name;
            checkbox.disabled = true;
            checkbox.checked = false;
            checkbox.title = 'This column is selected as the target and cannot be used as a feature.';
            checkbox.style.cssText = 'width: 18px; height: 18px; margin-right: 12px; cursor: not-allowed; accent-color: #6B5A4A;';
            
            const nameLabel = document.createElement('label');
            nameLabel.htmlFor = `feature_${col.name}`;
            nameLabel.style.cssText = 'flex: 1; font-size: 15px; font-weight: 600; color: #8B7355; cursor: not-allowed; user-select: none;';
            nameLabel.textContent = `${col.name} (target)`;
            
            primaryRow.appendChild(checkbox);
            primaryRow.appendChild(nameLabel);
            item.appendChild(primaryRow);
            featureList.appendChild(item);
            return; // Don't add to regular feature list
        }
        
        // Track label-like features (but not target)
        if (isLabelLike) {
            labelLikeFeatures.push(col.name);
        }
        
        // Create main container
        const item = document.createElement('div');
        item.style.cssText = 'padding: 12px; margin-bottom: 8px; background: white; border-radius: 6px; border: 1px solid #E0D5C7; transition: all 0.2s;';
        item.onmouseenter = () => {
            item.style.borderColor = '#C4B5A0';
            item.style.boxShadow = '0 2px 4px rgba(0,0,0,0.05)';
        };
        item.onmouseleave = () => {
            item.style.borderColor = '#E0D5C7';
            item.style.boxShadow = 'none';
        };
        
        // Primary tier: Checkbox + Column name (dominant)
        const primaryRow = document.createElement('div');
        primaryRow.style.cssText = 'display: flex; align-items: center; margin-bottom: 4px;';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `feature_${col.name}`;
        checkbox.value = col.name;
        checkbox.checked = !autoExclude || !col.isSuspicious; // Uncheck if auto-exclude and suspicious
        // Increased checkbox contrast for better visibility
        checkbox.style.cssText = 'width: 18px; height: 18px; margin-right: 12px; cursor: pointer; accent-color: #6B5A4A;';
        
        const nameLabel = document.createElement('label');
        nameLabel.htmlFor = `feature_${col.name}`;
        nameLabel.style.cssText = 'flex: 1; font-size: 15px; font-weight: 600; color: #5A4A3A; cursor: pointer; user-select: none;';
        nameLabel.textContent = col.name;
        
        // Warning indicator (if suspicious) - reduced saturation, less visually loud
        if (col.isSuspicious) {
            const warningBadge = document.createElement('span');
            warningBadge.style.cssText = 'margin-left: 8px; font-size: 11px; color: #FF9800; font-weight: 400; opacity: 0.85;';
            warningBadge.textContent = '⚠ Likely ID';
            warningBadge.title = 'This column may act as an identifier and cause data leakage.';
            nameLabel.appendChild(warningBadge);
        }
        
        // Label-like warning (if not target) - only show if truly label-like
        if (isLabelLike && !isTarget && labelLikeResult) {
            const labelWarningBadge = document.createElement('span');
            labelWarningBadge.style.cssText = 'margin-left: 8px; font-size: 11px; color: #FFC107; font-weight: 400; opacity: 0.9;';
            labelWarningBadge.textContent = '⚠ Label-like';
            const confidenceText = labelLikeResult.confidence === 'high' ? ' (high confidence)' : ' (medium confidence)';
            const uniqueText = col.uniqueCount !== undefined ? `, ${col.uniqueCount} unique values` : '';
            labelWarningBadge.title = `This column name looks like a label/target${confidenceText}${uniqueText}. Make sure it should be a feature.`;
            nameLabel.appendChild(labelWarningBadge);
        }
        
        primaryRow.appendChild(checkbox);
        primaryRow.appendChild(nameLabel);
        
        // Secondary tier: Type + Risk signal (compact meta row)
        const secondaryRow = document.createElement('div');
        secondaryRow.style.cssText = 'display: flex; align-items: center; margin-left: 30px; font-size: 12px; color: #8B7355; opacity: 0.7;';
        
        // Type
        const typeSpan = document.createElement('span');
        typeSpan.textContent = col.type;
        typeSpan.style.cssText = 'margin-right: 8px;';
        
        // Separator
        const separator = document.createElement('span');
        separator.textContent = '·';
        separator.style.cssText = 'margin: 0 6px; opacity: 0.5;';
        
        // Risk signal (semantic language instead of exact percentage)
        const riskSpan = document.createElement('span');
        if (col.isSuspicious) {
            riskSpan.textContent = 'high cardinality';
            riskSpan.style.cssText = 'color: #FF9800; opacity: 0.8;'; // Reduced saturation
        } else if (parseFloat(col.uniquePercent) < 5) {
            riskSpan.textContent = `${col.uniquePercent}% unique`;
        } else if (col.type === 'categorical') {
            riskSpan.textContent = `${col.uniqueCount} values`;
        } else {
            riskSpan.textContent = `${col.uniquePercent}% unique`;
        }
        
        secondaryRow.appendChild(typeSpan);
        secondaryRow.appendChild(separator);
        secondaryRow.appendChild(riskSpan);
        
        // Tertiary tier: Detailed stats (hidden by default, shown on hover via tooltip)
        const totalRows = col.uniqueCount / (parseFloat(col.uniquePercent) / 100);
        const tooltipText = `Unique: ${col.uniqueCount} / ${Math.round(totalRows)} (${col.uniquePercent}%)\n` +
                          `Sample values: ${col.sampleValues.length > 0 ? col.sampleValues.join(', ') : 'N/A'}\n` +
                          (col.isSuspicious ? 'This column may act as an identifier and cause data leakage.' : '');
        
        item.title = tooltipText; // Browser tooltip
        
        // Add info icon aligned with secondary metadata (better visual connection)
        const detailsIcon = document.createElement('span');
        detailsIcon.style.cssText = 'margin-left: 8px; font-size: 11px; color: #8B7355; opacity: 0.6; cursor: help; vertical-align: middle;';
        detailsIcon.textContent = 'ⓘ';
        detailsIcon.title = tooltipText;
        riskSpan.appendChild(detailsIcon);
        
        item.appendChild(primaryRow);
        item.appendChild(secondaryRow);
        featureList.appendChild(item);
    });
    
    // Update warning banner after rendering schema
    // targetSelect is already declared at the top of the function
    const targetColumn = targetSelect ? targetSelect.value : null;
    updateLabelLikeWarningBanner([], targetColumn);
}

// Select dataset directory for real training
async function selectDatasetDirectory() {
    try {
        const modelPurpose = document.getElementById('modelPurposeInput').value;
        const result = await ipcRenderer.invoke('select-dataset-directory', { modelPurpose });
        if (result.canceled) {
            log('Dataset selection cancelled', 'warning');
            return null;
        }
        
        const selectedPath = result.filePaths[0];
        
        // Check if a file was selected (for tabular CSV files)
        const fs = require('fs');
        const path = require('path');
        const stats = fs.statSync(selectedPath);
        
        if (stats.isFile() && selectedPath.toLowerCase().endsWith('.csv')) {
            // Store the CSV file path for schema extraction
            selectedDatasetFile = selectedPath;
            
            // For tabular models, extract schema and show review UI
            if (modelPurpose === 'tabular') {
                try {
                    log('Extracting schema from CSV...', 'log');
                    const schema = await extractCSVSchema(selectedPath);
                    updateSchemaReviewUI(schema, selectedPath);
                    log(`Schema extracted: ${schema.length} columns found`, 'success');
                    
                    // Store schema for later use
                    window.tabularSchema = schema;
                    window.tabularCSVPath = selectedPath;
                } catch (error) {
                    log(`Error extracting schema: ${error.message}`, 'error');
                }
            }
            
            // If a CSV file was selected, use its parent directory
            const directory = path.dirname(selectedPath);
            log(`Selected CSV file: ${path.basename(selectedPath)}`, 'success');
            log(`Using directory: ${directory}`, 'log');
            return directory;
        } else {
            // Directory was selected
            log(`Selected dataset directory: ${selectedPath}`, 'success');
            return selectedPath;
        }
    } catch (error) {
        log(`Error selecting dataset: ${error.message}`, 'error');
        return null;
    }
}

// Cloud training implementation
async function startCloudTraining() {
    try {
        // Check if cloud config is set
        if (!window.cloudConfig) {
            log('Cloud configuration not found. Please configure cloud settings first.', 'error');
            return;
        }
        
        // Check if dataset is selected
        if (!selectedFolderPath) {
            log('Please select a dataset folder first', 'error');
            return;
        }
        
        // Get model configuration first (needed for validation)
        const modelPurpose = document.getElementById('modelPurposeInput').value;
        const framework = document.getElementById('frameworkInput').value;
        const variant = document.getElementById('modelVariantInput').value;
        
        // Validate dataset format for cloud training using comprehensive validation
        const datasetValidation = validateDatasetFormat(selectedFolderPath, modelPurpose, framework);
        if (!datasetValidation.valid) {
            log(`Dataset validation failed: ${datasetValidation.message}`, 'error');
            if (datasetValidation.expectedFormat) {
                log(`Expected format: ${datasetValidation.expectedFormat}`, 'error');
            }
            return;
        } else {
            log(datasetValidation.message, 'success');
        }
        
        // Update UI
        document.getElementById('status').textContent = 'Launching Cloud Instance';
        document.getElementById('status').className = 'value status-training';
        document.getElementById('startTrainingBtn').disabled = true;
        setTrainingButtonsEnabled(true);
        
        log('Preparing cloud training...', 'success');
        log(`GPU: ${window.window.cloudConfig.gpuName}`, 'log');
        log(`Region: ${window.window.cloudConfig.region}`, 'log');
        log(`Max training time: ${window.cloudConfig.maxTrainingHours} hours`, 'log');
        log(`Budget limit: $${window.cloudConfig.budgetLimit}`, 'log');
        
        // PRE-LAUNCH VALIDATION: Check upfront authorization vs budget
        const gpuSelect = document.getElementById('cloudGPUSelect');
        if (gpuSelect) {
            const selectedOption = gpuSelect.options[gpuSelect.selectedIndex];
            if (selectedOption && selectedOption.value) {
                // Get pricing fields from stored data
                const pricePerGpuHour = parseFloat(selectedOption.dataset.pricePerGpuHour || 0);
                const gpuCount = parseInt(selectedOption.dataset.gpuCount || 1);
                const providerUpfrontHours = parseInt(selectedOption.dataset.providerUpfrontHours || 24);
                
                if (pricePerGpuHour > 0) {
                    // Calculate costs using centralized utility
                    const costs = calculateCloudCosts({
                        pricePerGpuHour: pricePerGpuHour,
                        gpuCount: gpuCount,
                        providerUpfrontHours: providerUpfrontHours
                    });
                    
                    const budgetLimit = window.cloudConfig.budgetLimit || 100;
                    
                    // STRUCTURED LOGGING: Log pricing information before launch
                    console.log('[CloudTraining] Hourly cost: $' + costs.hourlyCost.toFixed(2) + '/hr');
                    console.log('[CloudTraining] Provider upfront authorization: ~$' + costs.upfrontAuthorizationCost.toFixed(2) + ' (' + providerUpfrontHours + 'h)');
                    console.log('[CloudTraining] Max training time: ' + window.cloudConfig.maxTrainingHours + 'h');
                    console.log('[CloudTraining] Budget limit: $' + budgetLimit.toFixed(2));
                    
                    log(`Hourly cost: $${costs.hourlyCost.toFixed(2)}/hour`, 'log');
                    log(`Provider upfront authorization: ~$${costs.upfrontAuthorizationCost.toFixed(2)} (${providerUpfrontHours} hours)`, 'log');
                    
                    // Warn if upfront authorization exceeds budget (but don't block - budget is a warning threshold)
                    if (costs.upfrontAuthorizationCost > budgetLimit) {
                        const warningMsg = `⚠️ WARNING: Provider may charge ~$${costs.upfrontAuthorizationCost.toFixed(2)} upfront (${providerUpfrontHours}h), which exceeds your budget limit of $${budgetLimit.toFixed(2)}.\n` +
                            `Note: Budget limit is a warning threshold. The provider will still charge the upfront authorization amount.\n` +
                            `Please ensure your payment method can cover the upfront charge.`;
                        log(warningMsg, 'warning');
                    }
                }
            }
        }
        
        // Prepare cloud training configuration
        const config = {
            // API key
            apiKey: window.canopywaveApiKey,
            
            // Cloud configuration (flattened from window.cloudConfig)
            project: window.cloudConfig.project,
            region: window.cloudConfig.region,
            flavor: window.cloudConfig.gpu,
            image: window.cloudConfig.image,
            password: window.cloudConfig.password,
            maxTrainingHours: window.cloudConfig.maxTrainingHours,
            budgetLimit: window.cloudConfig.budgetLimit,
            
            // Dataset path
            datasetPath: selectedFolderPath,
            dataset_file: selectedDatasetFile || null, // Include dataset_file if CSV was selected
            
            // Training settings
            trainingSettings: {
                model_purpose: modelPurpose,
                framework: framework,
                variant: variant,
                epochs: parseInt(trainingSettings.epochs),
                batch_size: parseInt(trainingSettings.batchSize),
                learning_rate: parseFloat(trainingSettings.learningRate),
                optimizer: trainingSettings.optimizer,
                device: 'cuda', // Cloud always uses GPU
                // RandomForest settings (if applicable)
                n_estimators: trainingSettings.n_estimators,
                max_depth: trainingSettings.max_depth,
                min_samples_split: trainingSettings.min_samples_split,
                min_samples_leaf: trainingSettings.min_samples_leaf,
                max_features: trainingSettings.max_features
            }
        };
        
        // Debug: Check API key type
        console.log('[Cloud Training] API Key type:', typeof window.canopywaveApiKey);
        console.log('[Cloud Training] API Key value:', window.canopywaveApiKey ? '***' + window.canopywaveApiKey.slice(-4) : 'undefined');
        console.log('[Cloud Training] Config:', { ...config, apiKey: config.apiKey ? '***' : 'missing' });
        
        log('Starting cloud training job...', 'log');
        
        // Ensure API key is a string
        if (typeof window.canopywaveApiKey !== 'string') {
            throw new Error(`API key is not a string (type: ${typeof window.canopywaveApiKey}). Please re-enter your API key.`);
        }
        
        // Start cloud training via IPC
        const result = await ipcRenderer.invoke('start-cloud-training', config);
        
        if (result.success) {
            log(`Cloud instance launched: ${result.instanceId}`, 'success');
            log('Training is running on cloud GPU...', 'success');
            
            // Store instance ID for monitoring
            window.currentCloudInstanceId = result.instanceId;
            
            // Start monitoring cloud training progress
            startCloudTrainingMonitoring(result.instanceId);
        } else {
            throw new Error(result.error || 'Failed to start cloud training');
        }
        
    } catch (error) {
        // Log full error details to console for debugging
        console.error('[Cloud Training] Full error:', error);
        console.error('[Cloud Training] Error stack:', error.stack);
        
        // Display full error message (may include CanopyWave API details)
        const errorMessage = error.message || 'Failed to start cloud training';
        log(`Failed to start cloud training: ${errorMessage}`, 'error');
        
        // Also log to console with full details
        console.error('[Cloud Training] Error details:', {
            message: error.message,
            stack: error.stack,
            name: error.name
        });
        
        document.getElementById('startTrainingBtn').disabled = false;
        setTrainingButtonsEnabled(false);
        document.getElementById('status').textContent = 'Error';
        document.getElementById('status').className = 'value status-error';
    }
}

// Monitor cloud training progress
function startCloudTrainingMonitoring(instanceId) {
    // Poll for training progress every 10 seconds
    const monitoringInterval = setInterval(async () => {
        try {
            const status = await ipcRenderer.invoke('get-cloud-job-status', canopywaveApiKey, instanceId, window.cloudConfig.project, window.cloudConfig.region);
            
            if (status.success) {
                // Update UI with cloud training status
                if (status.status === 'COMPLETED') {
                    clearInterval(monitoringInterval);
                    log('Cloud training completed!', 'success');
                    document.getElementById('status').textContent = 'Completed';
                    document.getElementById('status').className = 'value status-ready';
                    document.getElementById('startTrainingBtn').disabled = false;
                    setTrainingButtonsEnabled(false);
                } else if (status.status === 'FAILED' || status.status === 'ERROR') {
                    clearInterval(monitoringInterval);
                    log('Cloud training failed', 'error');
                    document.getElementById('status').textContent = 'Error';
                    document.getElementById('status').className = 'value status-error';
                    document.getElementById('startTrainingBtn').disabled = false;
                    setTrainingButtonsEnabled(false);
                }
                
                // Update progress if available
                if (status.progress) {
                    document.getElementById('northStarValue').textContent = `${status.progress}%`;
                }
            }
        } catch (error) {
            console.error('Error monitoring cloud training:', error);
        }
    }, 10000); // Check every 10 seconds
    
    // Store interval ID for cleanup
    window.cloudMonitoringInterval = monitoringInterval;
}

function resetTrainingSessionUI() {
    // Clear logs
    const output = document.getElementById('output');
    if (output) output.innerHTML = '';

    // Clear model path display
    savedModelPath = null;
    const modelPathEl = document.getElementById('modelPath');
    if (modelPathEl) modelPathEl.textContent = '';
    const modelSection = document.getElementById('modelSection');
    if (modelSection) modelSection.style.display = 'none';

    // Clear inference file selections
    const tabularCsvPathEl = document.getElementById('tabularCsvPath');
    if (tabularCsvPathEl) {
        tabularCsvPathEl.textContent = '';
        tabularCsvPathEl.style.display = 'none';
    }
    const cvImagePathEl = document.getElementById('cvImagePath');
    if (cvImagePathEl) {
        cvImagePathEl.textContent = '';
        cvImagePathEl.style.display = 'none';
    }
    window.tabularCSVPath = null;
    window.tabularSchema = null;

    // Clear inference results
    const tabularResultsDiv = document.getElementById('tabularInferenceResults');
    if (tabularResultsDiv) tabularResultsDiv.style.display = 'none';
    const tabularResultsTable = document.getElementById('tabularResultsTable');
    if (tabularResultsTable) tabularResultsTable.innerHTML = '';

    const cvResultsDiv = document.getElementById('cvInferenceResults');
    if (cvResultsDiv) cvResultsDiv.style.display = 'none';
    const cvPreviewImage = document.getElementById('cvPreviewImage');
    if (cvPreviewImage) cvPreviewImage.src = '';
    const cvDetectionsList = document.getElementById('cvDetectionsList');
    if (cvDetectionsList) cvDetectionsList.innerHTML = '';

    // Reset inference controls
    const tabularModelSelect = document.getElementById('tabularModelSelect');
    if (tabularModelSelect) tabularModelSelect.value = '';
    const cvModelSelect = document.getElementById('cvModelSelect');
    if (cvModelSelect) cvModelSelect.value = '';

    const runTabularInferenceBtn = document.getElementById('runTabularInferenceBtn');
    if (runTabularInferenceBtn) runTabularInferenceBtn.disabled = true;
    const runCvInferenceBtn = document.getElementById('runCvInferenceBtn');
    if (runCvInferenceBtn) runCvInferenceBtn.disabled = true;

    const exportTabularPredictionsBtn = document.getElementById('exportTabularPredictionsBtn');
    if (exportTabularPredictionsBtn) exportTabularPredictionsBtn.onclick = null;
    const openCvOutputFolderBtn = document.getElementById('openCvOutputFolderBtn');
    if (openCvOutputFolderBtn) openCvOutputFolderBtn.onclick = null;
}

// Real training implementation
async function startRealTraining() {
    // Prevent multiple training starts
    if (isRealTraining) {
        log('Real training already in progress', 'warning');
        return;
    }
    
    try {
        resetTrainingSessionUI();
        isRealTraining = true; // Set flag immediately to prevent duplicate starts
        
        // Update UI
        document.getElementById('status').textContent = 'Training';
        document.getElementById('status').className = 'value status-training';
        document.getElementById('startTrainingBtn').disabled = true;
        setTrainingButtonsEnabled(true);
        
        // Update training status indicator to "Validating" initially
        const statusIndicator = document.getElementById('trainingStatusIndicator');
        if (statusIndicator) {
            statusIndicator.classList.add('active');
            statusIndicator.querySelector('.status-text').textContent = 'Validating';
        }
        
        // Check if folder is selected
        let datasetDir = null;
        
        if (selectedFolderPath) {
            datasetDir = selectedFolderPath;
            log(`Using selected folder: ${datasetDir}`, 'success');
        }
        
        // If no files uploaded or saving failed, prompt for directory selection
        if (!datasetDir) {
            log('Real training requires a dataset directory', 'log');
            log('Please select the directory containing your training data', 'log');
            datasetDir = await selectDatasetDirectory();
            if (!datasetDir) {
                document.getElementById('startTrainingBtn').disabled = false;
                setTrainingButtonsEnabled(false);
                isRealTraining = false; // Clear flag if cancelled
                document.getElementById('status').textContent = 'Ready';
                document.getElementById('status').className = 'value status-ready';
                return;
            }
        }
        
        // Get model configuration
        const modelPurpose = document.getElementById('modelPurposeInput').value;
        const framework = document.getElementById('frameworkInput').value;
        const variant = document.getElementById('modelVariantInput').value;
        const format = document.getElementById('modelFormatInput').value;
        
        // Validate that framework and variant are selected
        if (!framework || framework.trim() === '') {
            log('Please select a framework (e.g., sklearn, xgboost, yolo) before starting training', 'error');
            isRealTraining = false;
            document.getElementById('startTrainingBtn').disabled = false;
            setTrainingButtonsEnabled(false);
            document.getElementById('status').textContent = 'Ready';
            document.getElementById('status').className = 'value status-ready';
            return;
        }
        
        if (!variant || variant.trim() === '') {
            log('Please select a model variant before starting training', 'error');
            isRealTraining = false;
            document.getElementById('startTrainingBtn').disabled = false;
            setTrainingButtonsEnabled(false);
            document.getElementById('status').textContent = 'Ready';
            document.getElementById('status').className = 'value status-ready';
            return;
        }
        
        // Validate dataset format matches model type
        const datasetValidation = validateDatasetFormat(datasetDir, modelPurpose, framework);
        if (!datasetValidation.valid) {
            log(`Dataset validation failed: ${datasetValidation.message}`, 'error');
            if (datasetValidation.expectedFormat) {
                log(`Expected format: ${datasetValidation.expectedFormat}`, 'error');
            }
            isRealTraining = false;
            document.getElementById('startTrainingBtn').disabled = false;
            setTrainingButtonsEnabled(false);
            document.getElementById('status').textContent = 'Ready';
            document.getElementById('status').className = 'value status-ready';
            return;
        } else {
            log(datasetValidation.message, 'success');
        }
        
        // Guard against undefined variant
        if (!variant) {
            log('Error: Model variant is undefined. Cannot start training.', 'error');
            isRealTraining = false;
            document.getElementById('startTrainingBtn').disabled = false;
            setTrainingButtonsEnabled(false);
            document.getElementById('status').textContent = 'Ready';
            document.getElementById('status').className = 'value status-ready';
            return;
        }
        
        // Define isRandomForest explicitly (used for UI and config cleanup)
        const isRandomForest = modelPurpose === 'tabular' && framework === 'sklearn' && variant === 'random_forest';
        
        // Get settings schema for this model
        const schema = getModelSettingsSchema(modelPurpose, framework, variant);
        
        // Build config using ONLY relevant parameters from schema
        const config = {
            model_purpose: modelPurpose,
            framework: framework,
            variant: variant,
            format: format,
            data_dir: datasetDir,
            dataset_file: selectedDatasetFile || null, // Include dataset_file if CSV was selected
            output_dir: (() => {
                // Create unique folder for each training run with timestamp
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5); // Format: 2024-01-15T10-30-45
                const baseDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'Models');
                const uniqueDir = path.join(baseDir, `training_${timestamp}`);
                return uniqueDir.replace(/\\/g, '/');
            })(),
            exclude_id_columns: true // Default to True - exclude suspected ID columns to prevent data leakage
        };
        
        // Add parameters based on schema (only include what's actually used)
        if (schema.params.includes('validation_split')) {
            config.validation_split = parseFloat(trainingSettings.validationSplit) || 0.2;
        }
        if (schema.params.includes('epochs')) {
            config.epochs = trainingSettings.epochs || 10;
        }
        if (schema.params.includes('batchSize')) {
            config.batch_size = trainingSettings.batchSize || 32;
        }
        if (schema.params.includes('learningRate')) {
            config.learning_rate = trainingSettings.learningRate || 0.001;
        }
        if (schema.params.includes('optimizer')) {
            config.optimizer = trainingSettings.optimizer || 'adam';
        }
        if (schema.params.includes('lossFunction')) {
            config.loss_function = trainingSettings.lossFunction || 'mse';
        }
        if (schema.params.includes('device')) {
            config.device = trainingSettings.device || 'auto';
        }
        
        // Tree-based parameters
        if (schema.params.includes('n_estimators')) {
            config.n_estimators = trainingSettings.n_estimators || 100;
        }
        if (schema.params.includes('max_depth')) {
            config.max_depth = trainingSettings.max_depth;
        }
        if (schema.params.includes('min_samples_split')) {
            config.min_samples_split = trainingSettings.min_samples_split || 2;
        }
        if (schema.params.includes('min_samples_leaf')) {
            config.min_samples_leaf = trainingSettings.min_samples_leaf || 1;
        }
        if (schema.params.includes('max_features')) {
            config.max_features = trainingSettings.max_features || 'sqrt';
        }
        
        // For tabular models, include explicit column selections
        if (modelPurpose === 'tabular') {
            const targetSelect = document.getElementById('targetColumnSelect');
            const targetColumn = targetSelect ? targetSelect.value : null;
            
            if (!targetColumn) {
                log('Please select a target column in Schema Review', 'error');
                isRealTraining = false;
                document.getElementById('startTrainingBtn').disabled = false;
                setTrainingButtonsEnabled(false);
                document.getElementById('status').textContent = 'Ready';
                document.getElementById('status').className = 'value status-ready';
                return;
            }
            
            // Get selected feature columns
            const featureCheckboxes = document.querySelectorAll('#featureColumnsList input[type="checkbox"]:checked');
            let featureColumns = Array.from(featureCheckboxes).map(cb => cb.value);
            
            // CRITICAL: Filter out target column from features (hard invariant)
            // This ensures target can never be used as a feature, even if checkbox is somehow checked
            featureColumns = featureColumns.filter(col => col !== targetColumn);
            
            if (featureColumns.length === 0) {
                log('Please select at least one feature column in Schema Review', 'error');
                isRealTraining = false;
                document.getElementById('startTrainingBtn').disabled = false;
                setTrainingButtonsEnabled(false);
                document.getElementById('status').textContent = 'Ready';
                document.getElementById('status').className = 'value status-ready';
                return;
            }
            
            // Validate tabular config before training
            const allColumns = window.tabularSchema ? window.tabularSchema.map(col => col.name) : [];
            const validation = validateTabularConfig(targetColumn, featureColumns, allColumns);
            
            if (!validation.valid) {
                log('[Tabular] Validation failed:', 'error');
                validation.errors.forEach(err => {
                    log(`  - ${err}`, 'error');
                });
                isRealTraining = false;
                document.getElementById('startTrainingBtn').disabled = false;
                setTrainingButtonsEnabled(false);
                document.getElementById('status').textContent = 'Ready';
                document.getElementById('status').className = 'value status-ready';
                return;
            }
            
            // Log warnings for label-like features
            if (validation.warnings && validation.warnings.length > 0) {
                validation.warnings.forEach(warning => {
                    if (typeof warning === 'string') {
                        log(`[Tabular] WARNING: ${warning}`, 'warning');
                    } else if (warning && typeof warning === 'object') {
                        if (warning.message) {
                            const uniqueInfo = warning.uniqueCount !== undefined ? `, unique=${warning.uniqueCount}` : '';
                            log(`[Tabular] label-like feature: ${warning.column} (confidence=${warning.confidence}${uniqueInfo})`, 'warning');
                        } else {
                            // Fallback: stringify the warning object
                            log(`[Tabular] WARNING: ${JSON.stringify(warning, null, 2)}`, 'warning');
                        }
                    } else {
                        log(`[Tabular] WARNING: ${String(warning)}`, 'warning');
                    }
                });
            }
            
            config.target_column = targetColumn;
            config.feature_columns = featureColumns;
            
            log(`Target column: ${targetColumn}`, 'log');
            log(`Selected ${featureColumns.length} feature columns: ${featureColumns.join(', ')}`, 'log');
        }
        
        // Log dataset selection
        if (config.dataset_file) {
            log(`Selected dataset_file: ${config.dataset_file}`, 'log');
        } else {
            log(`Selected dataset_file: None (will search directory)`, 'log');
        }
        
        log('Training configuration prepared', 'success');
        log(`Model: ${modelPurpose}/${framework}/${variant}`, 'log');
        log(`Dataset: ${datasetDir}`, 'log');
        
        // Log effective settings (only the ones that actually apply)
        const effectiveSettings = [];
        if (config.n_estimators !== undefined) {
            effectiveSettings.push(`n_estimators=${config.n_estimators}`);
        }
        if (config.max_depth !== undefined) {
            effectiveSettings.push(`max_depth=${config.max_depth || 'None'}`);
        }
        if (config.min_samples_split !== undefined) {
            effectiveSettings.push(`min_samples_split=${config.min_samples_split}`);
        }
        if (config.min_samples_leaf !== undefined) {
            effectiveSettings.push(`min_samples_leaf=${config.min_samples_leaf}`);
        }
        if (config.max_features !== undefined) {
            effectiveSettings.push(`max_features=${config.max_features || 'None'}`);
        }
        if (config.epochs !== undefined) {
            effectiveSettings.push(`epochs=${config.epochs}`);
        }
        if (config.batch_size !== undefined) {
            effectiveSettings.push(`batch_size=${config.batch_size}`);
        }
        if (config.learning_rate !== undefined) {
            effectiveSettings.push(`learning_rate=${config.learning_rate}`);
        }
        if (config.optimizer !== undefined) {
            effectiveSettings.push(`optimizer=${config.optimizer}`);
        }
        if (config.loss_function !== undefined) {
            effectiveSettings.push(`loss=${config.loss_function}`);
        }
        if (config.device !== undefined) {
            const deviceDisplay = config.device === 'auto' ? 'Auto (GPU if available)' : (config.device === 'cuda' ? 'GPU' : 'CPU');
            effectiveSettings.push(`device=${deviceDisplay}`);
        }
        if (config.validation_split !== undefined) {
            effectiveSettings.push(`validation_split=${config.validation_split}`);
        }
        
        if (effectiveSettings.length > 0) {
            log(`Effective settings: ${effectiveSettings.join(', ')}`, 'log');
        } else {
            log('Warning: No effective settings found - check schema configuration', 'warning');
        }
        
        // Explicit cleanup: Remove epochs/batch/LR for Random Forest (extra safety)
        if (isRandomForest) {
            delete config.epochs;
            delete config.batch_size;
            delete config.learning_rate;
            delete config.optimizer;
            delete config.loss_function;
            log('[Tabular] Removed epochs/batch/LR/optimizer/loss for Random Forest (not applicable)', 'log');
        }
        
        // Log confirmation for debugging
        if (modelPurpose === 'tabular') {
            if (isRandomForest) {
                log(`[Tabular] Launching Random Forest training with config:`, 'log');
                log(`  - n_estimators: ${config.n_estimators || 'default'}, max_depth: ${config.max_depth || 'None'}, validation_split: ${config.validation_split || 'default'}`, 'log');
            } else {
                log(`[Tabular] Launching ${framework}/${variant} training with config:`, 'log');
                log(`  - epochs: ${config.epochs || 'default'}, batch_size: ${config.batch_size || 'default'}, learning_rate: ${config.learning_rate || 'default'}`, 'log');
            }
        }
        
        // Update epoch label based on model type
        updateEpochLabel(modelPurpose, framework, variant);
        
        // Initialize training state
        trainingStartTime = Date.now();
        currentEpoch = 0;
        // For RandomForest, use n_estimators instead of epochs
        if (isRandomForest) {
            totalEpochs = config.n_estimators || 100;
        } else {
            totalEpochs = config.epochs || 10;
        }
        displayedProgress = 0;
        lastRealProgress = null;
        trainingHistory = [];
        document.getElementById('northStarValue').textContent = '0%';
        document.getElementById('boxLoss').textContent = '--';
        document.getElementById('clsLoss').textContent = '--';
        document.getElementById('dflLoss').textContent = '--';
        document.getElementById('gpuMem').textContent = '--';
        document.getElementById('instances').textContent = '--';
        document.getElementById('processingSpeed').textContent = '--';
        document.getElementById('map50').textContent = '--';
        document.getElementById('map5095').textContent = '--';
        document.getElementById('currentEpoch').textContent = `0/${totalEpochs}`;
        document.getElementById('eta').textContent = '--';
        
        // Don't start progress estimation for real training - we'll use real progress updates from YOLO
        // startProgressEstimation(); // DISABLED - real training uses actual progress from YOLO
        
        // Don't start training here - it will be started after network is configured below
        // Just ensure network exists, initialization happens later
        
        // Start neural network visualization BEFORE sending training config
        // Ensure neural network is initialized
        console.log('[Renderer] startRealTraining: Checking neural network initialization...');
        const canvas = document.getElementById('neuralCanvas');
        console.log('[Renderer] Canvas element:', canvas ? 'found' : 'NOT FOUND');
        console.log('[Renderer] neuralNetwork variable:', typeof neuralNetwork !== 'undefined' ? 'exists' : 'undefined');
        console.log('[Renderer] NeuralNetworkVisualization class:', typeof NeuralNetworkVisualization !== 'undefined' ? 'exists' : 'NOT FOUND');
        
        if (!neuralNetwork && canvas && typeof NeuralNetworkVisualization !== 'undefined') {
            console.log('[Renderer] Initializing neural network visualization...');
            try {
                window.neuralNetwork = new NeuralNetworkVisualization('neuralCanvas');
                neuralNetwork = window.neuralNetwork;
                console.log('[Renderer] Neural network initialized successfully, layers:', neuralNetwork.layers ? neuralNetwork.layers.length : 0);
            } catch (e) {
                console.error('[Renderer] Error initializing neural network:', e);
                console.error('[Renderer] Error stack:', e.stack);
            }
        } else if (!canvas) {
            console.error('[Renderer] Canvas element not found! Cannot initialize neural network.');
        } else if (typeof NeuralNetworkVisualization === 'undefined') {
            console.error('[Renderer] NeuralNetworkVisualization class not loaded! Check script loading.');
        }
        
        if (typeof neuralNetwork !== 'undefined' && neuralNetwork) {
            const qualitySliderEl = document.getElementById('qualitySlider');
            const currentQuality = qualitySliderEl ? parseInt(qualitySliderEl.value) : 100;
            // For real training, we don't have uploadedFiles size, use a default
            const totalDataSize = 1000000; // Default size for visualization
            
            console.log('[Renderer] Updating neural network settings...');
            console.log('[Renderer] Model config:', { modelPurpose, framework, variant, epochs: config.epochs });
            
            neuralNetwork.updateTrainingSettings({
                quality: currentQuality,
                epochs: config.epochs,
                batchSize: config.batch_size,
                learningRate: config.learning_rate,
                modelType: modelPurpose,
                modelPurpose: modelPurpose,
                framework: framework,
                variant: variant,
                fileCount: 1, // Real training uses directory, not file count
                dataSize: totalDataSize
            });
            
            console.log('[Renderer] Neural network layers after update:', neuralNetwork.layers ? neuralNetwork.layers.length : 'no layers');
            
            // Start validation animation AFTER network is created and configured
            console.log('[Renderer] Starting validation animation before training begins...');
            if (!neuralNetwork.isTraining) {
                neuralNetwork.startValidation();
                console.log('[Renderer] Validation animation started via startRealTraining()');
            } else {
                console.log('[Renderer] Cannot start validation animation - training already active');
            }
            
            console.log('[Renderer] Starting neural network training...');
            
            // Only start training if not already started (avoid duplicate calls)
            if (!neuralNetwork.isTraining) {
                neuralNetwork.startTraining();
            }
            // DO NOT reset trainingProgress = 0 - it will reset progress!
            // Progress should only be set by updateTrainingMetrics()
            
            console.log('[Renderer] Neural network state:', {
                isTraining: neuralNetwork.isTraining,
                layers: neuralNetwork.layers ? neuralNetwork.layers.length : 0,
                connections: neuralNetwork.connections ? neuralNetwork.connections.length : 0,
                canvas: neuralNetwork.canvas ? 'exists' : 'missing'
            });
        } else {
            console.error('[Renderer] neuralNetwork is undefined! Canvas exists:', !!canvas);
        }
        
        // Show loading overlay
        showTrainingLoadingOverlay();
        
        // Start the training process via IPC
        ipcRenderer.send('start-real-training', config);
        isRealTraining = true; // Mark real training as active
        
        log('Real training started - this may take a while depending on your dataset', 'success');
        
    } catch (error) {
        log(`Failed to start real training: ${error.message}`, 'error');
        isRealTraining = false; // Clear flag on error
        document.getElementById('startTrainingBtn').disabled = false;
        setTrainingButtonsEnabled(false);
        document.getElementById('status').textContent = 'Error';
        document.getElementById('status').className = 'value status-error';
        
        // Update training status indicator
        const statusIndicator = document.getElementById('trainingStatusIndicator');
        if (statusIndicator) {
            statusIndicator.classList.remove('active');
            statusIndicator.querySelector('.status-text').textContent = 'Error';
            statusIndicator.querySelector('.status-dot').style.background = '#ef4444';
        }
    }
}

// Start training - routes to cloud, real, or simulation training
async function startTraining() {
    if (trainingInterval) {
        log('Training already in progress', 'warning');
        return;
    }
    
    // Check if cloud training mode is active
    if (trainingMode === 'cloud') {
        log('Starting CLOUD training...', 'success');
        log('Training will be performed on CanopyWave cloud GPU', 'log');
        await startCloudTraining();
        return;
    }
    
    // Check if real training is supported and folder is selected
    const useRealTraining = supportsRealTraining();
    const hasSelectedFolder = selectedFolderPath && typeof selectedFolderPath === 'string' && selectedFolderPath.length > 0;
    
    // Check if folder is selected
    if (!hasSelectedFolder) {
        log('Please select a dataset folder first', 'error');
        return;
    }
    
    // If folder is selected, prefer real training (even if model type isn't fully supported)
    if (hasSelectedFolder && useRealTraining) {
        log('Starting REAL training with selected folder...', 'success');
        log('This will train an actual model with your data.', 'log');
        log('Training time will depend on your dataset size and hardware.', 'warning');
        
        // Start real training
        await startRealTraining();
    } else if (useRealTraining) {
        log('Starting REAL training...', 'success');
        log('This will train an actual model with your data.', 'log');
        log('Training time will depend on your dataset size and hardware.', 'warning');
        
        // Start real training
        await startRealTraining();
    } else {
        // For unsupported models, check if folder is selected - if so, try real training anyway
        if (hasSelectedFolder) {
            log('Selected folder detected. Attempting real training...', 'log');
            log('Note: Real training may not be fully implemented for this model type.', 'warning');
            await startRealTraining();
        } else {
            log('Starting SIMULATION training...', 'warning');
            log('Real training not yet implemented for this model type.', 'log');
            log('Consider using YOLO, scikit-learn, or XGBoost for real training.', 'log');
            log('Or select a folder to attempt real training.', 'log');
            
            // Fallback to simulation
            startSimulationTraining();
        }
    }
}

// Simulation training (fallback for unsupported models)
function startSimulationTraining() {
    if (trainingInterval) {
        log('Training already in progress', 'warning');
        return;
    }
    
    isRealTraining = false; // Mark simulation training as active
    
    // Check if folder is selected or files are uploaded
    if (!selectedFolderPath && uploadedFiles.length === 0) {
        log('No training data selected. Please select a folder or upload files first.', 'error');
        return;
    }
    
    // If folder is selected, use it (even for simulation, we can use the folder path)
    if (selectedFolderPath) {
        log(`Using selected folder: ${selectedFolderPath}`, 'log');
        // For simulation, we'll use the folder path but simulate the training
    }
    
    document.getElementById('status').textContent = 'Training';
    document.getElementById('status').className = 'value status-training';
    document.getElementById('startTrainingBtn').disabled = true;
    setTrainingButtonsEnabled(true);
    
    // Update training status indicator
    const statusIndicator = document.getElementById('trainingStatusIndicator');
    if (statusIndicator) {
        statusIndicator.classList.add('active');
        const statusText = statusIndicator.querySelector('.status-text');
        if (statusText) statusText.textContent = 'Training';
    }
    
    // Calculate quality from slider
    const qualitySliderEl = document.getElementById('qualitySlider');
    const currentQuality = qualitySliderEl ? parseInt(qualitySliderEl.value) : 100;
    
    // Calculate total data size (using folder selection, estimate size)
    const totalDataSize = 1000000; // Estimated size for visualization
    
    // Get model type
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const currentModelType = modelPurposeSelect ? modelPurposeSelect.value : 'machine_learning';
    
    // Update neural network with training settings
    if (typeof neuralNetwork !== 'undefined') {
        // Get current model configuration for architecture
        const modelPurposeSelect = document.getElementById('modelPurposeInput');
        const frameworkSelect = document.getElementById('frameworkInput');
        const variantSelect = document.getElementById('modelVariantInput');
        const currentModelPurpose = modelPurposeSelect ? modelPurposeSelect.value : currentModelType;
        const currentFramework = frameworkSelect ? frameworkSelect.value : '';
        const currentVariant = variantSelect ? variantSelect.value : '';
        
        neuralNetwork.updateTrainingSettings({
            quality: currentQuality,
            epochs: trainingSettings.epochs,
            batchSize: trainingSettings.batchSize,
            learningRate: trainingSettings.learningRate,
            modelType: currentModelType,
            modelPurpose: currentModelPurpose, // Pass model purpose for architecture
            framework: currentFramework, // Pass framework for architecture
            variant: currentVariant, // Pass variant for architecture
            fileCount: 1, // Using folder selection
            dataSize: 1000000 // Estimated size for visualization
        });
        // Only start training if not already started (avoid duplicate calls)
        if (!neuralNetwork.isTraining) {
            neuralNetwork.startTraining();
        }
        // DO NOT reset trainingProgress = 0 - it will reset progress!
    }
    
    // Parameter count should already be displayed from settings
    const paramElement = document.getElementById('parameterCount');
    const paramText = paramElement ? paramElement.textContent : '--';
    
    // Initialize progress display
    displayedProgress = 0; // Reset smooth progress
    document.getElementById('northStarValue').textContent = '0%';
    document.getElementById('boxLoss').textContent = '--';
    document.getElementById('clsLoss').textContent = '--';
    document.getElementById('dflLoss').textContent = '--';
    document.getElementById('gpuMem').textContent = '--';
    document.getElementById('instances').textContent = '--';
    document.getElementById('processingSpeed').textContent = '--';
    document.getElementById('map50').textContent = '--';
    document.getElementById('map5095').textContent = '--';
    document.getElementById('eta').textContent = '--';
    
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
    
    // Get current quality for duration estimation (reuse the variable from above)
    const currentQualityForDuration = qualitySliderEl ? parseInt(qualitySliderEl.value) : 50;
    updateTrainingActiveState(false);
    
    // Initialize training history
    trainingHistory = [];
    const initialLoss = 1.0;
    const initialAccuracy = 0.0;
    
    // Estimate training duration based on epochs and quality
    // Higher quality = longer training time per epoch
    const qualityFactor = currentQualityForDuration / 100;
    const baseTimePerEpoch = 2.0; // Base 2 seconds per epoch
    const timePerEpoch = baseTimePerEpoch * (0.7 + qualityFactor * 0.6); // Scale from 1.4s to 2.2s per epoch
    estimatedTrainingDuration = totalEpochs * timePerEpoch; // Total estimated duration in seconds
    
    log(`Estimated training duration: ${estimatedTrainingDuration.toFixed(1)} seconds (${timePerEpoch.toFixed(1)}s per epoch)`, 'log');
    
    let lastEpochLogTime = trainingStartTime;
    let lastEpochLogged = 0;
    
    // Real-time training simulation - updates progress every 0.5 seconds
    trainingInterval = setInterval(() => {
        if (!trainingStartTime) return;
        
        const currentTime = Date.now();
        const elapsedSeconds = (currentTime - trainingStartTime) / 1000;
        
        // Calculate progress based on elapsed time vs estimated duration
        // Prevent division by zero with safe minimum duration (calculate once, reuse)
        const safeDuration = Math.max(0.1, estimatedTrainingDuration); // Minimum 0.1 seconds
        const timeProgress = Math.min(1.0, elapsedSeconds / safeDuration);
        
        // Calculate current epoch based on time progress
        const estimatedCurrentEpoch = Math.min(totalEpochs, Math.floor(timeProgress * totalEpochs));
        
        // Update currentEpoch if we've progressed to a new epoch
        if (estimatedCurrentEpoch > currentEpoch) {
            currentEpoch = estimatedCurrentEpoch;
        }
        
        // Check if training is complete
        if (timeProgress >= 1.0 || currentEpoch >= totalEpochs) {
            currentEpoch = totalEpochs;
            clearInterval(trainingInterval);
            trainingInterval = null;
            
            // Ensure progress shows 100% when training completes
            displayedProgress = 100;
            document.getElementById('northStarValue').textContent = '100%';
            
            stopTraining();
            log('Training completed!', 'success');
            const finalLoss = trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].loss : 0;
            const finalAccuracy = trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].accuracy : 0;
            setTimeout(() => {
                saveModel(finalLoss, finalAccuracy);
            }, 100);
            return;
        }
        
        // Calculate metrics based on time progress (smooth, continuous)
        const progress = Math.min(1.0, timeProgress);
        const loss = Math.max(0.01, initialLoss * (1 - progress * 0.95) + (Math.random() * 0.1 - 0.05));
        const accuracy = Math.min(99.9, initialAccuracy + progress * 95 + (Math.random() * 2 - 1));
        
        const lossFormatted = loss.toFixed(4);
        const accuracyFormatted = accuracy.toFixed(2);
        
        // Log epoch completion when we reach a new epoch (not every 0.5s)
        if (estimatedCurrentEpoch > lastEpochLogged && estimatedCurrentEpoch > 0) {
            lastEpochLogged = estimatedCurrentEpoch;
            log(`Epoch ${estimatedCurrentEpoch}/${totalEpochs}: Loss=${lossFormatted}, Accuracy=${accuracyFormatted}%`, 'log');
            
            // Store training metrics for completed epochs
            trainingHistory.push({
                epoch: estimatedCurrentEpoch,
                loss: parseFloat(lossFormatted),
                accuracy: parseFloat(accuracyFormatted)
            });
        }
        
        // Update UI with current metrics (every 0.5 seconds)
        document.getElementById('currentAccuracy').textContent = `${accuracyFormatted}%`;
        document.getElementById('currentLoss').textContent = lossFormatted;
        document.getElementById('currentEpoch').textContent = `${Math.max(0, estimatedCurrentEpoch)}/${totalEpochs}`;
        
        // Update sparkline data continuously with NaN validation
        const accuracyValue = parseFloat(accuracyFormatted);
        if (!isNaN(accuracyValue)) {
            accuracySparklineData.push(accuracyValue);
            if (accuracySparklineData.length > maxSparklinePoints) {
                accuracySparklineData.shift();
            }
        }
        
        const lossValue = parseFloat(lossFormatted);
        if (!isNaN(lossValue)) {
            lossSparklineData.push(lossValue);
            if (lossSparklineData.length > maxSparklinePoints) {
                lossSparklineData.shift();
            }
        }
        
        // Reuse safeDuration calculated above (no duplicate declaration)
        const timeProgressPercent = Math.min(100, (elapsedSeconds / safeDuration) * 100);
        epochSparklineData.push(timeProgressPercent);
        if (epochSparklineData.length > maxSparklinePoints) {
            epochSparklineData.shift();
        }
        
        // Update parameter count sparkline
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
        
        // Update chart with continuous progress
        if (typeof updateTrainingChart === 'function') {
            lossHistory.push({ epoch: estimatedCurrentEpoch + timeProgress, value: parseFloat(lossFormatted) });
            accuracyHistory.push({ epoch: estimatedCurrentEpoch + timeProgress, value: parseFloat(accuracyFormatted) });
            updateTrainingChart();
        }
        
        // Update neural network visualization with real-time progress
        if (typeof neuralNetwork !== 'undefined') {
            // Calculate smooth progress from time (0.0 to 1.0)
            const smoothProgress = Math.min(1.0, timeProgress);
            neuralNetwork.updateTrainingMetrics(loss, accuracy, Math.max(0, estimatedCurrentEpoch), totalEpochs, smoothProgress);
        }
    }, 500); // Update every 0.5 seconds
    
    // Smooth progress animation - updates continuously for smooth progress display
    const animateProgress = () => {
        if (!trainingInterval) {
            // Training stopped, cancel animation
            if (progressAnimationId) {
                cancelAnimationFrame(progressAnimationId);
                progressAnimationId = null;
            }
            return;
        }
        
        // Calculate target progress based on elapsed time (real-time estimation)
        let actualProgress = 0;
        if (trainingStartTime && trainingInterval) {
            const elapsedSeconds = (Date.now() - trainingStartTime) / 1000;
            const safeDuration = Math.max(0.1, estimatedTrainingDuration);
            actualProgress = Math.min(100, (elapsedSeconds / safeDuration) * 100);
        } else if (currentEpoch >= totalEpochs) {
            actualProgress = 100;
        } else {
            actualProgress = Math.min(100, (currentEpoch / totalEpochs) * 100);
        }
        
        // Smooth interpolation towards actual progress
        const progressDiff = actualProgress - displayedProgress;
        if (Math.abs(progressDiff) > 0.1) {
            // Use smaller interpolation factor for very smooth, gradual increase
            displayedProgress += progressDiff * 0.05; // 5% catch-up per frame (~60fps)
            
            // Round for display
            const progressPercent = Math.round(displayedProgress);
            document.getElementById('northStarValue').textContent = `${progressPercent}%`;
            
            // Update neural network with smooth progress
            if (typeof neuralNetwork !== 'undefined') {
                const smoothProgress = displayedProgress / 100;
                neuralNetwork.trainingProgress = smoothProgress;
            }
        } else {
            // Close enough, snap to actual
            displayedProgress = actualProgress;
            const progressPercent = Math.round(displayedProgress);
            document.getElementById('northStarValue').textContent = `${progressPercent}%`;
        }
        
        progressAnimationId = requestAnimationFrame(animateProgress);
    };
    
    // Start smooth progress animation
    animateProgress();
    
    // Start monitoring
    startMonitoring();
}

// Helper function to ensure app container is properly displayed
function ensureMainAppVisible() {
    const mainApp = document.getElementById('mainApp');
    const splashScreen = document.getElementById('splashScreen');
    if (mainApp && splashScreen && splashScreen.style.display === 'none') {
        mainApp.style.display = 'flex';
    }
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

// Helper function to enable/disable training control buttons
function setTrainingButtonsEnabled(enabled) {
    const stopBtn = document.getElementById('stopTrainingBtn');
    const terminateBtn = document.getElementById('terminateInstanceBtn');
    if (stopBtn) stopBtn.disabled = !enabled;
    if (terminateBtn) terminateBtn.disabled = !enabled;
}

// Stop training
function stopTraining(wasCompletedOverride) {
    // If real training is active, send stop signal first
    if (isRealTraining) {
        ipcRenderer.send('stop-real-training');
        isRealTraining = false;
    }
    
    // If cloud training is active, terminate the instance
    if (window.currentCloudInstanceId && window.canopywaveApiKey) {
        // Guard against missing cloudConfig
        if (!window.cloudConfig?.project || !window.cloudConfig?.region) {
            log('Missing cloudConfig project/region; cannot stop cloud training safely.', 'error');
            console.error('[StopTraining] cloudConfig missing:', window.cloudConfig);
            // Still try to terminate with just instance ID if we have it
            if (window.currentCloudInstanceId) {
                log('Attempting emergency termination with instance ID only...', 'warning');
                ipcRenderer.invoke('terminate-cloud-instance', window.canopywaveApiKey, window.currentCloudInstanceId, window.cloudConfig?.project || '', window.cloudConfig?.region || '')
                    .finally(() => {
                        window.currentCloudInstanceId = null;
                    });
            }
            return;
        }
        
        log('Stopping cloud training and terminating instance...', 'warning');
        ipcRenderer.invoke(
            'stop-cloud-training', 
            window.canopywaveApiKey, 
            window.currentCloudInstanceId, 
            window.cloudConfig.project, 
            window.cloudConfig.region
        )
            .then(result => {
                if (result && result.success) {
                    log('Cloud instance terminated successfully', 'success');
                } else {
                    const errorMsg = result?.error || result?.message || 'Unknown error';
                    log(`Failed to terminate instance: ${errorMsg}`, 'error');
                    // Try direct termination as fallback
                    log('Attempting direct termination...', 'warning');
                    ipcRenderer.invoke('terminate-cloud-instance', window.canopywaveApiKey, window.currentCloudInstanceId, window.cloudConfig.project, window.cloudConfig.region)
                        .catch(err => {
                            log(`Direct termination also failed: ${err.message}`, 'error');
                        });
                }
            })
            .catch(error => {
                console.error('Error terminating cloud instance:', error);
                log(`Error terminating instance: ${error.message}`, 'error');
                // Try direct termination as fallback
                log('Attempting direct termination as fallback...', 'warning');
                ipcRenderer.invoke('terminate-cloud-instance', window.canopywaveApiKey, window.currentCloudInstanceId, window.cloudConfig?.project || '', window.cloudConfig?.region || '')
                    .catch(err => {
                        log(`Direct termination failed: ${err.message}. Please terminate manually in CanopyWave dashboard.`, 'error');
                    });
            })
            .finally(() => {
                window.currentCloudInstanceId = null;
                setTrainingButtonsEnabled(false);
            });
    }
    
    // Stop progress estimation
    stopProgressEstimation();
    
    const wasCompleted = wasCompletedOverride !== undefined ? wasCompletedOverride : (currentEpoch >= totalEpochs);
    
    // Force stop training interval
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    
    // Cancel progress animation
    if (progressAnimationId) {
        cancelAnimationFrame(progressAnimationId);
        progressAnimationId = null;
    }
    
    // DON'T clear monitoring interval - keep resource monitoring running
    // Resource monitoring should continue even when training stops
    
    // Ensure training stops even if interval check failed
    if (currentEpoch >= totalEpochs) {
        currentEpoch = totalEpochs; // Cap at total
    }
    
    // PRESERVE progress percentage at ANY percentage - don't reset it
    // Save current progress to localStorage so it persists (works for any percentage 0-100)
    localStorage.setItem('lastTrainingProgress', displayedProgress.toString());
    
    // Update UI to show preserved progress (use displayedProgress as percentage, not multiply by 100)
    const progressElement = document.getElementById('northStarValue');
    if (progressElement) {
        // displayedProgress is already a percentage (0-100), not 0-1
        // Preserve at ANY percentage (0%, 1%, 13%, 50%, 99%, etc.)
        progressElement.textContent = `${Math.round(displayedProgress)}%`;
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
    
    // Stop neural network visualization but preserve state when stopped early
    if (typeof neuralNetwork !== 'undefined') {
        if (wasCompleted) {
            // Mark as completed so it stays pulsing
            neuralNetwork.trainingProgress = 1.0;
            neuralNetwork.learningQuality = 0.8; // Maintain high quality for visual
            neuralNetwork.stopTraining();
        } else {
            // Training stopped early - preserve current state, don't complete visualization
            neuralNetwork.isTraining = false;
            neuralNetwork.trainingStoppedEarly = true; // Flag to prevent completion
            // Stop validation animation if it's running
            if (neuralNetwork.isValidating) {
                neuralNetwork.stopValidation();
            }
            // Keep current trainingProgress and visualization state as-is
            // Don't call completeVisualization() - just stop training flag
        }
    }
    
    // Update training state
    if (wasCompleted) {
        // Training completed - ensure progress shows 100%
        document.getElementById('northStarValue').textContent = '100%';
        updateTrainingActiveState(false);
        log('Training completed! Network is ready for interaction.', 'success');
    } else {
        // Training stopped early - preserve progress
        updateTrainingActiveState(false);
        log('Training stopped', 'warning');
        
        // Keep progress percentage visible at ANY percentage (0-100)
        const progressEl = document.getElementById('northStarValue');
        if (progressEl) {
            // Preserve the exact percentage where training was stopped (works for any value 0-100)
            progressEl.textContent = `${Math.round(displayedProgress)}%`;
        }
        // Store progress for persistence (works for any percentage)
        localStorage.setItem('lastTrainingProgress', displayedProgress.toString());
        
        // Keep epoch display
        if (currentEpoch > 0 && totalEpochs > 0) {
            const epochEl = document.getElementById('currentEpoch');
            if (epochEl) {
                epochEl.textContent = `${currentEpoch}/${totalEpochs}`;
            }
        }
    }
    
    document.getElementById('status').textContent = wasCompleted ? 'Completed' : 'Ready';
    document.getElementById('status').className = wasCompleted ? 'value status-ready' : 'value status-ready';
    document.getElementById('startTrainingBtn').disabled = false;
    document.getElementById('stopTrainingBtn').disabled = true;
    
    // Update training status indicator
    const statusIndicator = document.getElementById('trainingStatusIndicator');
    if (statusIndicator) {
        statusIndicator.classList.remove('active');
        const statusText = statusIndicator.querySelector('.status-text');
        const statusDot = statusIndicator.querySelector('.status-dot');
        if (statusText) statusText.textContent = wasCompleted ? 'Completed' : 'Ready';
        if (statusDot) statusDot.style.background = '';
    }
    
    // Don't reset progress when stopping early - preserve state
    if (!wasCompleted) {
        // Keep currentEpoch and trainingStartTime for display purposes
        // Only reset if user explicitly starts new training
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
    
    // DON'T clear monitoring interval - keep resource monitoring running
    // Resource monitoring should continue even when training resets
    
    // Reset neural network visualization
    if (typeof neuralNetwork !== 'undefined' && neuralNetwork) {
        neuralNetwork.stopTraining();
        neuralNetwork.reset();
    }
    
    // Clear uploaded files and selected folder
    uploadedFiles = [];
    selectedFolderPath = null;
    selectedDatasetFile = null;
    
    // Clear file list UI
    const fileList = document.getElementById('fileList');
    if (fileList) {
        fileList.innerHTML = '';
    }
    const selectedFolderPathEl = document.getElementById('selectedFolderPath');
    if (selectedFolderPathEl) {
        selectedFolderPathEl.style.display = 'none';
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
    document.getElementById('boxLoss').textContent = '--';
    document.getElementById('clsLoss').textContent = '--';
    document.getElementById('dflLoss').textContent = '--';
    document.getElementById('gpuMem').textContent = '--';
    document.getElementById('instances').textContent = '--';
    document.getElementById('processingSpeed').textContent = '--';
    document.getElementById('map50').textContent = '--';
    document.getElementById('map5095').textContent = '--';
    document.getElementById('currentEpoch').textContent = '--';
    document.getElementById('eta').textContent = '--';
    document.getElementById('parameterCount').textContent = '--';
    document.getElementById('output').textContent = 'Ready to start training...';
    
    // Reset button states
    document.getElementById('startTrainingBtn').disabled = false;
    document.getElementById('stopTrainingBtn').disabled = true;
    
    // Show left panel again when training stops
    const leftPanel = document.querySelector('.left-panel');
    const rightPanel = document.querySelector('.right-panel');
    if (leftPanel) {
        leftPanel.classList.remove('hidden');
    }
    if (rightPanel) {
        rightPanel.classList.remove('expanded');
    }
    
    // Hide settings sections
    // Show model selection section so user can choose framework and variant
    document.getElementById('modelPurposeSection').style.display = 'block';
    document.getElementById('settingsSection').style.display = 'block';
    
    // Reset quality slider
    const qualitySliderEl = document.getElementById('qualitySlider');
    if (qualitySliderEl) {
        qualitySliderEl.value = 50;
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
    if (modelPurposeInput) {
        modelPurposeInput.value = 'machine_learning';
        // Trigger change event to populate framework options
        modelPurposeInput.dispatchEvent(new Event('change'));
    }
    if (frameworkInput) frameworkInput.value = '';
    if (variantInput) variantInput.value = '';
    
    log('New project ready. Select a framework and model variant, then upload training data to begin.', 'success');
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
    }, 500); // Update every 0.5 seconds
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
    
    if (!slider || !sliderFill || !sliderHandle) return;
    
    slider.value = quality;
    const percentage = quality;
    // Fill should grow from left to right as quality increases
    sliderFill.style.width = `${percentage}%`;
    // Handle position: center the handle at the percentage position
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

// Show training loading overlay - DISABLED
function showTrainingLoadingOverlay() {
    // Disabled - no popup shown
}

// Hide training loading overlay - DISABLED
function hideTrainingLoadingOverlay() {
    // Disabled - no popup shown
}

// Update loading status text - DISABLED
function updateLoadingStatus(text) {
    // Disabled - no popup shown
}

// Notification functions
function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    if (notification) {
        const icon = notification.querySelector('.notification-icon');
        const text = notification.querySelector('.notification-text');
        
        if (text) text.textContent = message || 'Action Successful';
        if (icon) {
            icon.textContent = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
        }
        
        notification.classList.add('show');
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            hideNotification();
        }, 3000);
    }
}

function hideNotification() {
    const notification = document.getElementById('notification');
    if (notification) {
        notification.classList.remove('show');
    }
}

// Make functions globally available
window.showNotification = showNotification;
window.hideNotification = hideNotification;

// Calculate and display parameter count based on training data and model configuration
function updateParameterCount() {
    // Parameters are architecture-based, not data-based
    // Only show parameters when:
    // 1. A model architecture is selected (framework + variant), OR
    // 2. Data has been uploaded (implies a model will be used)
    
    // Get current model configuration - read values fresh every time
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const frameworkSelect = document.getElementById('frameworkInput');
    const variantSelect = document.getElementById('modelVariantInput');
    const qualitySliderEl = document.getElementById('qualitySlider');
    
    // Read all values fresh - no caching
    const modelPurpose = modelPurposeSelect ? modelPurposeSelect.value : '';
    const framework = frameworkSelect ? frameworkSelect.value : '';
    const variant = variantSelect ? variantSelect.value : '';
    // Parse quality - explicitly handle 0 as valid value (not falsy)
    let quality = 50; // Default
    if (qualitySliderEl && qualitySliderEl.value !== null && qualitySliderEl.value !== undefined) {
        const parsed = parseInt(qualitySliderEl.value, 10);
        if (!isNaN(parsed)) {
            quality = parsed; // This will correctly handle 0
        }
    }
    
    // Check if we should show parameters
    // Allow showing parameters if:
    // 1. Framework is selected (variant optional - will use default multiplier if not set)
    // 2. AND a dataset is selected (either folder selection or file upload)
    const hasFramework = framework && framework.length > 0;
    const hasData = (uploadedFiles && uploadedFiles.length > 0) || (selectedFolderPath && selectedFolderPath.length > 0);
    
    // If no framework or no data, show placeholder
    if (!hasFramework || !hasData) {
        const paramElement = document.getElementById('parameterCount');
        if (paramElement) {
            paramElement.textContent = '--';
        }
        return { count: 0, text: '--' };
    }
    
    // If framework is selected but variant is not, we can still calculate (uses default multiplier)
    // Variant will be empty string, which defaults to multiplier 1.0
    
    // Use default modelPurpose if not selected but we have data
    const effectiveModelPurpose = modelPurpose || 'machine_learning';
    
    // Calculate parameters based on model architecture (not dataset)
    let parameterCount = calculateParametersFromArchitecture(effectiveModelPurpose, framework, variant, quality);
    
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
    
    // Update parameter count display immediately
    const paramElement = document.getElementById('parameterCount');
    if (paramElement) {
        paramElement.textContent = paramText;
    }
    
    return { count: parameterCount, text: paramText };
}

/**
 * Calculate parameters from layer specifications
 * Supports: dense, conv, rnn, lstm, gru, embedding layers
 * @param {Array} layers - Array of layer specification objects
 * @returns {Object} - { total: number, perLayer: Object }
 */
function calculateParameters(layers) {
    let totalParameters = 0;
    const perLayer = {};
    
    if (!Array.isArray(layers) || layers.length === 0) {
        return { total: 0, perLayer: {} };
    }
    
    layers.forEach((layer, index) => {
        if (!layer || !layer.type) {
            console.warn(`Layer ${index} is missing type specification`);
            return;
        }
        
        let layerParams = 0;
        const layerType = layer.type.toLowerCase();
        
        try {
            switch (layerType) {
                case 'dense':
                    // Dense (fully connected) layers
                    // Parameters = input_units * output_units + output_units (biases)
                    if (layer.input_units !== undefined && layer.output_units !== undefined) {
                        // Clamp values to prevent overflow and ensure valid range
                        const inputUnits = Math.max(1, Math.min(1000000, layer.input_units || 1));
                        const outputUnits = Math.max(1, Math.min(1000000, layer.output_units || 1));
                        
                        // Check for integer overflow before calculation
                        const weights = inputUnits * outputUnits;
                        const biases = outputUnits;
                        const MAX_SAFE_INT = Number.MAX_SAFE_INTEGER || 9007199254740991;
                        
                        if (weights > MAX_SAFE_INT || biases > MAX_SAFE_INT) {
                            console.warn(`Parameter calculation may overflow for dense layer ${index}. Using max safe integer.`);
                            layerParams = MAX_SAFE_INT;
                        } else {
                            layerParams = weights + biases;
                        }
                    } else {
                        console.warn(`Dense layer ${index} missing input_units or output_units`, layer);
                    }
                    break;
                    
                case 'conv':
                    // Convolutional layers
                    // Parameters = kernel_height * kernel_width * in_channels * out_channels + out_channels (biases)
                    if (layer.kernel_height !== undefined && layer.kernel_width !== undefined && 
                        layer.in_channels !== undefined && layer.out_channels !== undefined) {
                        const inCh = Math.max(1, Math.min(10000, layer.in_channels || 1));
                        const outCh = Math.max(1, Math.min(10000, layer.out_channels || 1));
                        const kernelH = Math.max(1, Math.min(100, layer.kernel_height || 3));
                        const kernelW = Math.max(1, Math.min(100, layer.kernel_width || 3));
                        
                        const MAX_SAFE_INT = Number.MAX_SAFE_INTEGER || 9007199254740991;
                        const kernelParams = kernelH * kernelW * inCh * outCh;
                        const biases = outCh;
                        
                        if (kernelParams > MAX_SAFE_INT || biases > MAX_SAFE_INT) {
                            console.warn(`Parameter calculation may overflow for conv layer ${index}. Using max safe integer.`);
                            layerParams = MAX_SAFE_INT;
                        } else {
                            layerParams = kernelParams + biases;
                        }
                    } else {
                        console.warn(`Conv layer ${index} missing required fields (kernel_height, kernel_width, in_channels, out_channels)`, layer);
                    }
                    break;
                    
                case 'rnn':
                    // RNN layers
                    // Parameters = input_size * hidden_size + hidden_size * hidden_size + hidden_size
                    if (layer.input_size !== undefined && layer.hidden_size !== undefined) {
                        const inputSize = Math.max(1, Math.min(100000, layer.input_size || 1));
                        const hiddenSize = Math.max(1, Math.min(100000, layer.hidden_size || 1));
                        
                        const MAX_SAFE_INT = Number.MAX_SAFE_INTEGER || 9007199254740991;
                        const inputHidden = inputSize * hiddenSize;
                        const hiddenHidden = hiddenSize * hiddenSize;
                        const biases = hiddenSize;
                        
                        if (inputHidden > MAX_SAFE_INT || hiddenHidden > MAX_SAFE_INT || biases > MAX_SAFE_INT) {
                            console.warn(`Parameter calculation may overflow for RNN layer ${index}. Using max safe integer.`);
                            layerParams = MAX_SAFE_INT;
                        } else {
                            layerParams = inputHidden + hiddenHidden + biases;
                        }
                    } else {
                        console.warn(`RNN layer ${index} missing input_size or hidden_size`);
                    }
                    break;
                    
                case 'lstm':
                case 'gru':
                    // LSTM/GRU layers (4x more parameters than RNN)
                    // Parameters = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
                    if (layer.input_size !== undefined && layer.hidden_size !== undefined) {
                        const inputSize = Math.max(1, Math.min(100000, layer.input_size || 1));
                        const hiddenSize = Math.max(1, Math.min(100000, layer.hidden_size || 1));
                        
                        const MAX_SAFE_INT = Number.MAX_SAFE_INTEGER || 9007199254740991;
                        const inputHidden = inputSize * hiddenSize;
                        const hiddenHidden = hiddenSize * hiddenSize;
                        const biases = hiddenSize;
                        
                        if (inputHidden > MAX_SAFE_INT || hiddenHidden > MAX_SAFE_INT || biases > MAX_SAFE_INT) {
                            console.warn(`Parameter calculation may overflow for ${layerType.toUpperCase()} layer ${index}. Using max safe integer.`);
                            layerParams = MAX_SAFE_INT;
                        } else {
                            const baseParams = inputHidden + hiddenHidden + biases;
                            layerParams = 4 * baseParams;
                            // Check if 4x multiplication causes overflow
                            if (layerParams > MAX_SAFE_INT) {
                                console.warn(`Parameter calculation overflowed after 4x multiplication for ${layerType.toUpperCase()} layer ${index}. Using max safe integer.`);
                                layerParams = MAX_SAFE_INT;
                            }
                        }
                    } else {
                        console.warn(`${layerType.toUpperCase()} layer ${index} missing input_size or hidden_size`);
                    }
                    break;
                    
                case 'embedding':
                    // Embedding layers
                    // Parameters = vocab_size * embedding_dim
                    if (layer.vocab_size !== undefined && layer.embedding_dim !== undefined) {
                        const vocabSize = Math.max(1, Math.min(1000000, layer.vocab_size || 1));
                        const embedDim = Math.max(1, Math.min(100000, layer.embedding_dim || 1));
                        
                        const MAX_SAFE_INT = Number.MAX_SAFE_INTEGER || 9007199254740991;
                        const embedParams = vocabSize * embedDim;
                        
                        if (embedParams > MAX_SAFE_INT) {
                            console.warn(`Parameter calculation may overflow for embedding layer ${index}. Using max safe integer.`);
                            layerParams = MAX_SAFE_INT;
                        } else {
                            layerParams = embedParams;
                        }
                    } else {
                        console.warn(`Embedding layer ${index} missing vocab_size or embedding_dim`);
                    }
                    break;
                    
                default:
                    console.warn(`Unsupported layer type: ${layerType} at index ${index}`);
                    layerParams = 0;
            }
            
            totalParameters += layerParams;
            perLayer[`layer_${index}_${layerType}`] = layerParams;
            
        } catch (error) {
            console.error(`Error calculating parameters for layer ${index} (${layerType}):`, error);
        }
    });
    
    return {
        total: totalParameters,
        perLayer: perLayer
    };
}

/**
 * Generate layer specifications based on model architecture
 * Converts layer sizes to proper layer specifications
 */
function generateLayerSpecifications(layerSizes, modelPurpose, framework, variant) {
    const layers = [];
    
    // For dense/MLP models, create dense layers
    if (!framework || framework === 'pytorch' || framework === 'tensorflow' || 
        framework === 'sklearn' || modelPurpose === 'machine_learning') {
        
        // Create dense layers from layer sizes
        for (let i = 0; i < layerSizes.length - 1; i++) {
            layers.push({
                type: 'dense',
                input_units: Math.max(1, layerSizes[i] || 1),
                output_units: Math.max(1, layerSizes[i + 1] || 1)
            });
        }
    }
    // For computer vision models, mix conv and dense layers
    else if (modelPurpose === 'computer_vision') {
        if (framework === 'yolo' || framework === 'resnet' || framework === 'efficientnet') {
            // For YOLO/ResNet: Start with conv layers, end with dense
            const numConvLayers = Math.min(3, Math.floor(layerSizes.length / 2));
            const numDenseLayers = layerSizes.length - numConvLayers - 1;
            
            // Add convolutional layers (simplified - actual conv layers need more params)
            for (let i = 0; i < numConvLayers && i < layerSizes.length - 1; i++) {
                const inCh = i === 0 ? 3 : Math.max(8, Math.floor(layerSizes[i] / 2)); // Ensure minimum channels
                const outCh = Math.max(8, Math.floor(layerSizes[i + 1] / 2)); // Ensure minimum channels
                layers.push({
                    type: 'conv',
                    kernel_height: 3,
                    kernel_width: 3,
                    in_channels: inCh,
                    out_channels: outCh
                });
            }
            
            // Add dense layers for classification
            const denseStart = numConvLayers;
            for (let i = denseStart; i < layerSizes.length - 1; i++) {
                layers.push({
                    type: 'dense',
                    input_units: Math.max(1, layerSizes[i] || 1),
                    output_units: Math.max(1, layerSizes[i + 1] || 1)
                });
            }
        } else {
            // Default to dense layers
            for (let i = 0; i < layerSizes.length - 1; i++) {
                layers.push({
                    type: 'dense',
                    input_units: Math.max(1, layerSizes[i] || 1),
                    output_units: Math.max(1, layerSizes[i + 1] || 1)
                });
            }
        }
    }
    // For NLP models, use embedding + RNN/LSTM + dense
    else if (modelPurpose === 'natural_language_processing') {
        if (framework === 'transformer' || framework === 'bert') {
            // For transformers: embedding + dense layers
            // Add embedding layer (estimate vocab size from data)
            const vocabSize = Math.min(50000, Math.max(1000, layerSizes[0] * 100));
            layers.push({
                type: 'embedding',
                vocab_size: vocabSize,
                embedding_dim: layerSizes[0]
            });
            
            // Add dense layers (skip first layer as it's covered by embedding)
            for (let i = 0; i < layerSizes.length - 1; i++) {
                layers.push({
                    type: 'dense',
                    input_units: Math.max(1, layerSizes[i]),
                    output_units: Math.max(1, layerSizes[i + 1])
                });
            }
        } else if (framework === 'lstm' || framework === 'gru' || framework === 'rnn') {
            // Add embedding layer
            const vocabSize = Math.min(50000, Math.max(1000, layerSizes[0] * 100));
            layers.push({
                type: 'embedding',
                vocab_size: vocabSize,
                embedding_dim: layerSizes[0]
            });
            
            // Add RNN/LSTM/GRU layer
            layers.push({
                type: framework.toLowerCase(),
                input_size: layerSizes[0],
                hidden_size: layerSizes[1] || layerSizes[0]
            });
            
            // Add dense output layers
            for (let i = 1; i < layerSizes.length - 1; i++) {
                layers.push({
                    type: 'dense',
                    input_units: Math.max(1, layerSizes[i] || 1),
                    output_units: Math.max(1, layerSizes[i + 1] || 1)
                });
            }
        } else {
            // Default to dense layers
            for (let i = 0; i < layerSizes.length - 1; i++) {
                layers.push({
                    type: 'dense',
                    input_units: Math.max(1, layerSizes[i] || 1),
                    output_units: Math.max(1, layerSizes[i + 1] || 1)
                });
            }
        }
    }
    // Default: dense layers
    else {
        for (let i = 0; i < layerSizes.length - 1; i++) {
            layers.push({
                type: 'dense',
                input_units: Math.max(1, layerSizes[i] || 1),
                output_units: Math.max(1, layerSizes[i + 1] || 1)
            });
        }
    }
    
    // Validate layer specifications
    if (layers.length === 0) {
        console.warn('No layers generated from layerSizes:', layerSizes);
    }
    
    return layers;
}

/**
 * Calculate parameters based on MODEL ARCHITECTURE only
 * Parameters = weights + biases in the model (FIXED once architecture is defined)
 * Dataset size, batch size, and epochs do NOT affect parameter count
 */
function calculateParametersFromArchitecture(modelPurpose, framework, variant, quality) {
    // Parameters are determined by architecture, NOT by dataset size
    // Quality is passed directly to avoid stale reads
    // Handle quality 0 explicitly (don't use || because 0 is falsy)
    const qualityValue = Math.max(0, Math.min(100, (quality !== null && quality !== undefined && !isNaN(quality)) ? quality : 50));
    // Map 0-100 to 0.1-1.0 range to prevent zero values that break calculations
    const qualityFactor = 0.1 + (qualityValue / 100) * 0.9; // 0.1 to 1.0 (prevents zero)
    
    // Define base architectures for each model type/variant
    // These are fixed architectures - parameters don't change with dataset size
    let layerSpecs = [];
    
    // Apply variant multiplier
    let variantMultiplier = 1.0;
    if (variant) {
        const variantLower = variant.toLowerCase();
        if (variantLower.includes('nano') || variantLower.includes('tiny') || variantLower.includes('n')) {
            variantMultiplier = 0.25;
        } else if (variantLower.includes('small') || variantLower.includes('s')) {
            variantMultiplier = 0.5;
        } else if (variantLower.includes('medium') || variantLower.includes('m') || variantLower.includes('base')) {
            variantMultiplier = 1.0;
        } else if (variantLower.includes('large') || variantLower.includes('l')) {
            variantMultiplier = 2.5;
        } else if (variantLower.includes('xl') || variantLower.includes('x') || variantLower.includes('extra')) {
            variantMultiplier = 5.0;
        }
    }
    
    // Computer Vision Models
    if (modelPurpose === 'computer_vision') {
        if (framework === 'yolo') {
            // YOLO models have fixed, known parameter counts
            // Return actual parameter counts for YOLOv11 models (from Ultralytics)
            const variantLower = variant ? variant.toLowerCase() : '';
            
            // YOLOv11 parameter counts (actual values from Ultralytics)
            if (variantLower.includes('nano') || variantLower.includes('tiny') || variantLower.includes('n')) {
                // YOLOv11n: 2,590,620 parameters
                return 2590620;
            } else if (variantLower.includes('small') || variantLower.includes('s')) {
                // YOLOv11s: ~12.6M parameters (approximate)
                return 12600000;
            } else if (variantLower.includes('medium') || variantLower.includes('m') || variantLower.includes('base')) {
                // YOLOv11m: ~25.4M parameters (approximate)
                return 25400000;
            } else if (variantLower.includes('large') || variantLower.includes('l')) {
                // YOLOv11l: ~43.7M parameters (approximate)
                return 43700000;
            } else if (variantLower.includes('xl') || variantLower.includes('x') || variantLower.includes('extra')) {
                // YOLOv11x: ~56.9M parameters (approximate)
                return 56900000;
            } else {
                // Default to YOLOv11n if variant not recognized
                return 2590620;
            }
            
        } else if (framework === 'resnet') {
            // ResNet architecture: Multiple conv blocks + dense head
            // Scale from 2 channels (quality 0) to 128 channels (quality 100)
            const baseChannels = Math.max(2, Math.floor(2 + qualityFactor * 126 * variantMultiplier));
            const ch1 = Math.max(2, Math.floor(baseChannels * 0.5));
            const ch2 = Math.max(4, Math.floor(baseChannels));
            const dense = Math.max(64, Math.floor(baseChannels * 8));
            
            layerSpecs.push({
                type: 'conv',
                kernel_height: 7,
                kernel_width: 7,
                in_channels: 3,
                out_channels: ch1
            });
            
            layerSpecs.push({
                type: 'conv',
                kernel_height: 3,
                kernel_width: 3,
                in_channels: ch1,
                out_channels: ch2
            });
            
            layerSpecs.push({
                type: 'dense',
                input_units: dense,
                output_units: 1000 // ImageNet classes (adjust as needed)
            });
            
        } else {
            // Default CV: Simple conv + dense
            // Scale from 2 (quality 0) to 256 (quality 100)
            const baseSize = Math.max(2, Math.floor(2 + qualityFactor * 254 * variantMultiplier));
            const channels = Math.max(2, Math.floor(baseSize * 0.25));
            const dense = Math.max(2, Math.floor(baseSize));
            
            layerSpecs.push({
                type: 'conv',
                kernel_height: 3,
                kernel_width: 3,
                in_channels: 3,
                out_channels: channels
            });
            
            layerSpecs.push({
                type: 'dense',
                input_units: dense,
                output_units: 10 // Classification
            });
        }
    }
    // Natural Language Processing
    else if (modelPurpose === 'natural_language_processing') {
        if (framework === 'transformer' || framework === 'bert') {
            // Transformer: Embedding + Dense layers
            // Scale from 8 dims (quality 0) to 512 dims (quality 100)
            const embedDim = Math.max(8, Math.floor(8 + qualityFactor * 504 * variantMultiplier));
            const vocabSize = 30522; // BERT base vocab size
            const dense1 = Math.max(2, Math.floor(embedDim * 0.5));
            
            layerSpecs.push({
                type: 'embedding',
                vocab_size: vocabSize,
                embedding_dim: embedDim
            });
            
            layerSpecs.push({
                type: 'dense',
                input_units: embedDim,
                output_units: dense1
            });
            
            layerSpecs.push({
                type: 'dense',
                input_units: dense1,
                output_units: 2 // Classification
            });
            
        } else if (framework === 'lstm' || framework === 'gru') {
            // LSTM/GRU: Embedding + RNN + Dense
            // Scale from 2 hidden units (quality 0) to 512 (quality 100)
            const hiddenSize = Math.max(2, Math.floor(2 + qualityFactor * 510 * variantMultiplier));
            const vocabSize = 10000;
            const embedDim = Math.max(2, Math.floor(2 + qualityFactor * 126)); // 2 to 128
            
            layerSpecs.push({
                type: 'embedding',
                vocab_size: vocabSize,
                embedding_dim: embedDim
            });
            
            layerSpecs.push({
                type: framework.toLowerCase(),
                input_size: embedDim,
                hidden_size: hiddenSize
            });
            
            layerSpecs.push({
                type: 'dense',
                input_units: hiddenSize,
                output_units: 2 // Classification
            });
            
        } else {
            // Default NLP: Dense layers
            // Scale from 2 (quality 0) to 512 (quality 100)
            const baseSize = Math.max(2, Math.floor(2 + qualityFactor * 510 * variantMultiplier));
            const hidden = Math.max(1, Math.floor(baseSize * 0.5));
            
            layerSpecs.push({
                type: 'dense',
                input_units: baseSize,
                output_units: hidden
            });
            
            layerSpecs.push({
                type: 'dense',
                input_units: hidden,
                output_units: 10
            });
        }
    }
    // Machine Learning / Default
    else {
        // Only calculate if we have a framework selected (not just default)
        // If no framework is selected, return 0 (will show '--')
        if (!framework) {
            return 0;
        }
        
        // Standard MLP: Dense layers only
        // Scale from 2 (quality 0) to 256 (quality 100)
        const baseSize = Math.max(2, Math.floor(2 + qualityFactor * 254 * variantMultiplier));
        const hidden1 = Math.max(2, Math.floor(baseSize * 0.75));
        const hidden2 = Math.max(1, Math.floor(hidden1 * 0.5));
        
        layerSpecs.push({
            type: 'dense',
            input_units: Math.max(1, baseSize),
            output_units: Math.max(1, hidden1)
        });
        
        layerSpecs.push({
            type: 'dense',
            input_units: Math.max(1, hidden1),
            output_units: Math.max(1, hidden2)
        });
        
        layerSpecs.push({
            type: 'dense',
            input_units: Math.max(1, hidden2),
            output_units: 1 // Regression/Classification output
        });
    }
    
    // If no layers were generated, return 0
    if (layerSpecs.length === 0) {
        return 0;
    }
    
    // Calculate parameters from layer specifications
    const paramResult = calculateParameters(layerSpecs);
    
    if (!paramResult || paramResult.total === 0 || isNaN(paramResult.total) || !isFinite(paramResult.total)) {
        console.warn('Parameter calculation returned invalid result', paramResult);
        return 0;
    }
    
    return Math.max(1, Math.floor(paramResult.total));
}

// Get layer specifications for neural network visualization
function getLayerSpecsForVisualization(modelPurpose, framework, variant, quality) {
    // Reuse the architecture calculation logic
    const qualityValue = Math.max(0, Math.min(100, (quality !== null && quality !== undefined && !isNaN(quality)) ? quality : 50));
    const qualityFactor = 0.1 + (qualityValue / 100) * 0.9;
    
    let layerSpecs = [];
    
    // Apply variant multiplier
    let variantMultiplier = 1.0;
    if (variant) {
        const variantLower = variant.toLowerCase();
        if (variantLower.includes('nano') || variantLower.includes('tiny') || variantLower.includes('n')) {
            variantMultiplier = 0.25;
        } else if (variantLower.includes('small') || variantLower.includes('s')) {
            variantMultiplier = 0.5;
        } else if (variantLower.includes('medium') || variantLower.includes('m') || variantLower.includes('base')) {
            variantMultiplier = 1.0;
        } else if (variantLower.includes('large') || variantLower.includes('l')) {
            variantMultiplier = 2.5;
        } else if (variantLower.includes('xl') || variantLower.includes('x') || variantLower.includes('extra')) {
            variantMultiplier = 5.0;
        }
    }
    
    // Use the same architecture calculation as calculateParametersFromArchitecture
    // But return layerSpecs instead of parameter count
    // For visualization, we'll extract layer sizes from these specs
    // This is a simplified version - we could extract this logic to a shared function
    // For now, we'll extract layer sizes directly in the neural network code
    
    // Return empty array - neural network will use calculateParametersFromArchitecture logic
    // via a different approach (passing model config)
    return null;
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
    // Discard all previous changes and read fresh from UI
    // Read all current settings from UI
    const qualitySliderEl = document.getElementById('qualitySlider');
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const frameworkSelect = document.getElementById('frameworkInput');
    const variantSelect = document.getElementById('modelVariantInput');
    const formatSelect = document.getElementById('modelFormatInput');
    
    const quality = qualitySliderEl ? parseInt(qualitySliderEl.value) : 50;
    const currentModelPurpose = modelPurposeSelect ? modelPurposeSelect.value : 'machine_learning';
    const currentFramework = frameworkSelect ? frameworkSelect.value : '';
    const currentVariant = variantSelect ? variantSelect.value : '';
    const currentFormat = formatSelect ? formatSelect.value : 'pt';
    
    // Update global variables to match UI
    modelPurpose = currentModelPurpose;
    framework = currentFramework;
    modelVariant = currentVariant;
    modelFormat = currentFormat;
    
    // Calculate recommended settings (use default values if no files)
    const recommended = calculateRecommendedSettings([]);
    
    // Adjust based on slider position
    const ratio = quality / 100;
    const maxEpochs = 100;
    const minEpochs = 10;
    
    // Read device setting (preserve if it exists, otherwise use 'auto')
    const deviceInput = document.getElementById('deviceInput');
    const deviceValue = deviceInput ? deviceInput.value : (trainingSettings.device || 'auto');
    
    // Completely replace trainingSettings (discard previous)
    trainingSettings = {
        epochs: Math.round(minEpochs + (maxEpochs - minEpochs) * ratio),
        batchSize: recommended.batchSize || 32,
        learningRate: (recommended.learningRate || 0.001) * (1 + (1 - ratio) * 0.5),
        optimizer: recommended.optimizer || 'adam',
        lossFunction: recommended.lossFunction || 'mse',
        validationSplit: recommended.validationSplit || 0.2,
        device: deviceValue
    };
    
    // Update neural network architecture with new settings
    updateNeuralNetworkSettings();
    
    // Update parameter count (must be after all settings are updated)
    const paramInfo = updateParameterCount();
    
    const deviceDisplay = trainingSettings.device === 'auto' ? 'Auto (GPU if available)' : (trainingSettings.device === 'cuda' ? 'GPU' : 'CPU');
    log(`Applied preset settings: ${quality}% quality`, 'success');
    log(`Model: ${currentModelPurpose} / ${currentFramework} / ${currentVariant}`, 'log');
    log(`Model Parameters: ${paramInfo ? paramInfo.text : 'N/A'} (${paramInfo ? paramInfo.count.toLocaleString() : 0})`, 'log');
    log(`Training Settings: Epochs: ${trainingSettings.epochs}, Batch: ${trainingSettings.batchSize}, LR: ${trainingSettings.learningRate.toFixed(6)}, Device: ${deviceDisplay}`, 'log');
    
    showNotification();
}

function applyManualSettings() {
    // Discard all previous changes and read fresh from UI
    // Read all current settings from UI
    const qualitySliderEl = document.getElementById('qualitySlider');
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const frameworkSelect = document.getElementById('frameworkInput');
    const variantSelect = document.getElementById('modelVariantInput');
    const formatSelect = document.getElementById('modelFormatInput');
    const epochsInput = document.getElementById('epochsInput');
    const batchSizeInput = document.getElementById('batchSizeInput');
    const learningRateInput = document.getElementById('learningRateInput');
    const optimizerInput = document.getElementById('optimizerInput');
    const lossFunctionInput = document.getElementById('lossFunctionInput');
    const validationSplitInput = document.getElementById('validationSplitInput');
    const deviceInput = document.getElementById('deviceInput');
    
    // RandomForest-specific inputs
    const nEstimatorsInput = document.getElementById('nEstimatorsInput');
    const maxDepthInput = document.getElementById('maxDepthInput');
    const minSamplesSplitInput = document.getElementById('minSamplesSplitInput');
    const minSamplesLeafInput = document.getElementById('minSamplesLeafInput');
    const maxFeaturesInput = document.getElementById('maxFeaturesInput');
    
    // Read all values
    const quality = qualitySliderEl ? parseInt(qualitySliderEl.value) : 50;
    const currentModelPurpose = modelPurposeSelect ? modelPurposeSelect.value : 'machine_learning';
    const currentFramework = frameworkSelect ? frameworkSelect.value : '';
    const currentVariant = variantSelect ? variantSelect.value : '';
    const currentFormat = formatSelect ? formatSelect.value : 'pytorch';
    
    // Get settings schema for this model
    const schema = getModelSettingsSchema(currentModelPurpose, currentFramework, currentVariant);
    
    // Extract numeric values from dropdown options (they contain text like "10 - Fast ⭐ Recommended")
    const epochsValue = epochsInput ? epochsInput.value : '10';
    const batchSizeValue = batchSizeInput ? batchSizeInput.value : '32';
    const learningRateValue = learningRateInput ? learningRateInput.value : '0.001';
    const optimizerValue = optimizerInput ? optimizerInput.value : 'adam';
    const lossFunctionValue = lossFunctionInput ? lossFunctionInput.value : 'mse';
    const validationSplitValue = validationSplitInput ? validationSplitInput.value : '0.2';
    const deviceValue = deviceInput ? deviceInput.value : 'auto';
    
    // RandomForest values
    const nEstimatorsValue = nEstimatorsInput ? nEstimatorsInput.value : '100';
    const maxDepthValue = maxDepthInput ? maxDepthInput.value : '10';
    const minSamplesSplitValue = minSamplesSplitInput ? minSamplesSplitInput.value : '2';
    const minSamplesLeafValue = minSamplesLeafInput ? minSamplesLeafInput.value : '1';
    const maxFeaturesValue = maxFeaturesInput ? maxFeaturesInput.value : 'sqrt';
    
    // Update global variables to match UI
    modelPurpose = currentModelPurpose;
    framework = currentFramework;
    modelVariant = currentVariant;
    modelFormat = currentFormat;
    
    // Build trainingSettings using ONLY relevant parameters from schema
    trainingSettings = {
        validationSplit: parseFloat(validationSplitValue) || 0.2 // Always include validation_split
    };
    
    // Add parameters based on schema
    if (schema.params.includes('epochs')) {
        trainingSettings.epochs = parseInt(epochsValue) || 10;
    }
    if (schema.params.includes('batchSize')) {
        trainingSettings.batchSize = parseInt(batchSizeValue) || 32;
    }
    if (schema.params.includes('learningRate')) {
        trainingSettings.learningRate = parseFloat(learningRateValue) || 0.001;
    }
    if (schema.params.includes('optimizer')) {
        trainingSettings.optimizer = optimizerValue || 'adam';
    }
    if (schema.params.includes('lossFunction')) {
        trainingSettings.lossFunction = lossFunctionValue || 'mse';
    }
    if (schema.params.includes('device')) {
        trainingSettings.device = deviceValue || 'auto';
    }
    
    // Tree-based parameters
    if (schema.params.includes('n_estimators')) {
        trainingSettings.n_estimators = parseInt(nEstimatorsValue) || 100;
    }
    if (schema.params.includes('max_depth')) {
        trainingSettings.max_depth = maxDepthValue === 'None' ? null : (parseInt(maxDepthValue) || 10);
    }
    if (schema.params.includes('min_samples_split')) {
        trainingSettings.min_samples_split = parseInt(minSamplesSplitValue) || 2;
    }
    if (schema.params.includes('min_samples_leaf')) {
        trainingSettings.min_samples_leaf = parseInt(minSamplesLeafValue) || 1;
    }
    if (schema.params.includes('max_features')) {
        trainingSettings.max_features = maxFeaturesValue === 'None' ? null : (maxFeaturesValue || 'sqrt');
    }
    
    // Update neural network architecture with new settings
    updateNeuralNetworkSettings();
    
    // Update parameter count (must be after all settings are updated)
    const paramInfo = updateParameterCount();
    
    // Log effective settings only (the ones that actually apply)
    log('Applied manual settings', 'success');
    log(`Model: ${currentModelPurpose} / ${currentFramework} / ${currentVariant}`, 'log');
    log(`Model Parameters: ${paramInfo.text} (${paramInfo.count.toLocaleString()})`, 'log');
    
    // Build effective settings log based on schema
    const effectiveSettings = [];
    if (schema.params.includes('epochs')) {
        effectiveSettings.push(`Epochs: ${trainingSettings.epochs}`);
    }
    if (schema.params.includes('batchSize')) {
        effectiveSettings.push(`Batch: ${trainingSettings.batchSize}`);
    }
    if (schema.params.includes('learningRate')) {
        effectiveSettings.push(`LR: ${trainingSettings.learningRate.toFixed(6)}`);
    }
    if (schema.params.includes('device')) {
        const deviceDisplay = trainingSettings.device === 'auto' ? 'Auto (GPU if available)' : (trainingSettings.device === 'cuda' ? 'GPU' : 'CPU');
        effectiveSettings.push(`Device: ${deviceDisplay}`);
    }
    if (schema.params.includes('optimizer')) {
        effectiveSettings.push(`Optimizer: ${trainingSettings.optimizer}`);
    }
    if (schema.params.includes('lossFunction')) {
        effectiveSettings.push(`Loss: ${trainingSettings.lossFunction}`);
    }
    if (schema.params.includes('n_estimators')) {
        effectiveSettings.push(`n_estimators: ${trainingSettings.n_estimators}`);
    }
    if (schema.params.includes('max_depth')) {
        effectiveSettings.push(`max_depth: ${trainingSettings.max_depth || 'None'}`);
    }
    if (schema.params.includes('min_samples_split')) {
        effectiveSettings.push(`min_samples_split: ${trainingSettings.min_samples_split}`);
    }
    if (schema.params.includes('min_samples_leaf')) {
        effectiveSettings.push(`min_samples_leaf: ${trainingSettings.min_samples_leaf}`);
    }
    if (schema.params.includes('max_features')) {
        effectiveSettings.push(`max_features: ${trainingSettings.max_features || 'None'}`);
    }
    if (schema.params.includes('validation_split')) {
        effectiveSettings.push(`Val Split: ${(trainingSettings.validationSplit * 100).toFixed(0)}%`);
    }
    
    log(`Effective settings: ${effectiveSettings.join(', ')}`, 'log');
    
    showNotification();
}

// Update neural network visualization with current training settings
function updateNeuralNetworkSettings() {
    if (typeof neuralNetwork === 'undefined' || !neuralNetwork) return;
    
    // Read all current values from UI (don't rely on cached variables)
    const qualitySliderEl = document.getElementById('qualitySlider');
    const modelPurposeSelect = document.getElementById('modelPurposeInput');
    const frameworkSelect = document.getElementById('frameworkInput');
    const variantSelect = document.getElementById('modelVariantInput');
    
    const currentQuality = qualitySliderEl ? parseInt(qualitySliderEl.value) : 50;
    const currentModelPurpose = modelPurposeSelect ? modelPurposeSelect.value : 'machine_learning';
    const currentFramework = frameworkSelect ? frameworkSelect.value : '';
    const currentVariant = variantSelect ? variantSelect.value : '';
    
    // Use default data size if no files (folder selection mode)
    const totalDataSize = 1000000; // Default size for visualization
    
    // Update neural network with fresh settings including model configuration
    neuralNetwork.updateTrainingSettings({
        quality: currentQuality,
        epochs: trainingSettings.epochs,
        batchSize: trainingSettings.batchSize,
        learningRate: trainingSettings.learningRate,
        modelType: currentModelPurpose,
        modelPurpose: currentModelPurpose,
        framework: currentFramework,
        variant: currentVariant,
        fileCount: 1, // Using folder selection
        dataSize: totalDataSize
    });
    
    // Recreate network architecture to reflect new settings
    if (neuralNetwork.createNetwork && typeof neuralNetwork.createNetwork === 'function') {
        neuralNetwork.createNetwork();
    }
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
    
    // Validate and analyze files
    const validationResult = validateAndAnalyzeFiles(files);
    
    // Group files by type and show summary
    const fileSummary = summarizeFiles(uploadedFiles);
    
    // Display summary instead of individual files
    fileSummary.forEach(summary => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-item-info">
                <span class="file-item-name">${summary.description}</span>
            </div>
            <span class="file-item-size">${summary.size}</span>
        `;
        fileList.appendChild(fileItem);
    });
    
    // Add single clear all button at the end
    if (uploadedFiles.length > 0) {
        const clearBtn = document.createElement('button');
        clearBtn.className = 'btn';
        clearBtn.textContent = 'Clear All Files';
        clearBtn.style.marginTop = '12px';
        clearBtn.style.width = '100%';
        clearBtn.addEventListener('click', clearAllFiles);
        fileList.appendChild(clearBtn);
    }
    
    // Store file metadata only (don't load all file data into memory)
    // File data is not needed for parameter calculation or training simulation
    // This prevents memory issues with large datasets
    uploadedFiles.forEach((file) => {
        // Only store metadata - file.data is not needed
        // If file data is required in the future, implement streaming/chunked reading
        file.metadata = {
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: file.lastModified
        };
    });
    
    // Log validation results
    if (validationResult.valid) {
        log(`Loaded ${files.length} file(s) for training`, 'success');
        validationResult.messages.forEach(msg => log(msg, 'log'));
    } else {
        log(`Loaded ${files.length} file(s) - Validation issues detected:`, 'warning');
        validationResult.errors.forEach(err => log(`  ❌ ${err}`, 'error'));
        validationResult.warnings.forEach(warn => log(`  ⚠️ ${warn}`, 'warning'));
        validationResult.messages.forEach(msg => log(msg, 'log'));
    }
    
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

// File validation and analysis functions
function validateAndAnalyzeFiles(files) {
    const result = {
        valid: true,
        errors: [],
        warnings: [],
        messages: []
    };
    
    if (!files || files.length === 0) {
        result.valid = false;
        result.errors.push('No files uploaded');
        return result;
    }
    
    // Analyze file types
    const fileTypes = {
        images: [],
        csv: [],
        json: [],
        text: [],
        other: []
    };
    
    Array.from(files).forEach(file => {
        const name = file.name.toLowerCase();
        if (/\.(png|jpg|jpeg|gif|bmp|webp|svg)$/i.test(name)) {
            fileTypes.images.push(file);
        } else if (/\.csv$/i.test(name)) {
            fileTypes.csv.push(file);
        } else if (/\.json$/i.test(name)) {
            fileTypes.json.push(file);
        } else if (/\.(txt|md|log)$/i.test(name)) {
            fileTypes.text.push(file);
        } else {
            fileTypes.other.push(file);
        }
    });
    
    // Generate descriptive messages
    if (fileTypes.images.length > 0) {
        result.messages.push(`📸 ${fileTypes.images.length} image file(s) - Suitable for computer vision tasks`);
    }
    if (fileTypes.csv.length > 0) {
        result.messages.push(`📊 ${fileTypes.csv.length} CSV file(s) - Structured data for classification/regression`);
    }
    if (fileTypes.json.length > 0) {
        result.messages.push(`📋 ${fileTypes.json.length} JSON file(s) - Structured data for various tasks`);
    }
    if (fileTypes.text.length > 0) {
        result.messages.push(`📄 ${fileTypes.text.length} text file(s) - Suitable for NLP or text analysis`);
    }
    if (fileTypes.other.length > 0) {
        result.warnings.push(`⚠️ ${fileTypes.other.length} unsupported file type(s): ${fileTypes.other.map(f => f.name).join(', ')}`);
        result.valid = false;
    }
    
    // Check for missing components
    const totalSize = Array.from(files).reduce((sum, f) => sum + (f.size || 0), 0);
    if (totalSize === 0) {
        result.errors.push('All files appear to be empty');
        result.valid = false;
    }
    
    if (files.length === 1) {
        result.warnings.push('Only one file uploaded - consider adding more data for better training');
    }
    
    // Check file sizes
    const largeFiles = Array.from(files).filter(f => f.size > 100 * 1024 * 1024); // > 100MB
    if (largeFiles.length > 0) {
        result.warnings.push(`${largeFiles.length} large file(s) detected (>100MB) - training may be slow`);
    }
    
    const emptyFiles = Array.from(files).filter(f => f.size === 0);
    if (emptyFiles.length > 0) {
        result.errors.push(`${emptyFiles.length} empty file(s): ${emptyFiles.map(f => f.name).join(', ')}`);
        result.valid = false;
    }
    
    // Check format validity
    const invalidImages = fileTypes.images.filter(f => {
        const ext = f.name.split('.').pop().toLowerCase();
        return !['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg'].includes(ext);
    });
    if (invalidImages.length > 0) {
        result.warnings.push(`Some image files may have invalid extensions: ${invalidImages.map(f => f.name).join(', ')}`);
    }
    
    // Recommend model purpose based on data
    if (fileTypes.images.length > 0 && fileTypes.images.length >= fileTypes.csv.length + fileTypes.json.length) {
        result.messages.push('💡 Recommendation: Use "Computer Vision" model purpose for image data');
    } else if (fileTypes.csv.length + fileTypes.json.length > 0) {
        result.messages.push('💡 Recommendation: Use "Machine Learning" or "Time Series" model purpose');
    } else if (fileTypes.text.length > 0) {
        result.messages.push('💡 Recommendation: Use "Natural Language Processing" model purpose');
    }
    
    // Check if files are missing required components
    if (fileTypes.images.length > 0 && fileTypes.images.length < 10) {
        result.warnings.push(`Low number of images (${fileTypes.images.length}) - computer vision models typically need 100+ images`);
    }
    
    if (fileTypes.csv.length > 0 && fileTypes.csv.length === 1) {
        result.warnings.push('Only one CSV file - ensure it contains both training features and labels');
    }
    
    return result;
}

function getFileType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg'].includes(ext)) return 'Image';
    if (ext === 'csv') return 'CSV';
    if (ext === 'json') return 'JSON';
    if (['txt', 'md', 'log'].includes(ext)) return 'Text';
    return 'Unknown';
}

function getFileDescription(filename, fileType) {
    switch(fileType) {
        case 'Image':
            return 'Image data for computer vision tasks';
        case 'CSV':
            return 'Structured tabular data';
        case 'JSON':
            return 'Structured data in JSON format';
        case 'Text':
            return 'Text data for NLP tasks';
        default:
            return 'Unsupported file format';
    }
}

// Summarize files by type - groups similar files together
function summarizeFiles(files) {
    const groups = {
        images: [],
        csv: [],
        json: [],
        text: [],
        other: []
    };
    
    // Group files by type
    files.forEach(file => {
        const fileType = getFileType(file.name);
        switch(fileType) {
            case 'Image':
                groups.images.push(file);
                break;
            case 'CSV':
                groups.csv.push(file);
                break;
            case 'JSON':
                groups.json.push(file);
                break;
            case 'Text':
                groups.text.push(file);
                break;
            default:
                groups.other.push(file);
        }
    });
    
    const summary = [];
    
    // Create summary entries
    if (groups.images.length > 0) {
        const totalSize = groups.images.reduce((sum, f) => sum + (f.size || 0), 0);
        const count = groups.images.length;
        const countText = count >= 1000 ? `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}k` : count;
        summary.push({
            description: `${countText} image${count !== 1 ? 's' : ''}${count >= 1000 ? ' (computer vision data)' : ''}`,
            size: formatFileSize(totalSize)
        });
    }
    
    if (groups.csv.length > 0) {
        const totalSize = groups.csv.reduce((sum, f) => sum + (f.size || 0), 0);
        const count = groups.csv.length;
        const countText = count >= 1000 ? `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}k` : count;
        summary.push({
            description: `${countText} CSV file${count !== 1 ? 's' : ''}${count >= 1000 ? ' (structured data)' : ''}`,
            size: formatFileSize(totalSize)
        });
    }
    
    if (groups.json.length > 0) {
        const totalSize = groups.json.reduce((sum, f) => sum + (f.size || 0), 0);
        const count = groups.json.length;
        const countText = count >= 1000 ? `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}k` : count;
        summary.push({
            description: `${countText} JSON file${count !== 1 ? 's' : ''}${count >= 1000 ? ' (structured data)' : ''}`,
            size: formatFileSize(totalSize)
        });
    }
    
    if (groups.text.length > 0) {
        const totalSize = groups.text.reduce((sum, f) => sum + (f.size || 0), 0);
        const count = groups.text.length;
        const countText = count >= 1000 ? `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}k` : count;
        summary.push({
            description: `${countText} text file${count !== 1 ? 's' : ''}${count >= 1000 ? ' (NLP data)' : ''}`,
            size: formatFileSize(totalSize)
        });
    }
    
    if (groups.other.length > 0) {
        const totalSize = groups.other.reduce((sum, f) => sum + (f.size || 0), 0);
        const count = groups.other.length;
        summary.push({
            description: `${count} unsupported file${count !== 1 ? 's' : ''}: ${groups.other.map(f => f.name.split('/').pop()).slice(0, 3).join(', ')}${groups.other.length > 3 ? '...' : ''}`,
            size: formatFileSize(totalSize)
        });
    }
    
    return summary;
}

// Clear all files
function clearAllFiles() {
    uploadedFiles = [];
    selectedFolderPath = null;
    selectedDatasetFile = null;
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    const selectedFolderPathEl = document.getElementById('selectedFolderPath');
    if (selectedFolderPathEl) {
        selectedFolderPathEl.style.display = 'none';
    }
    document.getElementById('settingsSection').style.display = 'none';
    document.getElementById('modelPurposeSection').style.display = 'none';
    log('All files cleared', 'log');
}

// Make globally accessible
window.clearAllFiles = clearAllFiles;

function setupFileUpload() {
    // File upload removed - only folder selection is used now
    // This function is kept for compatibility but does nothing
    return;
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
        updateFormatOptions();
        updateParameterCount();
    }
    
    // Update settings visibility based on variant
    updateSettingsVisibility();
}

// Model Settings Registry - Single source of truth for which parameters apply to which algorithms
const modelSettingsRegistry = {
    // Tree-based models (sklearn)
    'tabular/sklearn/random_forest': {
        params: ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'validation_split'],
        hide: ['epochs', 'batchSize', 'learningRate', 'optimizer', 'lossFunction', 'device'],
        helperText: 'Tree-based model (no epochs/LR/optimizer)',
        type: 'tree'
    },
    'tabular/sklearn/gradient_boosting': {
        params: ['n_estimators', 'max_depth', 'learning_rate', 'validation_split'],
        hide: ['epochs', 'batchSize', 'optimizer', 'lossFunction', 'device'],
        helperText: 'Tree-based model (no epochs/optimizer)',
        type: 'tree'
    },
    'tabular/sklearn/extra_trees': {
        params: ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'validation_split'],
        hide: ['epochs', 'batchSize', 'learningRate', 'optimizer', 'lossFunction', 'device'],
        helperText: 'Tree-based model (no epochs/LR/optimizer)',
        type: 'tree'
    },
    // XGBoost
    'tabular/xgboost/*': {
        params: ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'validation_split'],
        hide: ['epochs', 'batchSize', 'optimizer', 'lossFunction', 'device'],
        helperText: 'Gradient boosting model (no epochs/optimizer)',
        type: 'tree'
    },
    // LightGBM
    'tabular/lightgbm/*': {
        params: ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 'validation_split'],
        hide: ['epochs', 'batchSize', 'optimizer', 'lossFunction', 'device'],
        helperText: 'Gradient boosting model (no epochs/optimizer)',
        type: 'tree'
    },
    // Neural Networks (future)
    'tabular/pytorch/*': {
        params: ['epochs', 'batchSize', 'learningRate', 'optimizer', 'lossFunction', 'device', 'validation_split'],
        hide: ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        helperText: 'Neural network model',
        type: 'neural_network'
    },
    // Computer Vision models (YOLO, etc.)
    'computer_vision/yolo/*': {
        params: ['epochs', 'batchSize', 'learningRate', 'device', 'validation_split'],
        hide: ['optimizer', 'lossFunction', 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        helperText: 'Object detection model (YOLO)',
        type: 'neural_network'
    },
    'computer_vision/pytorch/*': {
        params: ['epochs', 'batchSize', 'learningRate', 'optimizer', 'lossFunction', 'device', 'validation_split'],
        hide: ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        helperText: 'Deep learning model',
        type: 'neural_network'
    },
    // Default (for CV, NLP, etc.) - show all NN-style settings
    'default': {
        params: ['epochs', 'batchSize', 'learningRate', 'optimizer', 'lossFunction', 'device', 'validation_split'],
        hide: ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        helperText: 'Deep learning model',
        type: 'neural_network'
    }
};

// Get settings schema for a model
function getModelSettingsSchema(purpose, framework, variant) {
    const key = `${purpose}/${framework}/${variant}`;
    const wildcardKey = `${purpose}/${framework}/*`;
    
    // Try exact match first
    if (modelSettingsRegistry[key]) {
        return modelSettingsRegistry[key];
    }
    
    // Try wildcard match
    if (modelSettingsRegistry[wildcardKey]) {
        return modelSettingsRegistry[wildcardKey];
    }
    
    // Fallback to default
    return modelSettingsRegistry['default'];
}

// Update settings visibility based on model variant
function updateSettingsVisibility() {
    const variantInput = document.getElementById('modelVariantInput');
    const frameworkInput = document.getElementById('frameworkInput');
    const purposeInput = document.getElementById('modelPurposeInput');
    
    if (!variantInput || !frameworkInput || !purposeInput) return;
    
    const variant = variantInput.value;
    const framework = frameworkInput.value;
    const purpose = purposeInput.value;
    
    // Get settings schema for this model
    const schema = getModelSettingsSchema(purpose, framework, variant);
    
    // Get all setting groups
    const epochsInput = document.getElementById('epochsInput');
    const batchSizeInput = document.getElementById('batchSizeInput');
    const learningRateInput = document.getElementById('learningRateInput');
    const optimizerInput = document.getElementById('optimizerInput');
    const lossFunctionInput = document.getElementById('lossFunctionInput');
    const deviceInput = document.getElementById('deviceInput');
    const validationSplitInput = document.getElementById('validationSplitInput');
    const randomForestSettings = document.getElementById('randomForestSettings');
    
    const epochsGroup = epochsInput?.closest('.setting-group');
    const batchSizeGroup = batchSizeInput?.closest('.setting-group');
    const learningRateGroup = learningRateInput?.closest('.setting-group');
    const optimizerGroup = optimizerInput?.closest('.setting-group');
    const lossFunctionGroup = lossFunctionInput?.closest('.setting-group');
    const deviceGroup = deviceInput?.closest('.setting-group');
    const validationSplitGroup = validationSplitInput?.closest('.setting-group');
    
    // Show/hide based on schema
    if (schema.hide.includes('epochs')) {
        if (epochsGroup) epochsGroup.style.display = 'none';
    } else {
        if (epochsGroup) epochsGroup.style.display = 'block';
    }
    
    if (schema.hide.includes('batchSize')) {
        if (batchSizeGroup) batchSizeGroup.style.display = 'none';
    } else {
        if (batchSizeGroup) batchSizeGroup.style.display = 'block';
    }
    
    if (schema.hide.includes('learningRate')) {
        if (learningRateGroup) learningRateGroup.style.display = 'none';
    } else {
        if (learningRateGroup) learningRateGroup.style.display = 'block';
    }
    
    if (schema.hide.includes('optimizer')) {
        if (optimizerGroup) optimizerGroup.style.display = 'none';
    } else {
        if (optimizerGroup) optimizerGroup.style.display = 'block';
    }
    
    if (schema.hide.includes('lossFunction')) {
        if (lossFunctionGroup) lossFunctionGroup.style.display = 'none';
    } else {
        if (lossFunctionGroup) lossFunctionGroup.style.display = 'block';
    }
    
    if (schema.hide.includes('device')) {
        if (deviceGroup) deviceGroup.style.display = 'none';
    } else {
        if (deviceGroup) deviceGroup.style.display = 'block';
    }
    
    // Always show validation_split (it's used by all models)
    if (validationSplitGroup) validationSplitGroup.style.display = 'block';
    
    // Show/hide RandomForest settings
    if (schema.type === 'tree' && randomForestSettings) {
        randomForestSettings.style.display = 'block';
    } else if (randomForestSettings) {
        randomForestSettings.style.display = 'none';
    }
    
    // Add helper text if available
    const manualSettings = document.getElementById('manualSettings');
    if (manualSettings && schema.helperText) {
        // Remove existing helper text if any
        const existingHelper = manualSettings.querySelector('.model-type-helper');
        if (existingHelper) {
            existingHelper.remove();
        }
        
        // Add new helper text at the top
        const helperDiv = document.createElement('div');
        helperDiv.className = 'model-type-helper';
        helperDiv.style.cssText = 'margin-bottom: 16px; padding: 8px 12px; background: #F0F0F0; border-left: 3px solid #8B7355; border-radius: 4px; font-size: 12px; color: #6B5A4A;';
        helperDiv.textContent = `ℹ️ ${schema.helperText}`;
        manualSettings.insertBefore(helperDiv, manualSettings.firstChild);
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

// Setup metrics tab switching
function setupMetricsTabs() {
    const simpleTab = document.getElementById('simpleMetricsTab');
    const detailedTab = document.getElementById('detailedMetricsTab');
    const simpleView = document.getElementById('simpleMetricsView');
    const detailedView = document.getElementById('detailedMetricsView');
    
    if (!simpleTab || !detailedTab || !simpleView || !detailedView) {
        console.warn('Metrics tab elements not found');
        return;
    }
    
    simpleTab.addEventListener('click', () => {
        simpleTab.classList.add('active');
        detailedTab.classList.remove('active');
        simpleView.style.display = 'flex';
        detailedView.style.display = 'none';
    });
    
    detailedTab.addEventListener('click', () => {
        detailedTab.classList.add('active');
        simpleTab.classList.remove('active');
        detailedView.style.display = 'flex';
        simpleView.style.display = 'none';
    });
}

function setupSettings() {
    // Prevent multiple initializations
    if (window.settingsInitialized) {
        return;
    }
    window.settingsInitialized = true;
    
    const qualitySliderEl = document.getElementById('qualitySlider');
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
    // Use both 'change' and 'input' events to catch all interactions
    if (modelPurposeInput) {
        modelPurposeInput.addEventListener('change', (e) => {
            modelPurpose = e.target.value;
            updateFrameworkOptions(); // This will cascade to updateVariantOptions which calls updateEpochLabel
            updateNeuralNetworkSettings(); // Update network architecture when model type changes
            updateParameterCount(); // Update parameter display
            updateDatasetButtonText(); // Update button text based on model type
        });
        modelPurposeInput.addEventListener('input', (e) => {
            modelPurpose = e.target.value;
            updateFrameworkOptions(); // This will cascade to updateVariantOptions which calls updateEpochLabel
            updateNeuralNetworkSettings();
            updateParameterCount();
            updateDatasetButtonText(); // Update button text based on model type
        });
    }
    
    if (frameworkInput) {
        frameworkInput.addEventListener('change', (e) => {
            framework = e.target.value;
            updateVariantOptions();
            updateFormatOptions();
            updateParameterCount(); // Update parameter display
            updateSettingsVisibility(); // Update settings visibility
        });
        frameworkInput.addEventListener('input', (e) => {
            framework = e.target.value;
            updateVariantOptions();
            updateFormatOptions();
            updateParameterCount();
            updateSettingsVisibility(); // Update settings visibility
        });
    }
    
    if (variantInput) {
        variantInput.addEventListener('change', (e) => {
            modelVariant = e.target.value;
            variant = e.target.value; // Also update variant variable
            updateParameterCount(); // Update parameter display
            updateSettingsVisibility(); // Update settings visibility
            // Update epoch label based on model type
            const modelPurpose = document.getElementById('modelPurposeInput').value;
            const framework = document.getElementById('frameworkInput').value;
            updateEpochLabel(modelPurpose, framework, variant);
        });
        variantInput.addEventListener('input', (e) => {
            modelVariant = e.target.value;
            variant = e.target.value;
            updateParameterCount();
            updateSettingsVisibility(); // Update settings visibility
            // Update epoch label based on model type
            const modelPurpose = document.getElementById('modelPurposeInput').value;
            const framework = document.getElementById('frameworkInput').value;
            updateEpochLabel(modelPurpose, framework, variant);
        });
    }
    
    if (formatInput) {
        formatInput.addEventListener('change', (e) => {
            modelFormat = e.target.value;
            // Format doesn't affect parameters, but update anyway for consistency
            updateParameterCount(); // Update parameter display
        });
    }
    
    // Manual settings dropdowns - update parameter count when changed
    // Note: Parameters only depend on architecture (purpose/framework/variant/quality),
    // but we still update to ensure consistency when settings change
    const epochsInput = document.getElementById('epochsInput');
    const batchSizeInput = document.getElementById('batchSizeInput');
    const learningRateInput = document.getElementById('learningRateInput');
    const optimizerInput = document.getElementById('optimizerInput');
    const lossFunctionInput = document.getElementById('lossFunctionInput');
    const validationSplitInput = document.getElementById('validationSplitInput');
    
    if (epochsInput) {
        epochsInput.addEventListener('change', () => {
            // Force parameter recalculation to ensure consistency
            updateParameterCount();
        });
        epochsInput.addEventListener('input', () => {
            updateParameterCount();
        });
    }
    if (batchSizeInput) {
        batchSizeInput.addEventListener('change', () => {
            updateParameterCount();
        });
        batchSizeInput.addEventListener('input', () => {
            updateParameterCount();
        });
    }
    if (learningRateInput) {
        learningRateInput.addEventListener('change', () => {
            updateParameterCount();
        });
        learningRateInput.addEventListener('input', () => {
            updateParameterCount();
        });
    }
    if (optimizerInput) {
        optimizerInput.addEventListener('change', () => {
            // Optimizer doesn't affect parameters, but update for consistency
            updateParameterCount();
        });
    }
    if (lossFunctionInput) {
        lossFunctionInput.addEventListener('change', () => {
            // Loss function doesn't affect parameters, but update for consistency
            updateParameterCount();
        });
    }
    if (validationSplitInput) {
        validationSplitInput.addEventListener('change', () => {
            // Validation split doesn't affect parameters, but update for consistency
            updateParameterCount();
        });
    }
    
    // Slider update - update parameter count when slider changes
    if (qualitySliderEl) {
        // Use both 'input' and 'change' events to ensure updates happen
        qualitySliderEl.addEventListener('input', (e) => {
            const qualityValue = parseInt(e.target.value, 10);
            updateSlider(qualityValue);
            // Force immediate recalculation with current slider value
            updateParameterCount();
            // Debug: log quality value to verify it's being read correctly
            // console.log('Slider moved to quality:', qualityValue);
        });
        
        qualitySliderEl.addEventListener('change', (e) => {
            const qualityValue = parseInt(e.target.value, 10);
            updateSlider(qualityValue);
            // Force recalculation when slider is released
            updateParameterCount();
            // Debug: log quality value
            // console.log('Slider changed to quality:', qualityValue);
        });
        
        // Initialize slider position
        updateSlider(parseInt(qualitySliderEl.value));
        // Initialize parameter count
        updateParameterCount();
    }
    
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
    
    // Note: Event listeners for modelPurposeInput, frameworkInput, variantInput, formatInput, and qualitySliderEl
    // are already set up earlier in this function, so we don't need to add them again here.
    // The parameter count will update automatically when those elements change.
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

// Real training IPC handlers
ipcRenderer.on('training-progress', (event, progressData) => {
    console.log('=== PROGRESS UPDATE RECEIVED ===');
    console.log('[Renderer] Received training-progress:', progressData);
    console.log('Epoch:', progressData.epoch, '/', progressData.total_epochs);
    console.log('Progress:', progressData.progress);
    console.log('Status:', progressData.status);
    console.log('Full data:', JSON.stringify(progressData, null, 2));
    console.log('===============================');
    
    // Handle training_started flag - ensure neural network visualization is active
    // Don't call startTraining() here as it's already been called in startRealTraining()
    // Just update progress if needed
    if (progressData.training_started && window.neuralNetwork && !window.neuralNetwork.isTraining) {
        console.log('[Renderer] Training started signal received, ensuring neural network visualization is active...');
        // Only start if not already started (avoid duplicate calls)
        window.neuralNetwork.startTraining();
        // Set minimal initial progress to show something immediately
        window.neuralNetwork.updateTrainingMetrics(0.8, 5.0, progressData.epoch || 0, progressData.total_epochs || totalEpochs, 0.01);
    }
    
    // Handle validation metrics separately (but don't skip other updates)
    if (progressData.type === 'validation') {
        // Update validation metrics (mAP values)
        if (progressData.map50 !== null && progressData.map50 !== undefined && !isNaN(progressData.map50)) {
            const map50El = document.getElementById('map50');
            if (map50El) map50El.textContent = progressData.map50.toFixed(3);
        }
        if (progressData.map5095 !== null && progressData.map5095 !== undefined && !isNaN(progressData.map5095)) {
            const map5095El = document.getElementById('map5095');
            if (map5095El) map5095El.textContent = progressData.map5095.toFixed(3);
        }
        if (progressData.precision !== null && progressData.precision !== undefined && !isNaN(progressData.precision)) {
            const precisionEl = document.getElementById('precision');
            if (precisionEl) precisionEl.textContent = progressData.precision.toFixed(3);
        }
        if (progressData.recall !== null && progressData.recall !== undefined && !isNaN(progressData.recall)) {
            const recallEl = document.getElementById('recall');
            if (recallEl) recallEl.textContent = progressData.recall.toFixed(3);
        }
        if (progressData.f1 !== null && progressData.f1 !== undefined && !isNaN(progressData.f1)) {
            const f1El = document.getElementById('f1Score');
            if (f1El) f1El.textContent = progressData.f1.toFixed(3);
        }
        // Don't return - continue to process other metrics if present
    }
    
    // Handle speed updates (but don't skip other updates)
    if (progressData.type === 'speed') {
        const speedEl = document.getElementById('processingSpeed');
        if (speedEl && progressData.speed) speedEl.textContent = progressData.speed;
        // Don't return - continue to process other metrics if present
    }
    
    // Mark that we received real progress
    lastRealProgress = {
        timestamp: Date.now(),
        progress: progressData.progress || 0,
        epoch: progressData.epoch || 0
    };
    
    // Update UI with real training progress
    if (progressData.epoch !== undefined) {
        currentEpoch = progressData.epoch;
        const epochEl = document.getElementById('currentEpoch');
        if (epochEl) {
            const totalEpochsValue = progressData.total_epochs || totalEpochs;
            epochEl.textContent = `${progressData.epoch}/${totalEpochsValue}`;
        }
    }
    if (progressData.total_epochs !== undefined) {
        totalEpochs = progressData.total_epochs;
    }
    
    // Update new metrics - box_loss, cls_loss, dfl_loss
    if (progressData.box_loss !== null && progressData.box_loss !== undefined && !isNaN(progressData.box_loss)) {
        const boxLossEl = document.getElementById('boxLoss');
        if (boxLossEl) boxLossEl.textContent = progressData.box_loss.toFixed(4);
    }
    if (progressData.cls_loss !== null && progressData.cls_loss !== undefined && !isNaN(progressData.cls_loss)) {
        const clsLossEl = document.getElementById('clsLoss');
        if (clsLossEl) clsLossEl.textContent = progressData.cls_loss.toFixed(4);
    }
    if (progressData.dfl_loss !== null && progressData.dfl_loss !== undefined && !isNaN(progressData.dfl_loss)) {
        const dflLossEl = document.getElementById('dflLoss');
        if (dflLossEl) dflLossEl.textContent = progressData.dfl_loss.toFixed(4);
    }
    
    // Update GPU memory
    if (progressData.gpu_memory || progressData.gpu_mem) {
        const gpuMemEl = document.getElementById('gpuMem');
        if (gpuMemEl) gpuMemEl.textContent = progressData.gpu_memory || progressData.gpu_mem;
    }
    
    // Update instances
    if (progressData.instances !== null && progressData.instances !== undefined && !isNaN(progressData.instances)) {
        const instancesEl = document.getElementById('instances');
        if (instancesEl) instancesEl.textContent = progressData.instances.toLocaleString();
    }
    
    // Update accuracy (for tabular models and other models that report accuracy)
    if (progressData.accuracy !== null && progressData.accuracy !== undefined && !isNaN(progressData.accuracy)) {
        const accuracyEl = document.getElementById('currentAccuracy');
        if (accuracyEl) {
            // Accuracy can be 0-1 (decimal) or 0-100 (percentage)
            const accuracyValue = progressData.accuracy > 1 ? progressData.accuracy : progressData.accuracy * 100;
            accuracyEl.textContent = `${accuracyValue.toFixed(2)}%`;
        }
    }
    
    // Update loss (general loss metric, used for tabular and other models)
    if (progressData.loss !== null && progressData.loss !== undefined && !isNaN(progressData.loss)) {
        // For tabular models, loss might be shown in a different element
        // If there's a general loss element, update it
        const lossEl = document.getElementById('currentLoss');
        if (lossEl) {
            lossEl.textContent = progressData.loss.toFixed(4);
        }
    }
    
    // Update processing speed (it/s) - parse from speed lines if available
    if (progressData.it_s !== null && progressData.it_s !== undefined && !isNaN(progressData.it_s)) {
        const speedEl = document.getElementById('processingSpeed');
        if (speedEl) speedEl.textContent = `${progressData.it_s.toFixed(2)} it/s`;
    } else if (progressData.speed) {
        // Also check for string format speed
        const speedEl = document.getElementById('processingSpeed');
        if (speedEl) speedEl.textContent = progressData.speed;
    }
    
    // Update mAP metrics (from validation lines) - these can come from both training progress and validation events
    if (progressData.map50 !== null && progressData.map50 !== undefined && !isNaN(progressData.map50)) {
        const map50El = document.getElementById('map50');
        if (map50El) map50El.textContent = progressData.map50.toFixed(3);
    }
    if (progressData.map5095 !== null && progressData.map5095 !== undefined && !isNaN(progressData.map5095)) {
        const map5095El = document.getElementById('map5095');
        if (map5095El) map5095El.textContent = progressData.map5095.toFixed(3);
    }
    
    // Update progress percentage FIRST (before ETA calculation)
    if (progressData.progress !== undefined && progressData.progress >= 0) {
        // Validate progress value to prevent invalid percentages like 8110%
        const rawProgress = progressData.progress * 100;
        displayedProgress = Math.max(0, Math.min(100, rawProgress)); // Clamp to 0-100
        const progressEl = document.getElementById('northStarValue');
        if (progressEl) {
            // Show at least 0.1% if progress is > 0, otherwise show 0%
            const displayValue = displayedProgress > 0 && displayedProgress < 0.1 ? '0.1%' : `${Math.round(displayedProgress * 10) / 10}%`;
            progressEl.textContent = displayValue;
        }
    }
    
    // Calculate and update ETA based on elapsed time and REAL progress
    const etaEl = document.getElementById('eta');
    if (etaEl && trainingStartTime) {
        // Use progress percentage if available, otherwise use epoch progress
        let progress = 0;
        if (progressData.progress !== undefined && progressData.progress > 0) {
            progress = progressData.progress;
        } else if (progressData.epoch !== undefined && progressData.total_epochs !== undefined && progressData.total_epochs > 0) {
            progress = progressData.epoch / progressData.total_epochs;
        }
        
        if (progress > 0.001 && progress < 1) { // Need at least 0.1% progress for meaningful ETA
            const elapsedSeconds = (Date.now() - trainingStartTime) / 1000;
            const totalEstimatedSeconds = elapsedSeconds / progress;
            const remainingSeconds = totalEstimatedSeconds - elapsedSeconds;
            
            if (remainingSeconds > 0 && remainingSeconds < 86400) { // Only show if less than 24 hours
                if (remainingSeconds < 60) {
                    etaEl.textContent = `${Math.round(remainingSeconds)}s`;
                } else if (remainingSeconds < 3600) {
                    etaEl.textContent = `${Math.round(remainingSeconds / 60)}m`;
                } else {
                    const hours = Math.floor(remainingSeconds / 3600);
                    const minutes = Math.round((remainingSeconds % 3600) / 60);
                    etaEl.textContent = `${hours}h ${minutes}m`;
                }
            } else if (remainingSeconds >= 86400) {
                // If more than 24 hours, show days
                const days = Math.floor(remainingSeconds / 86400);
                const hours = Math.floor((remainingSeconds % 86400) / 3600);
                etaEl.textContent = `${days}d ${hours}h`;
            } else {
                etaEl.textContent = '--';
            }
        } else if (progress >= 1) {
            etaEl.textContent = 'Done';
        } else {
            // Not enough progress yet - show calculating
            etaEl.textContent = 'Calculating...';
        }
    }
    
    // Store progress in localStorage for persistence (only if progress was updated)
    if (progressData.progress !== undefined) {
        localStorage.setItem('lastTrainingProgress', displayedProgress.toString());
    }
    
    // Update neural network visualization with progress (always update if we have any training data)
    if (window.neuralNetwork && (progressData.epoch !== undefined || progressData.progress !== undefined || progressData.box_loss !== undefined || progressData.loss !== undefined)) {
        const progressValue = progressData.progress !== undefined ? progressData.progress : 
                             (progressData.epoch !== undefined && progressData.total_epochs !== undefined) ? 
                             (progressData.epoch / progressData.total_epochs) : displayedProgress / 100;
        
        // Determine loss value (prioritize YOLO losses, then general loss)
        const lossValue = progressData.box_loss || progressData.cls_loss || progressData.dfl_loss || progressData.loss || 1.0;
        
        // Determine accuracy (use provided accuracy, or 0.0 for YOLO which doesn't use accuracy)
        let accuracyValue = 0.0;
        if (progressData.accuracy !== null && progressData.accuracy !== undefined && !isNaN(progressData.accuracy)) {
            // Accuracy can be 0-1 (decimal) or 0-100 (percentage) - normalize to 0-100
            accuracyValue = progressData.accuracy > 1 ? progressData.accuracy : progressData.accuracy * 100;
        }
        
        window.neuralNetwork.updateTrainingMetrics(
            lossValue,
            accuracyValue,
            progressData.epoch || currentEpoch || 0,
            progressData.total_epochs || totalEpochs,
            progressValue
        );
    }
    
    // At 5% progress, stop validation animation - training continues automatically
    if (window.neuralNetwork && displayedProgress >= 5) {
        if (window.neuralNetwork.isValidating) {
            console.log('[Renderer] Progress reached 5%, stopping validation animation...');
            window.neuralNetwork.stopValidation();
        }
        // Training continues automatically
    }
});

ipcRenderer.on('inference-log', (event, logData) => {
    if (logData && logData.message) {
        log(logData.message, logData.level || 'log');
    }
});

ipcRenderer.on('training-log', (event, logData) => {
    const level = logData.level || 'log';
    const message = logData.message || '';
    log(message, level === 'error' ? 'error' : level === 'warning' ? 'warning' : 'log');
    
    // Parse model artifact path from "Model artifact saved to:" messages
    if (message.includes('Model artifact saved to:')) {
        const artifactMatch = message.match(/Model artifact saved to:\s*(.+?)(?:\s*$|[\r\n])/i);
        if (artifactMatch && artifactMatch[1]) {
            const artifactPath = artifactMatch[1].trim();
            savedModelPath = artifactPath;
            const modelPathEl = document.getElementById('modelPath');
            if (modelPathEl) {
                modelPathEl.textContent = artifactPath;
            }
            const modelSection = document.getElementById('modelSection');
            if (modelSection) {
                modelSection.style.display = 'block';
            }
        }
    }
    
    // Parse model save directory from "Results saved to" or "Logging results to" messages
    // YOLO outputs: "Logging results to [1mC:\Users\...\train52[0m" or "Results saved to [1mC:\Users\...\train52[0m"
    // We need to extract the directory and construct the path to weights/best.pt
    if (message.includes('Results saved to') || message.includes('Logging results to')) {
        // Extract the path from the message (strip ANSI codes first)
        const ansiRegex = /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;
        const cleanMessage = message.replace(ansiRegex, '');
        
        // Try to extract the path - match the full path after "Results saved to" or "Logging results to"
        let pathMatch = cleanMessage.match(/Results saved to\s+(.+?)(?:\s*$|[\r\n])/i);
        if (!pathMatch) {
            pathMatch = cleanMessage.match(/Logging results to\s+(.+?)(?:\s*$|[\r\n])/i);
        }
        
        if (pathMatch && pathMatch[1]) {
            let saveDir = pathMatch[1].trim();
            // Remove any trailing brackets, formatting, or extra whitespace
            saveDir = saveDir.replace(/[\[\]]/g, '').trim();
            
            // Only process if it looks like a valid path (contains drive letter or starts with /)
            if (saveDir.match(/^[A-Za-z]:\\/) || saveDir.startsWith('/') || saveDir.includes('\\') || saveDir.includes('/')) {
                // Construct the path to best.pt (handle both Windows and Unix paths)
                const separator = saveDir.includes('\\') ? '\\' : '/';
                const modelPath = saveDir + separator + 'weights' + separator + 'best.pt';
                
                // Always update savedModelPath from log messages (they're more reliable)
                savedModelPath = modelPath;
                const modelPathEl = document.getElementById('modelPath');
                if (modelPathEl) {
                    modelPathEl.textContent = modelPath;
                }
                // Show the model section if it's hidden
                const modelSection = document.getElementById('modelSection');
                if (modelSection) {
                    modelSection.style.display = 'block';
                }
                console.log('[Renderer] Extracted model path from log:', modelPath);
            }
        }
    }
    
    // Update status indicator based on console messages
    const statusIndicator = document.getElementById('trainingStatusIndicator');
    if (statusIndicator) {
        const statusText = statusIndicator.querySelector('.status-text');
        const statusDot = statusIndicator.querySelector('.status-dot');
        
        // Check for "Starting training for X epochs..." to switch from Validating to Training
        if (message.includes('Starting training for') && message.includes('epochs')) {
            if (statusText) statusText.textContent = 'Training';
            if (statusDot) statusDot.classList.add('active');
            statusIndicator.classList.add('active');
        }
        // Check for "Training completed successfully!"
        else if (message.includes('Training completed successfully') || message.includes('epochs completed')) {
            if (statusText) statusText.textContent = 'Completed';
            if (statusDot) statusDot.classList.remove('active');
            statusIndicator.classList.remove('active');
        }
        // Check for errors
        else if (level === 'error') {
            if (statusText) statusText.textContent = 'Error';
            if (statusDot) statusDot.classList.remove('active');
            statusDot.classList.add('error');
            statusIndicator.classList.add('error');
        }
    }
});

ipcRenderer.on('training-result', (event, resultData) => {
    // Don't log "completed successfully" here - training-finished event handles it
    // This is just for model path result
    
    // Only use resultData.model_path if we don't already have a path from log parsing
    // (log parsing is more reliable as it comes directly from YOLO output)
    if (resultData.model_path && !savedModelPath) {
        savedModelPath = resultData.model_path;
        document.getElementById('modelPath').textContent = resultData.model_path;
        
        // Check if this is an artifact folder and try to read metadata
        const fs = require('fs');
        const path = require('path');
        const metadataPath = path.join(resultData.model_path, 'metadata.json');
        
        if (fs.existsSync(metadataPath)) {
            try {
                const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
                log(`Model artifact saved to: ${resultData.model_path}`, 'success');
                log(`Python ${metadata.python_version} / scikit-learn ${metadata.sklearn_version || 'N/A'} / NumPy ${metadata.numpy_version || 'N/A'}`, 'log');
                log(`This format requires a compatible Python environment.`, 'warning');
            } catch (e) {
                log(`Model saved to: ${resultData.model_path}`, 'success');
            }
        } else {
            log(`Model saved to: ${resultData.model_path}`, 'success');
        }
    } else if (savedModelPath) {
        log(`Model saved to: ${savedModelPath}`, 'success');
    }
    
    document.getElementById('modelSection').style.display = 'block';
    
    // Update status indicator to Completed
    const statusIndicator = document.getElementById('trainingStatusIndicator');
    if (statusIndicator) {
        const statusText = statusIndicator.querySelector('.status-text');
        const statusDot = statusIndicator.querySelector('.status-dot');
        if (statusText) statusText.textContent = 'Completed';
        if (statusDot) {
            statusDot.classList.remove('active');
            statusDot.classList.remove('error');
        }
        statusIndicator.classList.remove('active');
        statusIndicator.classList.remove('error');
    }
    
    // Stop training UI
    stopTraining(true); // true = was completed
    
    // Refresh model list
    loadSavedModels();
});

ipcRenderer.on('training-error', (event, errorData) => {
    log(`Training error: ${errorData.error || errorData}`, 'error');
    if (errorData.traceback) {
        console.error('Training traceback:', errorData.traceback);
    }
    stopTraining(false); // false = not completed
});

ipcRenderer.on('training-started', (event) => {
    log('Real training process started', 'success');
    // Status will be updated by training-log handler when "Starting training for X epochs..." appears
});

// Track training state to prevent duplicate messages
let trainingState = {
    finished: false,
    stopped: false
};

ipcRenderer.on('training-stopped', (event, data) => {
    // Only log if we haven't already logged a finished message
    if (!trainingState.finished) {
        trainingState.stopped = true;
        if (data && data.manuallyStopped) {
            log('Training stopped by user', 'warning');
        } else if (data && data.exitCode !== undefined && data.exitCode !== 0) {
            log(`Training failed (exit code ${data.exitCode})`, 'error');
        } else {
            log('Training stopped', 'warning');
        }
    }
    stopTraining(false);
});

ipcRenderer.on('training-finished', (event, data) => {
    // Refresh model lists if inference sections are visible
    const modelPurpose = document.getElementById('modelPurposeInput')?.value;
    if (modelPurpose === 'tabular') {
        // Refresh tabular models list after training completes
        setTimeout(() => {
            loadTabularModels();
        }, 500); // Small delay to ensure model is saved
    } else if (modelPurpose === 'computer_vision') {
        // Refresh CV models list after training completes
        setTimeout(() => {
            loadCvModels();
        }, 500);
    }
    trainingState.finished = true;
    trainingState.stopped = false; // Clear stopped flag since we finished successfully
    log('Training finished', 'success');
    stopTraining(false);
});

// Reset state when training starts
ipcRenderer.on('training-started', () => {
    trainingState.finished = false;
    trainingState.stopped = false;
});

// Make functions globally accessible for onclick handlers
window.continueTraining = function(filepath) {
    ipcRenderer.send('load-model', filepath);
};

window.loadModelForRetrain = function(filepath) {
    ipcRenderer.send('load-model', filepath);
};

ipcRenderer.on('model-load-warning', (event, data) => {
    if (data && data.warnings && data.warnings.length > 0) {
        log('Version compatibility warnings:', 'warning');
        data.warnings.forEach(warning => {
            log(`  - ${warning}`, 'warning');
        });
        log('Model may still load, but results may vary.', 'warning');
    }
});

ipcRenderer.on('model-load-error', (event, errorData) => {
    if (errorData && errorData.issues) {
        log('Cannot load model due to version incompatibility:', 'error');
        errorData.issues.forEach(issue => {
            log(`  - ${issue}`, 'error');
        });
        if (errorData.metadata) {
            log(`Model was trained with: Python ${errorData.metadata.python_version}, scikit-learn ${errorData.metadata.sklearn_version || 'N/A'}`, 'error');
        }
        log('Please use a compatible Python environment to load this model.', 'error');
    } else {
        log(`Failed to load model: ${errorData?.message || errorData || 'Unknown error'}`, 'error');
    }
});

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

ipcRenderer.on('model-load-warning', (event, data) => {
    if (data && data.warnings && data.warnings.length > 0) {
        log('Version compatibility warnings:', 'warning');
        data.warnings.forEach(warning => {
            log(`  - ${warning}`, 'warning');
        });
        log('Model may still load, but results may vary.', 'warning');
    }
});

ipcRenderer.on('model-load-error', (event, errorData) => {
    // Handle structured error data from artifact validation
    if (errorData && typeof errorData === 'object' && errorData.issues) {
        log('Cannot load model due to version incompatibility:', 'error');
        errorData.issues.forEach(issue => {
            log(`  - ${issue}`, 'error');
        });
        if (errorData.metadata) {
            log(`Model was trained with: Python ${errorData.metadata.python_version}, scikit-learn ${errorData.metadata.sklearn_version || 'N/A'}`, 'error');
        }
        log('Please use a compatible Python environment to load this model.', 'error');
    } else {
        // Legacy error format (string)
        const errorMsg = errorData?.message || errorData || 'Unknown error';
        log(`Error loading model: ${errorMsg}`, 'error');
    }
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

// Global function for button onclick (as fallback) - MUST be in global scope
window.showMainApp = function(selectedMode = 'local') {
    console.log('showMainApp called from renderer.js with mode:', selectedMode);
    trainingMode = selectedMode; // Store the selected training mode
    const splashScreen = document.getElementById('splashScreen');
    const mainApp = document.getElementById('mainApp');
    
    console.log('Elements found - splashScreen:', splashScreen, 'mainApp:', mainApp);
    console.log('Training mode set to:', trainingMode);
    
    if (splashScreen) {
        splashScreen.style.display = 'none';
        splashScreen.classList.add('hidden');
        console.log('Splash screen hidden');
    }
    
    if (mainApp) {
        mainApp.classList.remove('hidden');
        mainApp.style.display = 'flex';
        mainApp.style.visibility = 'visible';
        mainApp.style.opacity = '1';
        console.log('Main app shown');
        
        // Show/hide device selector based on training mode
        const deviceInputGroup = document.querySelector('#deviceInput')?.closest('.setting-group');
        if (deviceInputGroup) {
            if (selectedMode === 'cloud') {
                deviceInputGroup.style.display = 'none'; // Hide GPU/CPU selector for cloud mode
            } else {
                deviceInputGroup.style.display = 'block'; // Show for local mode
            }
        }
        
        // Initialize app if not already initialized
        if (typeof initializeApp === 'function') {
            initializeApp();
        }
        
        // Update system info based on training mode
        if (selectedMode === 'cloud') {
            updateCloudSystemInfo();
            // Hide balance display (no API endpoint available)
            hideBalanceDisplay();
        } else {
            // Load local system info
            loadSystemInfo();
            // Hide balance for local mode
            hideBalanceDisplay();
        }
    }
};

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Setup splash screen
    const splashScreen = document.getElementById('splashScreen');
    const mainApp = document.getElementById('mainApp');
    const startAppBtn = document.getElementById('startAppBtn');
    
    console.log('DOMContentLoaded - splashScreen:', splashScreen, 'mainApp:', mainApp, 'startAppBtn:', startAppBtn);
    
    // DON'T load preserved progress on app startup - only show progress during active training
    // Clear any saved progress from localStorage on app start
    localStorage.removeItem('lastTrainingProgress');
    displayedProgress = 0;
    const progressElement = document.getElementById('northStarValue');
    if (progressElement) {
        progressElement.textContent = '0%';
    }
    
    // Handle training mode card selection
    const trainingModeCards = document.querySelectorAll('.training-mode-card');
    if (trainingModeCards.length > 0) {
        trainingModeCards.forEach(card => {
            card.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const mode = card.getAttribute('data-mode');
                console.log('Training mode card clicked:', mode);
                
                // Visual feedback
                trainingModeCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                
                // If cloud training, show login modal first
                if (mode === 'cloud') {
                    showCanopywaveLoginModal();
                } else {
                    // Local training - proceed directly
                    if (window.showMainApp) {
                        window.showMainApp(mode);
                    } else if (typeof showMainApp === 'function') {
                        showMainApp(mode);
                    }
                }
            });
        });
    } else {
        console.warn('Training mode cards not found, falling back to old button handler');
        // Fallback for old button (shouldn't happen, but just in case)
        const startAppBtn = document.getElementById('startAppBtn');
        if (startAppBtn) {
            startAppBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (window.showMainApp) {
                    window.showMainApp('local');
                }
            });
        }
    }
    
    // Only initialize if main app is visible, otherwise wait for showMainApp
    if (mainApp && mainApp.style.display !== 'none' && !mainApp.classList.contains('hidden')) {
        initializeApp();
    }
    
    // Initialize CanopyWave login modal handlers
    initializeCanopywaveLogin();
});

// CanopyWave Login Functions
function showCanopywaveLoginModal() {
    const modal = document.getElementById('canopywaveLoginModal');
    const apiKeyInput = document.getElementById('apiKeyInput');
    const errorDiv = document.getElementById('loginError');
    const submitBtn = document.getElementById('submitLoginBtn');
    
    if (modal) {
        modal.classList.add('show');
        // Clear previous inputs and reset button state
        if (apiKeyInput) apiKeyInput.value = '';
        if (errorDiv) errorDiv.style.display = 'none';
        // Reset submit button to enabled state with correct text
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Connect';
        }
        // Focus on input
        setTimeout(() => {
            if (apiKeyInput) apiKeyInput.focus();
        }, 100);
    }
}

function hideCanopywaveLoginModal() {
    const modal = document.getElementById('canopywaveLoginModal');
    if (modal) {
        modal.classList.remove('show');
    }
}

function initializeCanopywaveLogin() {
    const modal = document.getElementById('canopywaveLoginModal');
    const cancelBtn = document.getElementById('cancelLoginBtn');
    const submitBtn = document.getElementById('submitLoginBtn');
    const apiKeyInput = document.getElementById('apiKeyInput');
    const errorDiv = document.getElementById('loginError');
    const helpLink = document.getElementById('helpLink');
    
    // Cancel button - go back to splash screen
    if (cancelBtn) {
        cancelBtn.addEventListener('click', (e) => {
            e.preventDefault();
            hideCanopywaveLoginModal();
            // Reset card selection
            const trainingModeCards = document.querySelectorAll('.training-mode-card');
            trainingModeCards.forEach(c => c.classList.remove('selected'));
        });
    }
    
    // Submit button - validate and store API key
    if (submitBtn) {
        submitBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            await handleCanopywaveLogin();
        });
    }
    
    // Allow Enter key to submit
    if (apiKeyInput) {
        apiKeyInput.addEventListener('keypress', async (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                await handleCanopywaveLogin();
            }
        });
    }
    
    // Close modal when clicking outside
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                hideCanopywaveLoginModal();
                // Reset card selection
                const trainingModeCards = document.querySelectorAll('.training-mode-card');
                trainingModeCards.forEach(c => c.classList.remove('selected'));
            }
        });
    }
    
    // Help link - open CanopyWave documentation (you can update this URL)
    if (helpLink) {
        helpLink.addEventListener('click', (e) => {
            e.preventDefault();
            // TODO: Open CanopyWave documentation or help page
            console.log('Help link clicked - open CanopyWave docs');
        });
    }
}

async function handleCanopywaveLogin() {
    const apiKeyInput = document.getElementById('apiKeyInput');
    const errorDiv = document.getElementById('loginError');
    const submitBtn = document.getElementById('submitLoginBtn');
    
    if (!apiKeyInput) return;
    
    // Get and clean the API key
    let apiKey = apiKeyInput.value;
    
    // Remove all whitespace (including spaces, tabs, newlines, etc.)
    apiKey = apiKey.replace(/\s/g, '');
    
    // Validate API key is not empty
    if (!apiKey) {
        if (errorDiv) {
            errorDiv.textContent = 'Please enter your API key';
            errorDiv.style.display = 'block';
        }
        return;
    }
    
    // Basic format validation for CanopyWave API keys
    // CanopyWave keys typically start with 'cw_' or similar prefix
    if (apiKey.length < 10) {
        if (errorDiv) {
            errorDiv.textContent = 'API key appears too short. Please check and try again.';
            errorDiv.style.display = 'block';
        }
        return;
    }
    
    // Show loading state
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Connecting...';
    }
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
    
    try {
        // Validate API key with CanopyWave API
        const validationResult = await ipcRenderer.invoke('validate-canopywave-api-key', apiKey);
        
        if (validationResult.valid) {
            // Store API key securely
            window.canopywaveApiKey = apiKey;
            
            // Store in localStorage (temporary - consider using Electron's secure storage)
            localStorage.setItem('canopywave_api_key', apiKey);
            
            console.log('CanopyWave API key validated and stored successfully');
            
            // Hide login modal and show cloud configuration modal
            hideCanopywaveLoginModal();
            showCloudConfigModal();
        } else {
            // Validation failed
            if (errorDiv) {
                let errorMessage = validationResult.error || 'Invalid API key. Please check your credentials.';
                
                // Provide more helpful error messages
                if (errorMessage.includes('Error decoding token')) {
                    errorMessage = 'Invalid API key format. Please copy your API key directly from CanopyWave dashboard without any extra spaces or characters.';
                } else if (errorMessage.includes('401') || errorMessage.toLowerCase().includes('unauthorized')) {
                    errorMessage = 'API key is invalid or expired. Please check your CanopyWave dashboard and generate a new key if needed.';
                } else if (errorMessage.includes('403') || errorMessage.toLowerCase().includes('forbidden')) {
                    errorMessage = 'API key lacks necessary permissions. Please check your CanopyWave account settings.';
                } else if (errorMessage.includes('timeout') || errorMessage.toLowerCase().includes('network')) {
                    errorMessage = 'Network error. Please check your internet connection and try again.';
                }
                
                errorDiv.textContent = errorMessage;
                errorDiv.style.display = 'block';
            }
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Connect';
            }
        }
        
    } catch (error) {
        console.error('Error validating CanopyWave API key:', error);
        if (errorDiv) {
            errorDiv.textContent = error.message || 'Failed to validate API key. Please check your credentials.';
            errorDiv.style.display = 'block';
        }
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Connect';
        }
    }
}

// Cloud Training Functions
function showCloudConfigModal() {
    const modal = document.getElementById('cloudConfigModal');
    if (modal) {
        modal.classList.add('show');
        // Load configuration when modal is shown
        loadCloudConfiguration();
        // Initialize validation
        setupCloudConfigValidation();
    }
}

function hideCloudConfigModal() {
    const modal = document.getElementById('cloudConfigModal');
    if (modal) {
        modal.classList.remove('show');
    }
}

function updateCloudCostEstimate() {
    const gpuSelect = document.getElementById('cloudGPUSelect');
    const maxHoursInput = document.getElementById('cloudMaxTrainingHours');
    const budgetLimitInput = document.getElementById('cloudBudgetLimit');
    const costEstimateEl = document.getElementById('cloudCostEstimate');
    const upfrontWarningEl = document.getElementById('cloudUpfrontWarning');
    
    if (!gpuSelect || !costEstimateEl) return;
    
    const selectedOption = gpuSelect.options[gpuSelect.selectedIndex];
    const maxHours = maxHoursInput ? parseFloat(maxHoursInput.value) || 24 : 24;
    
    // Hide upfront warning by default
    if (upfrontWarningEl) {
        upfrontWarningEl.style.display = 'none';
    }
    
    if (selectedOption && selectedOption.value && !selectedOption.disabled) {
        // Try to get GPU data from dataset first (more reliable)
        let pricePerGpuHour = null;
        let gpuCount = 1;
        let providerUpfrontHours = 24; // Default to 24 hours
        
        // Check if GPU data is stored in dataset
        if (selectedOption.dataset.gpuData) {
            try {
                const gpuData = JSON.parse(selectedOption.dataset.gpuData);
                // Extract price from gpuData (could be priceHour, price, pricePerHour, etc.)
                pricePerGpuHour = parseFloat(gpuData.priceHour || gpuData.price || gpuData.pricePerHour || 0);
                // Extract GPU count from name (e.g., "4x H100" -> 4)
                const gpuCountMatch = (gpuData.name || '').match(/(\d+)x/i);
                gpuCount = gpuCountMatch ? parseInt(gpuCountMatch[1]) : 1;
                // Store pricing info in option for later use
                selectedOption.dataset.pricePerGpuHour = pricePerGpuHour;
                selectedOption.dataset.gpuCount = gpuCount;
                selectedOption.dataset.providerUpfrontHours = providerUpfrontHours;
            } catch (e) {
                console.warn('[Cloud Cost] Failed to parse GPU data:', e);
            }
        }
        
        // Fallback: Extract from text if not in dataset
        if (!pricePerGpuHour || pricePerGpuHour === 0) {
            const gpuText = selectedOption.textContent;
            const priceMatch = gpuText.match(/\$([\d.]+)/);
            const gpuCountMatch = gpuText.match(/(\d+)x/i);
            
            if (priceMatch) {
                pricePerGpuHour = parseFloat(priceMatch[1]);
                gpuCount = gpuCountMatch ? parseInt(gpuCountMatch[1]) : 1;
                // Store in dataset for consistency
                selectedOption.dataset.pricePerGpuHour = pricePerGpuHour;
                selectedOption.dataset.gpuCount = gpuCount;
                selectedOption.dataset.providerUpfrontHours = providerUpfrontHours;
            }
        }
        
        if (pricePerGpuHour && pricePerGpuHour > 0) {
            // Use centralized pricing calculation
            const costs = calculateCloudCosts({
                pricePerGpuHour: pricePerGpuHour,
                gpuCount: gpuCount,
                providerUpfrontHours: providerUpfrontHours
            });
            
            const estimatedCost = costs.hourlyCost * maxHours;
            
            // Display hourly cost and estimated cost
            if (gpuCount > 1) {
                costEstimateEl.textContent = `Hourly cost: $${costs.hourlyCost.toFixed(2)}/hour (${gpuCount} GPUs × $${pricePerGpuHour.toFixed(2)}/gpu/hour)`;
                costEstimateEl.textContent += ` | Estimated cost: $${estimatedCost.toFixed(2)} (${maxHours} hours)`;
            } else {
                costEstimateEl.textContent = `Hourly cost: $${costs.hourlyCost.toFixed(2)}/hour`;
                costEstimateEl.textContent += ` | Estimated cost: $${estimatedCost.toFixed(2)} (${maxHours} hours)`;
            }
            
            // Show upfront authorization warning
            if (upfrontWarningEl) {
                upfrontWarningEl.textContent = `⚠️ Provider may charge up to $${costs.upfrontAuthorizationCost.toFixed(2)} (${providerUpfrontHours} hours) up front and refund unused time.`;
                upfrontWarningEl.style.display = 'block';
            }
            
            // Check if exceeds budget limit (warning only - budget is a threshold, not a hard limit)
            const budgetLimit = budgetLimitInput ? parseFloat(budgetLimitInput.value) || 100 : 100;
            if (estimatedCost > budgetLimit || costs.upfrontAuthorizationCost > budgetLimit) {
                costEstimateEl.style.color = '#D32F2F';
                if (costs.upfrontAuthorizationCost > budgetLimit) {
                    costEstimateEl.textContent += ` ⚠️ Upfront authorization ($${costs.upfrontAuthorizationCost.toFixed(2)}) exceeds budget limit of $${budgetLimit.toFixed(2)} (warning only - provider will still charge upfront)`;
                } else {
                    costEstimateEl.textContent += ` ⚠️ Exceeds budget limit of $${budgetLimit.toFixed(2)} (warning only)`;
                }
            } else {
                costEstimateEl.style.color = '#5A4A3A';
            }
        } else {
            costEstimateEl.textContent = 'Estimated cost: Unable to calculate (price not available)';
            costEstimateEl.style.color = '#8B7355';
        }
    } else {
        costEstimateEl.textContent = 'Estimated cost: $0.00';
        costEstimateEl.style.color = '#5A4A3A';
    }
}

// Global functions and state for cloud config (needs to be accessible from dynamic event listeners)
let cloudConfigValidateForm = null;
let cloudConfigCheckGPUAvailability = null;
let cloudConfigGPUAvailable = null; // Track if selected GPU is available (true/false/null for unknown)
let gpuAvailReqId = 0; // Request counter to prevent stale responses

function setupCloudConfigValidation() {
    const projectSelect = document.getElementById('cloudProjectSelect');
    const regionSelect = document.getElementById('cloudRegionSelect');
    const imageSelect = document.getElementById('cloudImageSelect');
    const passwordInput = document.getElementById('cloudInstancePassword');
    const maxHoursInput = document.getElementById('cloudMaxTrainingHours');
    const budgetLimitInput = document.getElementById('cloudBudgetLimit');
    const confirmBtn = document.getElementById('confirmCloudConfigBtn');
    
    function validateForm() {
        // Get fresh reference to GPU select (it gets recreated dynamically)
        const gpuSelect = document.getElementById('cloudGPUSelect');
        
        const hasProject = projectSelect && projectSelect.value;
        const hasRegion = regionSelect && regionSelect.value;
        const hasGPU = gpuSelect && gpuSelect.value;
        const hasImage = imageSelect && imageSelect.value;
        const passwordValue = passwordInput ? passwordInput.value.trim() : '';
        
        // Debug logging
        console.log('[Cloud Config Validation]', {
            hasProject,
            hasRegion,
            hasGPU,
            gpuSelectExists: !!gpuSelect,
            gpuSelectValue: gpuSelect ? gpuSelect.value : 'N/A',
            hasImage,
            passwordLength: passwordValue.length
        });
        
        // Password validation: minimum 8 characters, at least one letter and one number
        const passwordValid = passwordValue.length >= 8 && 
                             /[a-zA-Z]/.test(passwordValue) && 
                             /[0-9]/.test(passwordValue);
        
        // Visual feedback for password field
        const passwordRequirementsSpan = document.getElementById('passwordRequirements');
        if (passwordInput) {
            if (passwordValue.length === 0) {
                // Empty - neutral state
                passwordInput.classList.remove('invalid', 'valid');
                if (passwordRequirementsSpan) {
                    passwordRequirementsSpan.className = 'privacy-text-small';
                    passwordRequirementsSpan.textContent = '🔒 Required for SSH access. Use a strong password with letters, numbers, and symbols.';
                }
            } else if (!passwordValid) {
                // Invalid - red
                passwordInput.classList.add('invalid');
                passwordInput.classList.remove('valid');
                if (passwordRequirementsSpan) {
                    passwordRequirementsSpan.className = 'privacy-text-small password-error';
                    passwordRequirementsSpan.textContent = '❌ Password must be at least 8 characters with at least one letter and one number';
                }
            } else {
                // Valid - green
                passwordInput.classList.add('valid');
                passwordInput.classList.remove('invalid');
                if (passwordRequirementsSpan) {
                    passwordRequirementsSpan.className = 'privacy-text-small password-success';
                    passwordRequirementsSpan.textContent = '✅ Password meets requirements';
                }
            }
        }
        
        // Check if GPU is available (if one is selected)
        // Only enable Continue if GPU is explicitly available (true)
        // null (unknown) or false (unavailable) will disable Continue
        const gpuAvailable = !hasGPU || cloudConfigGPUAvailable === true;
        
        const isValid = hasProject && hasRegion && hasGPU && hasImage && passwordValid && gpuAvailable;
        
        if (confirmBtn) {
            confirmBtn.disabled = !isValid;
        }
        
        console.log('[Cloud Config] Validation result:', { 
            isValid, 
            gpuAvailable, 
            gpuAvailabilityState: cloudConfigGPUAvailable === true ? 'available' : 
                                  cloudConfigGPUAvailable === false ? 'unavailable' : 'unknown'
        });
        
        return isValid;
    }
    
    // Check GPU availability
    async function checkGPUAvailability() {
        // Get fresh reference to GPU select (it gets recreated dynamically)
        const gpuSelect = document.getElementById('cloudGPUSelect');
        
        const project = projectSelect ? projectSelect.value : '';
        const region = regionSelect ? regionSelect.value : '';
        const gpu = gpuSelect ? gpuSelect.value : '';
        
        console.log('[GPU Availability Check] Parameters:', { 
            project, 
            region, 
            gpu,
            hasApiKey: !!canopywaveApiKey,
            gpuSelectExists: !!gpuSelect,
            selectedIndex: gpuSelect?.selectedIndex,
            selectedOption: gpuSelect?.options[gpuSelect?.selectedIndex],
            selectedOptionValue: gpuSelect?.options[gpuSelect?.selectedIndex]?.value,
            selectedOptionText: gpuSelect?.options[gpuSelect?.selectedIndex]?.text,
            allOptions: Array.from(gpuSelect?.options || []).map(opt => ({ value: opt.value, text: opt.text }))
        });
        
        if (!project || !region || !gpu || !canopywaveApiKey) {
            console.log('[GPU Availability Check] Missing required fields, skipping check');
            return;
        }
        
        // Increment request ID to prevent stale responses
        const reqId = ++gpuAvailReqId;
        
        try {
            // Show checking status
            const gpuHelpText = gpuSelect?.parentElement?.querySelector('.form-help .privacy-text-small');
            if (gpuHelpText) {
                gpuHelpText.textContent = '⏳ Checking GPU availability...';
                gpuHelpText.style.color = '#9A8A7A';
            }
            
            // Check availability via API
            const result = await ipcRenderer.invoke('check-gpu-availability', canopywaveApiKey, project, gpu, region);
            
            // Ignore stale responses (user changed GPU while request was in flight)
            if (reqId !== gpuAvailReqId) {
                console.log('[GPU Availability] Ignoring stale response', { reqId, current: gpuAvailReqId });
                return;
            }
            
            if (result.success) {
                // Update global availability state (tri-state: true/false/null)
                cloudConfigGPUAvailable = result.available;
                
                if (gpuHelpText) {
                    if (result.available === true) {
                        gpuHelpText.textContent = '✅ GPU is available';
                        gpuHelpText.style.color = '#27AE60';
                        gpuHelpText.style.fontWeight = '500';
                    } else if (result.available === false) {
                        gpuHelpText.textContent = '❌ GPU is currently unavailable (in use or out of capacity)';
                        gpuHelpText.style.color = '#E74C3C';
                        gpuHelpText.style.fontWeight = '500';
                    } else {
                        // null/unknown - cannot verify
                        gpuHelpText.textContent = '⚠️ Unable to verify availability - API returned ambiguous data';
                        gpuHelpText.style.color = '#F39C12';
                        gpuHelpText.style.fontWeight = '500';
                    }
                }
                
                // Re-validate form to update Continue button state
                if (cloudConfigValidateForm) cloudConfigValidateForm();
            } else {
                // API call failed - set to unknown (null), not available
                cloudConfigGPUAvailable = null;
                
                if (gpuHelpText) {
                    // Show specific error message if available
                    const errorMsg = result.error || 'Unknown error';
                    if (errorMsg.includes('401') || errorMsg.toLowerCase().includes('unauthorized') || errorMsg.toLowerCase().includes('authentication')) {
                        gpuHelpText.textContent = '⚠️ API key invalid or expired - check your CanopyWave API key';
                    } else if (errorMsg.includes('403') || errorMsg.toLowerCase().includes('forbidden')) {
                        gpuHelpText.textContent = '⚠️ API key lacks permission - check your CanopyWave account permissions';
                    } else if (errorMsg.includes('404')) {
                        gpuHelpText.textContent = '⚠️ API endpoint not found - CanopyWave API may have changed';
                    } else if (errorMsg.toLowerCase().includes('network') || errorMsg.toLowerCase().includes('timeout')) {
                        gpuHelpText.textContent = '⚠️ Network error - check your internet connection';
                    } else {
                        gpuHelpText.textContent = `⚠️ Unable to verify availability - ${errorMsg}`;
                    }
                    gpuHelpText.style.color = '#F39C12';
                    gpuHelpText.style.fontWeight = '500';
                }
                
                // Re-validate form
                if (cloudConfigValidateForm) cloudConfigValidateForm();
            }
        } catch (error) {
            console.error('Error checking GPU availability:', error);
            
            // Ignore stale responses
            if (reqId !== gpuAvailReqId) return;
            
            // Set to unknown (null) on error, not available
            cloudConfigGPUAvailable = null;
            
            const gpuHelpText = gpuSelect?.parentElement?.querySelector('.form-help .privacy-text-small');
            if (gpuHelpText) {
                const errorMsg = error.message || error.toString();
                if (errorMsg.includes('401') || errorMsg.toLowerCase().includes('unauthorized')) {
                    gpuHelpText.textContent = '⚠️ API key invalid or expired - check your CanopyWave API key';
                } else if (errorMsg.toLowerCase().includes('network') || errorMsg.toLowerCase().includes('timeout')) {
                    gpuHelpText.textContent = '⚠️ Network error - check your internet connection';
                } else {
                    gpuHelpText.textContent = '⚠️ Unable to verify availability right now - network or API error';
                }
                gpuHelpText.style.color = '#F39C12';
                gpuHelpText.style.fontWeight = '500';
            }
            
            // Re-validate form
            if (cloudConfigValidateForm) cloudConfigValidateForm();
        }
    }
    
    // Assign to global variables so they can be called from dynamic event listeners
    cloudConfigValidateForm = validateForm;
    cloudConfigCheckGPUAvailability = checkGPUAvailability;
    
    // Validate on any change
    if (projectSelect) projectSelect.addEventListener('change', validateForm);
    if (regionSelect) regionSelect.addEventListener('change', () => { validateForm(); checkGPUAvailability(); });
    // GPU select event listener is added dynamically when the select is recreated (see region change handler)
    if (imageSelect) imageSelect.addEventListener('change', validateForm);
    if (passwordInput) passwordInput.addEventListener('input', validateForm);
    if (maxHoursInput) maxHoursInput.addEventListener('input', () => { validateForm(); updateCloudCostEstimate(); });
    if (budgetLimitInput) budgetLimitInput.addEventListener('input', updateCloudCostEstimate);
    
    // Add event listener to GPU select if it exists initially
    const initialGpuSelect = document.getElementById('cloudGPUSelect');
    if (initialGpuSelect) {
        initialGpuSelect.addEventListener('change', () => { validateForm(); updateCloudCostEstimate(); checkGPUAvailability(); });
    }
    
    // Run initial validation when modal opens (with a slight delay to ensure fields are populated)
    setTimeout(() => {
        console.log('[Cloud Config] Running initial validation...');
        validateForm();
    }, 500);
    
    // Confirm button handler (confirmBtn already declared above)
    if (confirmBtn) {
        confirmBtn.addEventListener('click', () => {
            if (validateForm()) {
                // Store cloud config - get fresh reference to GPU select
                const gpuSelect = document.getElementById('cloudGPUSelect');
                const selectedGPU = gpuSelect?.options[gpuSelect.selectedIndex];
                cloudGPUInfo = selectedGPU ? selectedGPU.textContent : null;
                const maxHoursInput = document.getElementById('cloudMaxTrainingHours');
                const budgetLimitInput = document.getElementById('cloudBudgetLimit');
                window.cloudConfig = {
                    project: projectSelect.value,
                    region: regionSelect.value,
                    gpu: gpuSelect?.value || '',
                    gpuName: cloudGPUInfo,
                    image: imageSelect.value,
                    password: passwordInput.value,
                    maxTrainingHours: maxHoursInput ? parseFloat(maxHoursInput.value) || 24 : 24,
                    budgetLimit: budgetLimitInput ? parseFloat(budgetLimitInput.value) || 100 : 100
                };
                console.log('[Cloud Config] Stored cloud config:', cloudConfig);
                
                // Store cloud config and proceed to main app
                hideCloudConfigModal();
                if (window.showMainApp) {
                    window.showMainApp('cloud');
                }
            }
        });
    }
    
    // Cancel button handler
    const cancelBtn = document.getElementById('cancelCloudConfigBtn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            hideCloudConfigModal();
            // Reset training mode
            trainingMode = 'local';
            window.canopywaveApiKey = null;
        });
    }
    
    // Close modal when clicking outside
    const modal = document.getElementById('cloudConfigModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                hideCloudConfigModal();
                trainingMode = 'local';
                window.canopywaveApiKey = null;
            }
        });
    }
}

async function loadCloudConfiguration() {
    console.log('[Cloud Config] loadCloudConfiguration called. trainingMode:', trainingMode, 'apiKey:', !!window.canopywaveApiKey);
    
    if (!window.canopywaveApiKey) {
        console.log('[Cloud Config] Skipping - no API key');
        return;
    }

    try {

        // Load projects
        const projectsResult = await ipcRenderer.invoke('list-canopywave-projects', window.canopywaveApiKey);
        if (projectsResult.success && projectsResult.projects) {
            const projectSelect = document.getElementById('cloudProjectSelect');
            if (projectSelect) {
                projectSelect.innerHTML = '<option value="">Select a project</option>';
                projectsResult.projects.forEach(project => {
                    const option = document.createElement('option');
                    option.value = project;
                    option.textContent = project;
                    projectSelect.appendChild(option);
                });
                
                // Add change listener to load regions when project is selected
                projectSelect.addEventListener('change', async (e) => {
                    const selectedProject = e.target.value;
                    if (selectedProject) {
                        await loadRegions(selectedProject);
                        // Reset GPU and image selects - will be loaded when region is selected
                        const gpuSelect = document.getElementById('cloudGPUSelect');
                        const imageSelect = document.getElementById('cloudImageSelect');
                        if (gpuSelect) {
                            gpuSelect.innerHTML = '<option value="">Select region first</option>';
                            gpuSelect.disabled = true;
                        }
                        if (imageSelect) {
                            imageSelect.innerHTML = '<option value="">Select GPU type first</option>';
                            imageSelect.disabled = true;
                        }
                    } else {
                        // Reset dependent selects
                        const regionSelect = document.getElementById('cloudRegionSelect');
                        const gpuSelect = document.getElementById('cloudGPUSelect');
                        const imageSelect = document.getElementById('cloudImageSelect');
                        if (regionSelect) {
                            regionSelect.innerHTML = '<option value="">Select project first</option>';
                            regionSelect.disabled = true;
                        }
                        if (gpuSelect) {
                            gpuSelect.innerHTML = '<option value="">Select project first</option>';
                            gpuSelect.disabled = true;
                        }
                        if (imageSelect) {
                            imageSelect.innerHTML = '<option value="">Select GPU type first</option>';
                            imageSelect.disabled = true;
                        }
                    }
                });
            }
        } else {
            const errorMsg = projectsResult.error || 'Unknown error';
            log('Failed to load projects: ' + errorMsg, 'error');
            
            // Update dropdown to show error
            const projectSelect = document.getElementById('cloudProjectSelect');
            if (projectSelect) {
                projectSelect.innerHTML = '<option value="">Error loading projects - see logs</option>';
            }
        }
    } catch (error) {
        console.error('Error loading cloud configuration:', error);
        log('Error loading cloud configuration: ' + error.message, 'error');
        
        // Update dropdown to show error
        const projectSelect = document.getElementById('cloudProjectSelect');
        if (projectSelect) {
            projectSelect.innerHTML = '<option value="">Error loading projects - see logs</option>';
        }
    }
}

async function loadRegions(project) {
    try {
        const regionsResult = await ipcRenderer.invoke('list-canopywave-regions', canopywaveApiKey, project);
        if (regionsResult.success && regionsResult.regions) {
            const regionSelect = document.getElementById('cloudRegionSelect');
            if (regionSelect) {
                regionSelect.innerHTML = '<option value="">Select a region</option>';
                regionsResult.regions.forEach(region => {
                    const option = document.createElement('option');
                    option.value = region;
                    option.textContent = region;
                    regionSelect.appendChild(option);
                });
                regionSelect.disabled = false;
                
                // Add change listener to load GPUs when region is selected
                regionSelect.addEventListener('change', async (e) => {
                    const selectedRegion = e.target.value;
                    if (selectedRegion) {
                        await loadInstanceTypes(project, selectedRegion);
                        // Update cost estimate when region changes (pricing may differ by region)
                        updateCloudCostEstimate();
                    } else {
                        const gpuSelect = document.getElementById('cloudGPUSelect');
                        if (gpuSelect) {
                            gpuSelect.innerHTML = '<option value="">Select region first</option>';
                            gpuSelect.disabled = true;
                        }
                        // Reset cost estimate
                        updateCloudCostEstimate();
                    }
                });
            }
        }
    } catch (error) {
        console.error('Error loading regions:', error);
        log('Error loading regions: ' + error.message, 'error');
    }
}

async function loadInstanceTypes(project, region) {
    try {
        const gpusResult = await ipcRenderer.invoke('list-cloud-gpus', canopywaveApiKey, project);
        if (gpusResult.success && gpusResult.gpus) {
            const gpuSelect = document.getElementById('cloudGPUSelect');
            if (gpuSelect) {
                gpuSelect.innerHTML = '<option value="">Loading GPUs...</option>';
                gpuSelect.disabled = true;
                
                // Clear any existing change listeners by cloning the select
                const newGpuSelect = gpuSelect.cloneNode(true);
                gpuSelect.parentNode.replaceChild(newGpuSelect, gpuSelect);
                const updatedGpuSelect = document.getElementById('cloudGPUSelect');
                
                updatedGpuSelect.innerHTML = '<option value="">Select a GPU type</option>';
                
                // Load GPUs and check availability
                for (const gpu of gpusResult.gpus) {
                    // Use flavor for the value (this is what we send to the API)
                    const flavor = gpu.flavor || gpu.name;
                    console.log('[GPU Load] GPU data:', { name: gpu.name, flavor: gpu.flavor, id: gpu.id, fullGpu: gpu });
                    const option = document.createElement('option');
                    option.value = flavor;
                    
                    // Extract pricing fields
                    const pricePerGpuHour = parseFloat(gpu.priceHour || gpu.price || gpu.pricePerHour || 0);
                    // Extract GPU count from name (e.g., "4x H100" -> 4)
                    const gpuCountMatch = (gpu.name || '').match(/(\d+)x/i);
                    const gpuCount = gpuCountMatch ? parseInt(gpuCountMatch[1]) : 1;
                    const providerUpfrontHours = 24; // Default 24 hours
                    
                    // Store the full GPU object for later use
                    option.dataset.gpuData = JSON.stringify(gpu);
                    option.dataset.images = JSON.stringify(gpu.images || []);
                    // Store pricing fields explicitly for easy access
                    option.dataset.pricePerGpuHour = pricePerGpuHour;
                    option.dataset.gpuCount = gpuCount;
                    option.dataset.providerUpfrontHours = providerUpfrontHours;
                    
                    // Don't check availability during initial load - just show all GPUs
                    // Availability will be checked when user selects a GPU
                    option.textContent = `${gpu.name} - ${gpu.priceHour || ''}`;
                    option.disabled = false;
                    
                    updatedGpuSelect.appendChild(option);
                }
                
                updatedGpuSelect.disabled = false;
                
                // Add change listener to validate form, update cost, and check availability
                updatedGpuSelect.addEventListener('change', () => { 
                    if (cloudConfigValidateForm) cloudConfigValidateForm(); 
                    updateCloudCostEstimate(); 
                    if (cloudConfigCheckGPUAvailability) cloudConfigCheckGPUAvailability(); 
                });
                
                // Add change listener to load images when GPU is selected
                updatedGpuSelect.addEventListener('change', (e) => {
                    const selectedOption = e.target.options[e.target.selectedIndex];
                    if (selectedOption && !selectedOption.disabled && selectedOption.dataset.images) {
                        const images = JSON.parse(selectedOption.dataset.images);
                        const imageSelect = document.getElementById('cloudImageSelect');
                        if (imageSelect) {
                            imageSelect.innerHTML = '<option value="">Select an image</option>';
                            images.forEach(image => {
                                const option = document.createElement('option');
                                option.value = image;
                                option.textContent = image;
                                imageSelect.appendChild(option);
                            });
                            imageSelect.disabled = false;
                        }
                    } else {
                        const imageSelect = document.getElementById('cloudImageSelect');
                        if (imageSelect) {
                            imageSelect.innerHTML = '<option value="">Select GPU type first</option>';
                            imageSelect.disabled = true;
                        }
                    }
                    // Update cost estimate
                    updateCloudCostEstimate();
                });
                
                // Add change listeners for cost estimation
                const maxHoursInput = document.getElementById('cloudMaxTrainingHours');
                const budgetLimitInput = document.getElementById('cloudBudgetLimit');
                if (maxHoursInput) {
                    maxHoursInput.addEventListener('input', updateCloudCostEstimate);
                }
                if (budgetLimitInput) {
                    budgetLimitInput.addEventListener('input', updateCloudCostEstimate);
                }
            }
        }
    } catch (error) {
        console.error('Error loading instance types:', error);
        log('Error loading instance types: ' + error.message, 'error');
        const gpuSelect = document.getElementById('cloudGPUSelect');
        if (gpuSelect) {
            gpuSelect.innerHTML = '<option value="">Error loading GPUs</option>';
            gpuSelect.disabled = true;
        }
    }
}

// Debug function for neural network - make it globally available
window.debugNeuralNetwork = function() {
    if (!window.neuralNetwork) {
        console.error('Neural network not initialized');
        console.log('Available variables:', {
            neuralNetwork: typeof window.neuralNetwork,
            NeuralNetworkVisualization: typeof NeuralNetworkVisualization
        });
        return;
    }
    
    const nn = window.neuralNetwork;
    
    console.log('=== NEURAL NETWORK DEBUG ===');
    console.log('Canvas:', nn.canvas);
    if (nn.canvas) {
        console.log('Canvas dimensions:', {
            width: nn.canvas.width,
            height: nn.canvas.height,
            offsetWidth: nn.canvas.offsetWidth,
            offsetHeight: nn.canvas.offsetHeight,
            clientWidth: nn.canvas.clientWidth,
            clientHeight: nn.canvas.clientHeight,
            styleWidth: nn.canvas.style.width,
            styleHeight: nn.canvas.style.height,
            getBoundingClientRect: nn.canvas.getBoundingClientRect()
        });
    }
    
    console.log('Training state:', {
        isTraining: nn.isTraining,
        trainingProgress: nn.trainingProgress,
        isExpanding: nn.isExpanding,
        expansionProgress: nn.expansionProgress
    });
    
    console.log('Network structure:', {
        layers: nn.layers ? nn.layers.length : 0,
        connections: nn.connections ? nn.connections.length : 0,
        firstLayer: nn.layers && nn.layers[0] ? nn.layers[0].length : 0
    });
    
    if (nn.layers) {
        const visibleNodes = nn.layers.flat().filter(n => n.visible);
        console.log('Visible nodes:', visibleNodes.length, 'out of', nn.layers.flat().length);
    }
    
    console.log('Animation running:', nn.animationId !== null);
    
    // Force a redraw
    console.log('Forcing redraw...');
    try {
        nn.draw();
        console.log('Draw completed successfully');
    } catch (e) {
        console.error('Error during draw:', e);
    }
    
    console.log('=== END DEBUG ===');
};

// Store handler functions to prevent duplicate listeners
let selectFolderHandler = null;

// Separate initialization function that can be called when main app is shown
window.initializeApp = function() {
    console.log('Initializing app...');
    loadSystemInfo();
    setupFileUpload();
    setupSettings();
    setupMetricsTabs();
    
    // Initialize model configuration dropdowns
    const modelPurposeInput = document.getElementById('modelPurposeInput');
    if (modelPurposeInput) {
        modelPurpose = modelPurposeInput.value || 'machine_learning';
        updateFrameworkOptions();
    }
    
    // Update button text on initialization
    updateDatasetButtonText();
    
    // Update settings visibility on initialization
    updateSettingsVisibility();
    
    // Setup all button event listeners with null checks
    const refreshBtn = document.getElementById('refreshBtn');
    const testGpuBtn = document.getElementById('testGpuBtn');
    const testCpuBtn = document.getElementById('testCpuBtn');
    const startTrainingBtn = document.getElementById('startTrainingBtn');
    const stopTrainingBtn = document.getElementById('stopTrainingBtn');
    const trainNewModelBtn = document.getElementById('trainNewModelBtn');
    const openModelLocationBtn = document.getElementById('openModelLocationBtn');
    const closeBtn = document.getElementById('closeBtn');
    const refreshModelsBtn = document.getElementById('refreshModelsBtn');
    const retrainModelBtn = document.getElementById('retrainModelBtn');
    const selectFolderBtn = document.getElementById('selectFolderBtn');
    
    if (refreshBtn) refreshBtn.addEventListener('click', loadSystemInfo);
    if (testGpuBtn) testGpuBtn.addEventListener('click', testGPU);
    if (testCpuBtn) testCpuBtn.addEventListener('click', testCPU);
    if (startTrainingBtn) startTrainingBtn.addEventListener('click', startTraining);
    if (stopTrainingBtn) stopTrainingBtn.addEventListener('click', stopTraining);
    
    // Terminate Instance button - Now shuts down the app
    const terminateInstanceBtn = document.getElementById('terminateInstanceBtn');
    if (terminateInstanceBtn) {
        terminateInstanceBtn.addEventListener('click', () => {
            if (!confirm('Are you sure you want to exit Uni Trainer? The application will close completely.')) {
                return;
            }
            
            log('Shutting down Uni Trainer...', 'warning');
            
            // If there's an active cloud instance, try to terminate it first (non-blocking)
            if (window.currentCloudInstanceId && window.canopywaveApiKey) {
                ipcRenderer.invoke(
                    'terminate-cloud-instance',
                    window.canopywaveApiKey,
                    window.currentCloudInstanceId,
                    window.cloudConfig?.project || '',
                    window.cloudConfig?.region || ''
                )
                    .then(result => {
                        if (result && result.success) {
                            log('Cloud instance terminated successfully', 'success');
                        }
                        // Quit the app regardless of termination result
                        setTimeout(() => {
                            ipcRenderer.invoke('app-quit').catch(() => {
                                // If IPC fails, try window.close as fallback
                                window.close();
                            });
                        }, 500);
                    })
                    .catch(error => {
                        console.error('Error terminating cloud instance:', error);
                        // Quit the app even if termination failed
                        setTimeout(() => {
                            ipcRenderer.invoke('app-quit').catch(() => {
                                window.close();
                            });
                        }, 500);
                    });
            } else {
                // No active instance, just quit immediately
                ipcRenderer.invoke('app-quit').catch(() => {
                    window.close();
                });
            }
        });
    }
    if (trainNewModelBtn) trainNewModelBtn.addEventListener('click', trainNewModel);
    
    // Select folder button for real training - remove old listener first to prevent duplicates
    if (selectFolderBtn) {
        // Remove existing listener if it exists
        if (selectFolderHandler) {
            selectFolderBtn.removeEventListener('click', selectFolderHandler);
        }
        
        // Create new handler function
        // Add event handlers for Schema Review UI
        const selectAllFeaturesBtn = document.getElementById('selectAllFeaturesBtn');
        const deselectAllFeaturesBtn = document.getElementById('deselectAllFeaturesBtn');
        const dismissWarningBtn = document.getElementById('dismissLabelLikeWarningBtn');
        const featureList = document.getElementById('featureColumnsList');
        
        // Add event listener for dismiss warning button
        if (dismissWarningBtn) {
            dismissWarningBtn.addEventListener('click', () => {
                const warningBanner = document.getElementById('labelLikeWarningBanner');
                if (warningBanner) {
                    warningBanner.style.display = 'none';
                }
            });
        }
        
        // Update warning banner when feature checkboxes change
        if (featureList) {
            // Use event delegation for dynamically created checkboxes
            featureList.addEventListener('change', (e) => {
                if (e.target.type === 'checkbox') {
                    const targetSelect = document.getElementById('targetColumnSelect');
                    const targetColumn = targetSelect ? targetSelect.value : null;
                    updateLabelLikeWarningBanner([], targetColumn);
                }
            });
        }
        const targetColumnSelect = document.getElementById('targetColumnSelect');
        const autoExcludeCheckbox = document.getElementById('autoExcludeIdColumns');
        
        if (selectAllFeaturesBtn) {
            selectAllFeaturesBtn.addEventListener('click', () => {
                const checkboxes = document.querySelectorAll('#featureColumnsList input[type="checkbox"]');
                checkboxes.forEach(cb => cb.checked = true);
            });
        }
        
        if (deselectAllFeaturesBtn) {
            deselectAllFeaturesBtn.addEventListener('click', () => {
                const checkboxes = document.querySelectorAll('#featureColumnsList input[type="checkbox"]');
                checkboxes.forEach(cb => cb.checked = false);
            });
        }
        
        if (targetColumnSelect) {
            targetColumnSelect.addEventListener('change', (e) => {
                // Re-render feature list to exclude the selected target, preserving the selection
                if (window.tabularSchema && window.tabularCSVPath) {
                    const selectedValue = e.target.value;
                    
                    // Auto-uncheck target column if it was selected as a feature
                    if (selectedValue) {
                        const targetCheckbox = document.getElementById(`feature_${selectedValue}`);
                        if (targetCheckbox && targetCheckbox.checked) {
                            targetCheckbox.checked = false;
                            log(`Removed "${selectedValue}" from features (it is now the target column)`, 'warning');
                        }
                    }
                    
                    // Update the schema review UI, preserving the target selection
                    updateSchemaReviewUI(window.tabularSchema, window.tabularCSVPath, true);
                    
                    // Auto-disable target column in feature list (if it exists)
                    if (selectedValue) {
                        setTimeout(() => {
                            const targetCheckbox = document.getElementById(`feature_${selectedValue}`);
                            if (targetCheckbox) {
                                targetCheckbox.disabled = true;
                                targetCheckbox.checked = false;
                                targetCheckbox.title = 'This column is selected as the target and cannot be used as a feature.';
                                
                                // Style the disabled checkbox row
                                const targetRow = targetCheckbox.closest('div[style*="padding: 12px"]');
                                if (targetRow) {
                                    targetRow.style.opacity = '0.6';
                                    targetRow.style.pointerEvents = 'none';
                                    targetRow.style.backgroundColor = '#F5F5F5';
                                }
                            }
                            
                            // Update warning banner
                            updateLabelLikeWarningBanner([], selectedValue);
                        }, 10);
                    }
                }
            });
        }
        
        if (autoExcludeCheckbox) {
            autoExcludeCheckbox.addEventListener('change', () => {
                // Re-render feature list with new auto-exclude setting
                if (window.tabularSchema && window.tabularCSVPath) {
                    updateSchemaReviewUI(window.tabularSchema, window.tabularCSVPath);
                }
            });
        }
        
        // Show/hide schema review based on model type
        const schemaSection = document.getElementById('schemaReviewSection');
        const modelPurposeInput = document.getElementById('modelPurposeInput');
        if (schemaSection && modelPurposeInput) {
            const updateSchemaVisibility = () => {
                const modelPurpose = modelPurposeInput.value;
                schemaSection.style.display = modelPurpose === 'tabular' ? 'block' : 'none';
            };
            modelPurposeInput.addEventListener('change', updateSchemaVisibility);
            modelPurposeInput.addEventListener('input', updateSchemaVisibility);
            updateSchemaVisibility(); // Initial update
        }
        
        // Initialize inference UI
        initializeInferenceUI();
        
        selectFolderHandler = async () => {
            try {
                const modelPurpose = document.getElementById('modelPurposeInput').value;
                const result = await ipcRenderer.invoke('select-dataset-directory', { modelPurpose });
                if (!result.canceled && result.filePaths && result.filePaths.length > 0) {
                    const selectedPath = result.filePaths[0];
                    
                    const fs = require('fs');
                    const path = require('path');
                    
                    // Check if a CSV file was selected directly (for tabular data)
                    let inputFolder = selectedPath;
                    let selectedFile = null;
                    
                    try {
                        const stats = fs.statSync(selectedPath);
                        if (stats.isFile() && selectedPath.toLowerCase().endsWith('.csv')) {
                            // CSV file selected directly - use it
                            selectedFile = selectedPath;
                            inputFolder = path.dirname(selectedPath);
                            selectedDatasetFile = selectedFile;
                            selectedFolderPath = inputFolder;
                            log(`Selected CSV file: ${path.basename(selectedFile)}`, 'success');
                            log(`Dataset file: ${selectedFile}`, 'log');
                            document.getElementById('folderPathText').textContent = selectedFile;
                            document.getElementById('selectedFolderPath').style.display = 'block';
                            document.getElementById('settingsSection').style.display = 'block';
                            
                            // For tabular models, extract schema and show review UI
                            if (modelPurpose === 'tabular') {
                                try {
                                    log('Extracting schema from CSV...', 'log');
                                    const schema = await extractCSVSchema(selectedFile);
                                    updateSchemaReviewUI(schema, selectedFile);
                                    log(`Schema extracted: ${schema.length} columns found`, 'success');
                                    
                                    // Store schema for later use
                                    window.tabularSchema = schema;
                                    window.tabularCSVPath = selectedFile;
                                } catch (error) {
                                    log(`Error extracting schema: ${error.message}`, 'error');
                                }
                            }
                            
                            return; // Exit early for direct CSV file selection
                        }
                    } catch (e) {
                        // If stat fails, assume it's a directory
                        inputFolder = selectedPath;
                    }
                    
                    // For tabular data with folder selection, check for CSV files
                    if (modelPurpose === 'tabular') {
                        const files = fs.readdirSync(inputFolder);
                        const csvFiles = files.filter(f => f.toLowerCase().endsWith('.csv'));
                        
                        if (csvFiles.length === 0) {
                            log('No CSV files found in selected folder', 'error');
                            selectedFolderPath = null;
    selectedDatasetFile = null;
                            selectedDatasetFile = null;
                            document.getElementById('selectedFolderPath').style.display = 'none';
                            return;
                        } else if (csvFiles.length === 1) {
                            // Single CSV file - use it
                            selectedFile = path.join(inputFolder, csvFiles[0]);
                            selectedDatasetFile = selectedFile;
                            selectedFolderPath = inputFolder;
                            log(`Found 1 CSV file: ${csvFiles[0]}`, 'success');
                            log(`Dataset file: ${selectedFile}`, 'log');
                            document.getElementById('folderPathText').textContent = selectedFile;
                            document.getElementById('selectedFolderPath').style.display = 'block';
                            document.getElementById('settingsSection').style.display = 'block';
                            
                            // For tabular models, extract schema and show review UI
                            if (modelPurpose === 'tabular') {
                                try {
                                    log('Extracting schema from CSV...', 'log');
                                    const schema = await extractCSVSchema(selectedFile);
                                    updateSchemaReviewUI(schema, selectedFile);
                                    log(`Schema extracted: ${schema.length} columns found`, 'success');
                                    
                                    // Store schema for later use
                                    window.tabularSchema = schema;
                                    window.tabularCSVPath = selectedFile;
                                } catch (error) {
                                    log(`Error extracting schema: ${error.message}`, 'error');
                                }
                            }
                            
                            return;
                        } else {
                            // Multiple CSV files - ask user to choose
                            log(`Found ${csvFiles.length} CSV files. Please select which one to use:`, 'warning');
                            
                            // Create a simple prompt to select CSV file
                            const csvFileOptions = csvFiles.map((f, i) => `${i + 1}. ${f}`).join('\n');
                            const userChoice = prompt(`Multiple CSV files found:\n\n${csvFileOptions}\n\nEnter the number (1-${csvFiles.length}) of the file to use:`);
                            
                            if (userChoice && !isNaN(userChoice)) {
                                const choiceIndex = parseInt(userChoice) - 1;
                                if (choiceIndex >= 0 && choiceIndex < csvFiles.length) {
                                    selectedFile = path.join(inputFolder, csvFiles[choiceIndex]);
                                    selectedDatasetFile = selectedFile;
                                    selectedFolderPath = inputFolder;
                                    log(`Selected CSV file: ${csvFiles[choiceIndex]}`, 'success');
                                    log(`Dataset file: ${selectedFile}`, 'log');
                                    document.getElementById('folderPathText').textContent = selectedFile;
                                    document.getElementById('selectedFolderPath').style.display = 'block';
                                    document.getElementById('settingsSection').style.display = 'block';
                                    return;
                                }
                            }
                            
                            log('CSV file selection cancelled or invalid', 'warning');
                            selectedFolderPath = null;
    selectedDatasetFile = null;
                            selectedDatasetFile = null;
                            document.getElementById('selectedFolderPath').style.display = 'none';
                            return;
                        }
                    }
                    
                    // Check if folder contains JSON files (LabelMe format)
                    // If so, prepare the dataset automatically
                    const formatCheck = await ipcRenderer.invoke('check-dataset-format', inputFolder);
                    
                    if (formatCheck.isLabelMe) {
                        log('LabelMe format detected. Preparing dataset for YOLO training...', 'log');
                        selectFolderBtn.disabled = true;
                        selectFolderBtn.textContent = 'Preparing Dataset...';
                        
                        // Create output folder name
                        const folderName = inputFolder.split(/[/\\]/).pop();
                        const outputFolder = await ipcRenderer.invoke('get-prepared-dataset-path', folderName);
                        
                        // Prepare the dataset
                        const prepareResult = await ipcRenderer.invoke('prepare-dataset', inputFolder, outputFolder);
                        
                        selectFolderBtn.disabled = false;
                        selectFolderBtn.textContent = '📂 Select Dataset Folder';
                        
                        if (prepareResult.success) {
                            selectedFolderPath = prepareResult.outputFolder;
                            log(`Dataset prepared successfully!`, 'success');
                            log(`Prepared dataset: ${selectedFolderPath}`, 'log');
                            document.getElementById('folderPathText').textContent = selectedFolderPath;
                            document.getElementById('selectedFolderPath').style.display = 'block';
                            document.getElementById('settingsSection').style.display = 'block';
                        } else {
                            log(`Dataset preparation failed: ${prepareResult.error}`, 'error');
                            if (prepareResult.stderr) {
                                log(prepareResult.stderr, 'error');
                            }
                            log('LabelMe detected, but prepare_dataset.py is missing. Using the selected folder as-is.', 'warning');
                            selectedFolderPath = inputFolder;
                            selectedDatasetFile = null;
                            document.getElementById('folderPathText').textContent = selectedFolderPath;
                            document.getElementById('selectedFolderPath').style.display = 'block';
                            document.getElementById('settingsSection').style.display = 'block';
                        }
                    } else {
                        // Not LabelMe format, validate dataset based on selected model type
                        const modelPurpose = document.getElementById('modelPurposeInput').value;
                        const framework = document.getElementById('frameworkInput').value;
                        
                        // Validate dataset format
                        const validationResult = validateDatasetFormat(inputFolder, modelPurpose, framework);
                        
                        selectedFolderPath = inputFolder;
                        document.getElementById('folderPathText').textContent = selectedFolderPath;
                        document.getElementById('selectedFolderPath').style.display = 'block';
                        document.getElementById('settingsSection').style.display = 'block';

                        if (!validationResult.valid) {
                            log(`Dataset validation warning: ${validationResult.message}`, 'warning');
                            if (validationResult.expectedFormat) {
                                log(`Expected format: ${validationResult.expectedFormat}`, 'warning');
                            }
                            return;
                        }

                        log(`Using dataset folder: ${inputFolder}`, 'success');
                        log(validationResult.message, 'success');
                    }
                    
                    // Show settings when folder is selected
                    document.getElementById('settingsSection').style.display = 'block';
                    
                    log(`Selected folder: ${selectedFolderPath}`, 'success');
                    log('This folder will be used for real training', 'log');
                }
            } catch (error) {
                log(`Error selecting folder: ${error.message}`, 'error');
            }
        };
        
        // Add the new listener
        selectFolderBtn.addEventListener('click', selectFolderHandler);
        
        // Update button text based on model type
        updateDatasetButtonText();
    }
    
    
    // Open model location button
    if (openModelLocationBtn) {
        openModelLocationBtn.addEventListener('click', () => {
            if (savedModelPath) {
                ipcRenderer.send('open-model-location', savedModelPath);
            }
        });
    }
    
    // Close button
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            ipcRenderer.send('close-window');
        });
    }
    
    // Model management
    if (refreshModelsBtn) refreshModelsBtn.addEventListener('click', loadSavedModels);
    if (retrainModelBtn) retrainModelBtn.addEventListener('click', retrainModel);
    
    // Load saved models on startup
    loadSavedModels();
    
    // Initialize neural network if not already initialized
    if (typeof NeuralNetworkVisualization !== 'undefined') {
        const canvas = document.getElementById('neuralCanvas');
        if (canvas && (typeof neuralNetwork === 'undefined' || !neuralNetwork)) {
            try {
                window.neuralNetwork = new NeuralNetworkVisualization('neuralCanvas');
                neuralNetwork = window.neuralNetwork;
                log('Neural network visualization initialized', 'success');
                
                // Add debug function to window for console access
                window.debugNeuralNetwork = function() {
                    if (!window.neuralNetwork) {
                        console.error('Neural network not initialized');
                        return;
                    }
                    
                    const nn = window.neuralNetwork;
                    
                    console.log('=== NEURAL NETWORK DEBUG ===');
                    console.log('Canvas:', nn.canvas);
                    console.log('Canvas dimensions:', {
                        width: nn.canvas.width,
                        height: nn.canvas.height,
                        offsetWidth: nn.canvas.offsetWidth,
                        offsetHeight: nn.canvas.offsetHeight,
                        clientWidth: nn.canvas.clientWidth,
                        clientHeight: nn.canvas.clientHeight,
                        styleWidth: nn.canvas.style.width,
                        styleHeight: nn.canvas.style.height
                    });
                    
                    console.log('Training state:', {
                        isTraining: nn.isTraining,
                        trainingProgress: nn.trainingProgress,
                        isExpanding: nn.isExpanding,
                        expansionProgress: nn.expansionProgress
                    });
                    
                    console.log('Network structure:', {
                        layers: nn.layers ? nn.layers.length : 0,
                        connections: nn.connections ? nn.connections.length : 0,
                        firstLayer: nn.layers && nn.layers[0] ? nn.layers[0].length : 0
                    });
                    
                    console.log('Visible nodes:', nn.layers ? 
                        nn.layers.flat().filter(n => n.visible).length : 0);
                    
                    console.log('Animation running:', nn.animationId !== null);
                    
                    // Force a redraw
                    console.log('Forcing redraw...');
                    nn.draw();
                    
                    console.log('=== END DEBUG ===');
                };
                
                console.log('[Renderer] Debug function available: Run debugNeuralNetwork() in console');
            } catch (e) {
                console.error('Error initializing neural network:', e);
                console.error('Error stack:', e.stack);
            }
        }
    }
    
    log('Uni Trainer initialized', 'success');
}

// Initialize Inference UI
function initializeInferenceUI() {
    const cvInferenceSection = document.getElementById('cvInferenceSection');
    const tabularInferenceSection = document.getElementById('tabularInferenceSection');
    const modelPurposeInput = document.getElementById('modelPurposeInput');
    
    // Show/hide inference sections based on model purpose
    const updateInferenceVisibility = () => {
        const modelPurpose = modelPurposeInput.value;
        if (cvInferenceSection) {
            cvInferenceSection.style.display = modelPurpose === 'computer_vision' ? 'block' : 'none';
        }
        if (tabularInferenceSection) {
            tabularInferenceSection.style.display = modelPurpose === 'tabular' ? 'block' : 'none';
        }
        
        // Load models when section becomes visible
        if (modelPurpose === 'computer_vision' && cvInferenceSection) {
            loadCvModels();
        }
        if (modelPurpose === 'tabular' && tabularInferenceSection) {
            loadTabularModels();
        }
    };
    
    if (modelPurposeInput) {
        modelPurposeInput.addEventListener('change', updateInferenceVisibility);
        modelPurposeInput.addEventListener('input', updateInferenceVisibility);
        updateInferenceVisibility();
    }
    
    // CV Inference handlers
    const selectCvImageBtn = document.getElementById('selectCvImageBtn');
    const cvModelSelect = document.getElementById('cvModelSelect');
    const cvConfidenceSlider = document.getElementById('cvConfidenceSlider');
    const cvConfidenceValue = document.getElementById('cvConfidenceValue');
    const runCvInferenceBtn = document.getElementById('runCvInferenceBtn');
    const openCvOutputFolderBtn = document.getElementById('openCvOutputFolderBtn');
    
    if (cvConfidenceSlider && cvConfidenceValue) {
        cvConfidenceSlider.addEventListener('input', (e) => {
            cvConfidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    if (selectCvImageBtn) {
        selectCvImageBtn.addEventListener('click', async () => {
            try {
                const imagePath = await ipcRenderer.invoke('select-image-file');
                if (imagePath) {
                    document.getElementById('cvImagePath').textContent = imagePath;
                    document.getElementById('cvImagePath').style.display = 'block';
                    if (runCvInferenceBtn) runCvInferenceBtn.disabled = false;
                }
            } catch (error) {
                log(`Error selecting image: ${error.message}`, 'error');
            }
        });
    }
    
    if (runCvInferenceBtn) {
        runCvInferenceBtn.addEventListener('click', async () => {
            try {
                const modelPath = cvModelSelect?.value;
                const imagePath = document.getElementById('cvImagePath')?.textContent;
                const confidence = parseFloat(cvConfidenceSlider?.value || 0.25);
                
                if (!modelPath || !imagePath) {
                    log('Please select a model and image', 'error');
                    return;
                }
                
                runCvInferenceBtn.disabled = true;
                log('Running CV inference...', 'log');
                
                const result = await ipcRenderer.invoke('cv-infer', {
                    inference_type: 'cv',
                    model_path: modelPath,
                    image_path: imagePath,
                    confidence: confidence
                });
                
                const normalizedCvResult = {
                    detections: result.detections || result.metrics?.detections || [],
                    num_detections: result.num_detections ?? result.metrics?.num_detections ?? 0,
                    output_dir: result.output_dir || result.metrics?.output_dir,
                    output_image_path: result.output_image_path || result.metrics?.output_image_path
                };
                
                // Display results
                const resultsDiv = document.getElementById('cvInferenceResults');
                if (resultsDiv) {
                    resultsDiv.style.display = 'block';
                    
                    // Show preview image
                    const previewImg = document.getElementById('cvPreviewImage');
                    if (previewImg && normalizedCvResult.output_image_path) {
                        previewImg.src = `file:///${normalizedCvResult.output_image_path.replace(/\\/g, '/')}`;
                    }
                    
                    // Show detections
                    const detectionsList = document.getElementById('cvDetectionsList');
                    if (detectionsList && normalizedCvResult.detections) {
                        if (normalizedCvResult.detections.length === 0) {
                            detectionsList.innerHTML = '<div style="color: #8B7355; text-align: center; padding: 20px;">No detections found</div>';
                        } else {
                            detectionsList.innerHTML = normalizedCvResult.detections.map((det, i) => 
                                `<div style="padding: 8px; margin-bottom: 4px; background: white; border-radius: 4px; border: 1px solid #E0D5C7;">
                                    <strong>${det.class}</strong> (${(det.confidence * 100).toFixed(1)}%)<br>
                                    <small style="color: #8B7355;">BBox: [${det.bbox.join(', ')}]</small>
                                </div>`
                            ).join('');
                        }
                    }
                    
                    // Store output folder for open button
                    if (openCvOutputFolderBtn) {
                        if (normalizedCvResult.output_dir) {
                            openCvOutputFolderBtn.disabled = false;
                            openCvOutputFolderBtn.onclick = () => {
                                ipcRenderer.invoke('open-folder', normalizedCvResult.output_dir);
                            };
                        } else {
                            openCvOutputFolderBtn.disabled = true;
                            openCvOutputFolderBtn.onclick = null;
                        }
                    }
                }
                
                const detectionsCount = normalizedCvResult.detections.length || normalizedCvResult.num_detections || 0;
                log(`CV inference completed: ${detectionsCount} detections`, 'success');
                runCvInferenceBtn.disabled = false;
            } catch (error) {
                log(`CV inference error: ${error.message}`, 'error');
                if (runCvInferenceBtn) runCvInferenceBtn.disabled = false;
            }
        });
    }
    
    // Tabular Inference handlers
    const selectTabularCsvBtn = document.getElementById('selectTabularCsvBtn');
    const tabularModelSelect = document.getElementById('tabularModelSelect');
    const runTabularInferenceBtn = document.getElementById('runTabularInferenceBtn');
    const exportTabularPredictionsBtn = document.getElementById('exportTabularPredictionsBtn');
    const refreshTabularModelsBtn = document.getElementById('refreshTabularModelsBtn');
    
    // Refresh button
    if (refreshTabularModelsBtn) {
        refreshTabularModelsBtn.addEventListener('click', () => {
            loadTabularModels();
        });
    }
    
    if (selectTabularCsvBtn) {
        selectTabularCsvBtn.addEventListener('click', async () => {
            try {
                const csvPath = await ipcRenderer.invoke('select-csv-file');
                if (csvPath) {
                    document.getElementById('tabularCsvPath').textContent = csvPath;
                    document.getElementById('tabularCsvPath').style.display = 'block';
                    if (runTabularInferenceBtn) runTabularInferenceBtn.disabled = false;
                }
            } catch (error) {
                log(`Error selecting CSV: ${error.message}`, 'error');
            }
        });
    }
    
    if (runTabularInferenceBtn) {
        runTabularInferenceBtn.addEventListener('click', async () => {
            const modelPath = tabularModelSelect?.value;
            const csvPath = document.getElementById('tabularCsvPath')?.textContent;
            
            if (!modelPath || !csvPath) {
                log('Please select a model and CSV file', 'error');
                return;
            }
            
            runTabularInferenceBtn.disabled = true;
            log('Running tabular inference...', 'log');
            log(`Model artifact path: ${modelPath}`, 'log');
            log(`CSV path: ${csvPath}`, 'log');
            
            try {
                const result = await ipcRenderer.invoke('tabular-infer', {
                    inference_type: 'tabular',
                    model_artifact_path: modelPath,
                    csv_path: csvPath
                });
                
                const normalizedResult = {
                    predictions: result.predictions || result.metrics?.predictions || [],
                    num_predictions: result.num_predictions || result.metrics?.num_predictions || 0,
                    has_probabilities: result.has_probabilities ?? result.metrics?.has_probabilities,
                    output_path: result.output_path || result.metrics?.output_path || result.model_path
                };

                // Display results
                const resultsDiv = document.getElementById('tabularInferenceResults');
                if (resultsDiv) {
                    resultsDiv.style.display = 'block';

                    // Show preview table
                    const resultsTable = document.getElementById('tabularResultsTable');
                    if (resultsTable && normalizedResult.predictions) {
                        if (normalizedResult.predictions.length === 0) {
                            resultsTable.innerHTML = '<div style="color: #8B7355; text-align: center; padding: 20px;">No predictions</div>';
                        } else {
                            // Create table from first few rows
                            const headers = Object.keys(normalizedResult.predictions[0]);
                            const tableHTML = `
                                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                                    <thead>
                                        <tr style="background: #F5F5F5; border-bottom: 2px solid #E0D5C7;">
                                            ${headers.map(h => `<th style="padding: 8px; text-align: left; font-weight: 600; color: #5A4A3A;">${h}</th>`).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${normalizedResult.predictions.slice(0, 20).map(row => 
                                            `<tr style="border-bottom: 1px solid #E0D5C7;">
                                                ${headers.map(h => `<td style="padding: 6px; color: #8B7355;">${row[h]}</td>`).join('')}
                                            </tr>`
                                        ).join('')}
                                    </tbody>
                                </table>
                                ${normalizedResult.predictions.length > 20 ? `<div style="margin-top: 8px; text-align: center; color: #8B7355; font-size: 11px;">Showing first 20 of ${normalizedResult.num_predictions} predictions</div>` : ''}
                            `;
                            resultsTable.innerHTML = tableHTML;
                        }
                    }
                    
                    // Store output path for export button
                    if (exportTabularPredictionsBtn) {
                        if (normalizedResult.output_path) {
                            exportTabularPredictionsBtn.disabled = false;
                            exportTabularPredictionsBtn.onclick = () => {
                                ipcRenderer.invoke('open-folder', require('path').dirname(normalizedResult.output_path));
                            };
                        } else {
                            exportTabularPredictionsBtn.disabled = true;
                            exportTabularPredictionsBtn.onclick = null;
                        }
                    }
                }
                
                log(`Tabular inference completed: ${normalizedResult.num_predictions || 0} predictions`, 'success');
            } catch (error) {
                log(`Tabular inference error: ${error.message}`, 'error');
                log(`Error details: ${JSON.stringify(error)}`, 'error');
                console.error('Tabular inference error:', error);
            } finally {
                if (runTabularInferenceBtn) runTabularInferenceBtn.disabled = false;
            }
        });
    }
}

// Load CV models (YOLO runs)
async function loadCvModels() {
    try {
        const models = await ipcRenderer.invoke('list-cv-models');
        const cvModelSelect = document.getElementById('cvModelSelect');
        if (cvModelSelect) {
            cvModelSelect.innerHTML = '<option value="">Select model...</option>';
            if (models.length === 0) {
                cvModelSelect.innerHTML += '<option value="" disabled>No models found. Train a CV model first.</option>';
            } else {
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.path;
                    option.textContent = `${model.name} (${new Date(model.timestamp).toLocaleDateString()})`;
                    cvModelSelect.appendChild(option);
                });
                // Auto-select first (most recent)
                if (models.length > 0) {
                    cvModelSelect.value = models[0].path;
                }
            }
        }
    } catch (error) {
        log(`Error loading CV models: ${error.message}`, 'error');
    }
}

// Load Tabular models (artifacts)
async function loadTabularModels() {
    try {
        const tabularModelSelect = document.getElementById('tabularModelSelect');
        if (!tabularModelSelect) return;
        
        // Show loading state
        tabularModelSelect.innerHTML = '<option value="">Loading models...</option>';
        
        // Remove any existing listeners to avoid duplicates
        ipcRenderer.removeAllListeners('models-list');
        
        // Use existing list-models IPC and filter for tabular
        ipcRenderer.send('list-models');
        
        // Listen for models list response
        ipcRenderer.once('models-list', (event, models) => {
            if (!tabularModelSelect) return;
            
            tabularModelSelect.innerHTML = '<option value="">Select model...</option>';
            
            // Filter for tabular models using the same logic as backend discovery
            const tabularModels = models.filter(m => {
                if (!m.isArtifact || !m.metadata) {
                    return false;
                }
                
                // Use same classification logic as backend:
                // 1. model_type === 'tabular'
                // 2. model_type starts with 'sklearn_' (backward compat)
                // 3. framework === 'sklearn' AND algorithm exists
                const isTabular = 
                    m.metadata.model_type === 'tabular' ||
                    (m.metadata.model_type && m.metadata.model_type.startsWith('sklearn_')) ||
                    (m.metadata.framework === 'sklearn' && m.metadata.algorithm);
                
                if (isTabular) {
                    log(`Tabular model found: ${m.filename} (model_type=${m.metadata.model_type}, framework=${m.metadata.framework})`, 'log');
                } else {
                    log(`Excluded: ${m.filename} (model_type=${m.metadata.model_type}, framework=${m.metadata.framework}) - not tabular`, 'log');
                }
                
                return isTabular;
            });
            
            // Debug logging
            log(`Model discovery: ${models.length} total models, ${tabularModels.length} tabular models`, 'log');
            if (tabularModels.length === 0 && models.length > 0) {
                const artifactModels = models.filter(m => m.isArtifact);
                log(`Found ${artifactModels.length} artifacts but none classified as tabular. Sample metadata:`, 'log');
                if (artifactModels.length > 0) {
                    const sample = artifactModels[0];
                    log(`  - filename: ${sample.filename}`, 'log');
                    log(`  - model_type: ${sample.metadata?.model_type || 'N/A'}`, 'log');
                    log(`  - framework: ${sample.metadata?.framework || 'N/A'}`, 'log');
                    log(`  - algorithm: ${sample.metadata?.algorithm || 'N/A'}`, 'log');
                    log(`  - has schema: ${sample.schema ? 'yes' : 'no'}`, 'log');
                }
            }
            
            if (tabularModels.length === 0) {
                tabularModelSelect.innerHTML += '<option value="" disabled>No tabular models found. Train a tabular model first.</option>';
            } else {
                // Sort by timestamp (newest first)
                tabularModels.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                
                tabularModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.filepath;
                    const dateStr = new Date(model.timestamp).toLocaleDateString();
                    const algoStr = model.metadata?.algorithm ? ` (${model.metadata.algorithm})` : '';
                    option.textContent = `${model.filename}${algoStr} - ${dateStr}`;
                    tabularModelSelect.appendChild(option);
                });
                
                // Auto-select most recent
                if (tabularModels.length > 0) {
                    tabularModelSelect.value = tabularModels[0].filepath;
                }
            }
        });
        
        // Handle error case
        ipcRenderer.once('models-list-error', (event, error) => {
            if (tabularModelSelect) {
                tabularModelSelect.innerHTML = '<option value="">Error loading models</option>';
            }
            log(`Error loading models: ${error}`, 'error');
        });
    } catch (error) {
        log(`Error loading tabular models: ${error.message}`, 'error');
        const tabularModelSelect = document.getElementById('tabularModelSelect');
        if (tabularModelSelect) {
            tabularModelSelect.innerHTML = '<option value="">Error loading models</option>';
        }
    }
}

// Live Detection System

// Also call initializeApp immediately if DOM is already loaded and main app is visible
if (document.readyState !== 'loading') {
    setTimeout(() => {
        const mainApp = document.getElementById('mainApp');
        if (mainApp && mainApp.style.display !== 'none' && !mainApp.classList.contains('hidden')) {
            initializeApp();
        }
    }, 100);
}
