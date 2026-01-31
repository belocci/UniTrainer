// Enhanced Neural Network Visualization with Aesthetic Movements
class NeuralNetworkVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.isTraining = false;
        this.isValidating = false;
        this.layers = [];
        this.connections = [];
        this.activationValues = [];
        this.animationId = null;
        this.trainingProgress = 0;
        this.maxRadius = 5;
        this.hoveredNode = null;
        this.mouseX = 0;
        this.mouseY = 0;
        this.isExpanding = false;
        this.expansionProgress = 0;
        this.heartbeatTime = 0;
        this.lastHeartbeatSound = 0;
        this.audioContext = null;
        this.currentHeartbeatIntensity = 0;
        this.lastUpdateTime = null;
        this.currentLoss = 1.0;
        this.currentAccuracy = 0.0;
        this.learningQuality = 0;
        
        // Validation animation system
        this.validationParticles = [];
        this.validationPulseWaves = [];
        this.validationPathwaySequence = 0;
        this.validationStartTime = 0;
        
        // 3D orbiting neuron for validation
        this.orbitAngle = 0;
        this.orbitRadius = 60;
        this.orbitSpeed = 0.02;
        this.neuron3DScale = 1.0;
        
        // Training parameters
        this.trainingSettings = {
            quality: 100,
            epochs: 10,
            batchSize: 32,
            learningRate: 0.001,
            modelType: 'machine_learning',
            modelPurpose: 'machine_learning',
            framework: '',
            variant: '',
            fileCount: 0,
            dataSize: 0
        };
        this.layerSpecs = null;
        
        // Animation settings
        this.animationSpeed = 1.0;
        this.useSubtleMovements = true;
        this.breathingAmplitude = 0.8;
        this.connectionFlowSpeed = 1.0;
        
        // Constants
        this.layersStartProgress = 0.0;
        this.layerTransitionRange = 0.15;
        
        // Connection reveal animation
        this.connectionRevealEnabled = true;
        this.revealSpeed = 1.0;
        this.connectionRevealDelay = 150;
        this.revealDuration = 1500;
        this.trainingComplete = false;
        this.visualizationComplete = false;
        this.trainingStoppedEarly = false; // Flag to prevent completion when stopped early
        
        // Aesthetic movement parameters
        this.globalTime = 0;
        this.breathingPhase = 0;
        this.sinuousPhase = 0;
        this.pulsePhase = 0;
        
        this.initHeartbeatAudio();
        this.init();
        this.animate();
        this.setupMouseHandlers();
    }
    
    // Enhanced easing functions for organic movements
    easingFunctions = {
        easeOutQuad: (t) => t * (2 - t),
        easeInOutCubic: (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
        easeOutBack: (t) => {
            const c1 = 1.70158;
            const c3 = c1 + 1;
            return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
        },
        easeOutElastic: (t) => {
            const c4 = (2 * Math.PI) / 3;
            return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
        },
        easeInOutQuart: (t) => t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2,
        easeInOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,
        easeOutExpo: (t) => t === 1 ? 1 : 1 - Math.pow(2, -10 * t)
    };
    
    // Set aesthetic parameters
    setAestheticParameters(params) {
        if (params.breathingAmplitude !== undefined) {
            this.breathingAmplitude = Math.max(0.1, Math.min(3.0, params.breathingAmplitude));
        }
        if (params.connectionFlowSpeed !== undefined) {
            this.connectionFlowSpeed = Math.max(0.1, Math.min(3.0, params.connectionFlowSpeed));
        }
        if (params.useSubtleMovements !== undefined) {
            this.useSubtleMovements = params.useSubtleMovements;
        }
    }

    initHeartbeatAudio() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.log('Web Audio API not supported');
        }
    }

    playHeartbeatSound() {
        if (!this.audioContext || !this.isTraining) return;
        
        const now = this.audioContext.currentTime;
        const currentTime = Date.now();
        
        if (currentTime - this.lastHeartbeatSound < 600) return;
        
        // Softer heartbeat sound for subtlety
        this.createHeartbeatPulse(now, 0.08);
        this.createHeartbeatPulse(now + 0.15, 0.06);
        
        this.lastHeartbeatSound = currentTime;
    }

    createHeartbeatPulse(startTime, volume) {
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.frequency.setValueAtTime(45, startTime);
        oscillator.frequency.exponentialRampToValueAtTime(35, startTime + 0.1);
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0, startTime);
        gainNode.gain.linearRampToValueAtTime(volume, startTime + 0.03);
        gainNode.gain.exponentialRampToValueAtTime(0.001, startTime + 0.1);
        
        oscillator.start(startTime);
        oscillator.stop(startTime + 0.1);
    }

    calculateNetworkArchitecture() {
        const { quality, epochs, batchSize, learningRate, modelType, fileCount, dataSize } = this.trainingSettings;
        
        let baseArchitecture;
        let inputSize, outputSize;
        
        switch(modelType) {
            case 'computer_vision':
                inputSize = Math.max(16, Math.min(32, Math.floor(fileCount / 10) || 16));
                outputSize = 4;
                baseArchitecture = 'wide';
                break;
            case 'natural_language_processing':
                inputSize = Math.max(12, Math.min(24, Math.floor(dataSize / 1000000) || 12));
                outputSize = 6;
                baseArchitecture = 'deep';
                break;
            case 'reinforcement_learning':
                inputSize = 10;
                outputSize = 8;
                baseArchitecture = 'medium';
                break;
            default:
                inputSize = 8;
                outputSize = 6;
                baseArchitecture = 'standard';
        }
        
        const qualityFactor = quality / 100;
        const depthFactor = Math.min(1.5, 0.5 + (epochs / 100));
        const widthFactor = Math.min(1.3, 0.7 + (batchSize / 200));
        const complexityFactor = Math.min(1.2, 0.8 + ((0.01 - learningRate) / 0.01));
        
        const numHiddenLayers = Math.max(2, Math.min(5, Math.floor(2 + qualityFactor * 3 * depthFactor)));
        
        let layerSizes = [inputSize];
        
        if (baseArchitecture === 'deep') {
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
        
        return layerSizes.map((size, i) => {
            if (i === 0 || i === layerSizes.length - 1) return size;
            return Math.max(6, size);
        });
    }
    
    updateTrainingSettings(settings) {
        this.trainingSettings = {
            ...this.trainingSettings,
            ...settings
        };
        if (settings.layerSpecs) {
            this.layerSpecs = settings.layerSpecs;
        }
        this.createNetwork();
    }
    
    getLayerSizesFromRealArchitecture() {
        const { modelPurpose = 'machine_learning', framework = '', variant = '', quality = 50 } = this.trainingSettings;
        
        if (!modelPurpose && !framework) {
            return null;
        }
        
        const qualityValue = Math.max(0, Math.min(100, quality));
        const qualityFactor = 0.1 + (qualityValue / 100) * 0.9;
        
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
        
        const sizes = [];
        
        // Computer Vision Models
        if (modelPurpose === 'computer_vision') {
            if (framework === 'yolo') {
                // YOLO architecture: simpler representation for visualization
                // Input (3 RGB channels) -> Conv layers -> Detection head (4 classes typically)
                const baseChannels = Math.max(2, Math.floor(2 + qualityFactor * 8 * variantMultiplier));
                sizes.push(3);  // Input (RGB)
                sizes.push(Math.max(3, Math.floor(baseChannels * 0.75)));  // First conv
                sizes.push(Math.max(4, Math.floor(baseChannels)));  // Second conv
                sizes.push(Math.max(6, Math.floor(baseChannels * 1.5)));  // Third conv
                sizes.push(Math.max(8, Math.floor(baseChannels * 2)));  // Fourth conv (reduced from *8)
                sizes.push(4);  // Output (detection classes)
            } else if (framework === 'resnet') {
                const baseChannels = Math.max(2, Math.floor(2 + qualityFactor * 126 * variantMultiplier));
                sizes.push(3);
                sizes.push(Math.max(2, Math.floor(baseChannels * 0.5)));
                sizes.push(Math.max(4, Math.floor(baseChannels)));
                sizes.push(Math.max(64, Math.floor(baseChannels * 8)));
                sizes.push(1000);
            } else {
                const baseSize = Math.max(2, Math.floor(2 + qualityFactor * 254 * variantMultiplier));
                sizes.push(3);
                sizes.push(Math.max(2, Math.floor(baseSize * 0.25)));
                sizes.push(Math.max(2, Math.floor(baseSize)));
                sizes.push(10);
            }
        }
        // Natural Language Processing
        else if (modelPurpose === 'natural_language_processing') {
            if (framework === 'transformer' || framework === 'bert') {
                const embedDim = Math.max(8, Math.floor(8 + qualityFactor * 504 * variantMultiplier));
                sizes.push(Math.max(8, embedDim));
                sizes.push(Math.max(2, Math.floor(embedDim * 0.5)));
                sizes.push(2);
            } else if (framework === 'lstm' || framework === 'gru') {
                const hiddenSize = Math.max(2, Math.floor(2 + qualityFactor * 510 * variantMultiplier));
                sizes.push(Math.max(2, Math.floor(2 + qualityFactor * 126)));
                sizes.push(hiddenSize);
                sizes.push(2);
            } else {
                const baseSize = Math.max(2, Math.floor(2 + qualityFactor * 510 * variantMultiplier));
                sizes.push(baseSize);
                sizes.push(Math.max(1, Math.floor(baseSize * 0.5)));
                sizes.push(10);
            }
        }
        // Machine Learning / Other
        else {
            const baseSize = Math.max(2, Math.floor(2 + qualityFactor * 126 * variantMultiplier));
            sizes.push(Math.max(8, Math.floor(baseSize * 0.5)));
            sizes.push(Math.max(6, Math.floor(baseSize)));
            sizes.push(Math.max(4, Math.floor(baseSize * 0.8)));
            sizes.push(Math.max(2, Math.floor(baseSize * 0.6)));
            sizes.push(6);
        }
        
        return sizes.length >= 2 ? sizes : null;
    }
    
    getNodeDescription(layerIndex, totalLayers, nodeIndex, layerSize) {
        let description = '';
        
        if (layerIndex === 0) {
            description = `Input neuron ${nodeIndex + 1}/${layerSize}. Receives raw data features and passes them to the next layer.`;
        } else if (layerIndex === totalLayers - 1) {
            description = `Output neuron ${nodeIndex + 1}/${layerSize}. Produces the final prediction or classification result.`;
        } else {
            description = `Hidden neuron ${nodeIndex + 1}/${layerSize} in Hidden Layer ${layerIndex}. Processes information from previous layer and learns complex patterns.`;
        }
        
        return description;
    }

    setupMouseHandlers() {
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            // Get mouse position relative to canvas
            const clientX = e.clientX - rect.left;
            const clientY = e.clientY - rect.top;
            
            // Scale mouse coordinates to match canvas internal resolution
            // Canvas has fixed resolution (1200x800) but is scaled via CSS
            const scaleX = this.canvas.width / rect.width;
            const scaleY = this.canvas.height / rect.height;
            
            this.mouseX = clientX * scaleX;
            this.mouseY = clientY * scaleY;
            
            this.hoveredNode = this.getNodeAtPosition(this.mouseX, this.mouseY);
            this.updateTooltip();
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.mouseX = -9999;
            this.mouseY = -9999;
            this.hoveredNode = null;
            this.hideTooltip();
        });
    }
    
    getNodeAtPosition(x, y) {
        for (let layer of this.layers) {
            for (let node of layer) {
                if (!node.visible) continue;
                
                const dx = x - node.x;
                const dy = y - node.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const hoverRadius = Math.max(8, node.currentRadius + 10);
                
                if (distance <= hoverRadius) {
                    return node;
                }
            }
        }
        return null;
    }
    
    updateTooltip() {
        let tooltip = document.getElementById('neuralTooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'neuralTooltip';
            tooltip.className = 'neural-tooltip';
            document.body.appendChild(tooltip);
        }
        
        if (this.hoveredNode) {
            const rect = this.canvas.getBoundingClientRect();
            // Scale node coordinates back to screen space for tooltip positioning
            const scaleX = rect.width / this.canvas.width;
            const scaleY = rect.height / this.canvas.height;
            const nodeX = rect.left + this.hoveredNode.x * scaleX;
            const nodeY = rect.top + this.hoveredNode.y * scaleY;
            
            const tooltipWidth = 280;
            const tooltipHeight = 120;
            const offset = 20;
            
            let tooltipX = nodeX + offset;
            let tooltipY = nodeY - tooltipHeight / 2;
            
            if (tooltipX + tooltipWidth > window.innerWidth) {
                tooltipX = nodeX - tooltipWidth - offset;
            }
            
            if (tooltipY + tooltipHeight > window.innerHeight) {
                tooltipY = window.innerHeight - tooltipHeight - 10;
            }
            
            if (tooltipY < 10) {
                tooltipY = 10;
            }
            
            tooltip.style.display = 'block';
            tooltip.style.left = `${tooltipX}px`;
            tooltip.style.top = `${tooltipY}px`;
            
            const node = this.hoveredNode;
            const totalConnections = node.connectionsIn + node.connectionsOut;
            const layerName = node.layerIndex === 0 ? 'Input' : 
                             node.layerIndex === node.totalLayers - 1 ? 'Output' : 
                             `Hidden ${node.layerIndex}`;
            
            tooltip.innerHTML = `
                <div class="tooltip-header">${node.role}</div>
                <div class="tooltip-content">${node.description}</div>
                <div class="tooltip-details">
                    <div class="detail-row">
                        <span class="detail-label">Position:</span>
                        <span class="detail-value">${layerName} Layer ${node.layerIndex + 1}/${node.totalLayers}, Node ${node.nodeIndex + 1}/${node.layerSize}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Connections:</span>
                        <span class="detail-value">${totalConnections} total (${node.connectionsIn} in, ${node.connectionsOut} out)</span>
                    </div>
                    ${node.visible ? `
                    <div class="detail-row">
                        <span class="detail-label">Activation:</span>
                        <span class="detail-value">${(node.activation * 100).toFixed(1)}%</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Movement:</span>
                        <span class="detail-value">${node.breathing ? 'Breathing' : 'Static'}</span>
                    </div>
                    ` : ''}
                </div>
            `;
        } else {
            tooltip.style.display = 'none';
        }
    }
    
    hideTooltip() {
        const tooltip = document.getElementById('neuralTooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }

    init() {
        if (!this.canvas) {
            console.error('[NeuralNetwork] Canvas element not found!');
            return;
        }
        
        // Set fixed canvas resolution for consistent rendering
        const FIXED_WIDTH = 1600;
        const FIXED_HEIGHT = 1000;
        
        // Set actual canvas resolution (for crisp rendering)
        this.canvas.width = FIXED_WIDTH;
        this.canvas.height = FIXED_HEIGHT;
        
        // Set CSS size to fill container while maintaining aspect ratio
        this.resize();
        
        // Debounced resize handler to prevent excessive redraws
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => this.resize(), 100);
        });
        
        this.createNetwork();
    }

    resize() {
        if (!this.canvas) {
            console.error('[NeuralNetwork] resize() called but canvas is null');
            return;
        }
        
        // Fixed resolution - just update CSS display size to fit container
        const FIXED_WIDTH = 1600;
        const FIXED_HEIGHT = 1000;
        
        const container = this.canvas.parentElement;
        if (!container) return;
        
        const containerWidth = container.clientWidth || FIXED_WIDTH;
        const containerHeight = container.clientHeight || FIXED_HEIGHT;
        
        // Calculate scaling to fit container while maintaining aspect ratio
        const scale = Math.min(containerWidth / FIXED_WIDTH, containerHeight / FIXED_HEIGHT);
        const displayWidth = FIXED_WIDTH * scale;
        const displayHeight = FIXED_HEIGHT * scale;
        
        // Update CSS display size (canvas resolution stays fixed)
        this.canvas.style.width = displayWidth + 'px';
        this.canvas.style.height = displayHeight + 'px';
    }

    createNetwork() {
        const width = this.canvas.width || this.canvas.offsetWidth || 800;
        const height = this.canvas.height || this.canvas.offsetHeight || 600;
        
        let layerSpecs = null;
        if (typeof window.getLayerSpecsFromArchitecture === 'function') {
            layerSpecs = window.getLayerSpecsFromArchitecture();
        }
        
        let layerSizes = null;
        if (layerSpecs && layerSpecs.length > 0) {
            layerSizes = layerSpecs.map(spec => {
                return spec.output_units || spec.out_channels || spec.hidden_size || 1;
            });
        }
        
        if (!layerSizes || layerSizes.length < 2) {
            layerSizes = this.getLayerSizesFromRealArchitecture();
        }
        
        if (!layerSizes || layerSizes.length < 2) {
            layerSizes = [8, 12, 10, 8, 6];
        }
        
        layerSizes = layerSizes.map(size => Math.min(50, Math.max(2, size)));
        
        const layerCount = layerSizes.length;
        
        this.layers = [];
        this.connections = [];
        
        // Create layers with aesthetic movement properties
        for (let i = 0; i < layerCount; i++) {
            const layer = [];
            const nodesInLayer = layerSizes[i];
            const layerX = 100 + (i * (width - 200) / (layerCount - 1));
            
            const totalHeight = height * 0.8;
            const spacing = totalHeight / (nodesInLayer + 1);
            const startY = (height - totalHeight) / 2;
            
            for (let j = 0; j < nodesInLayer; j++) {
                const baseY = startY + (j + 1) * spacing;
                
                let role = '';
                let detailedDescription = '';
                
                if (i === 0) {
                    role = 'Input Neuron';
                    detailedDescription = 'Receives raw input features from your training data.';
                } else if (i === layerCount - 1) {
                    role = 'Output Neuron';
                    detailedDescription = 'Produces the final prediction or classification.';
                } else if (i === Math.floor(layerCount / 2)) {
                    role = 'Hidden Neuron (Mid-Layer)';
                    detailedDescription = 'Located in the middle of the network architecture.';
                } else if (i < layerCount / 2) {
                    role = 'Hidden Neuron (Early Layer)';
                    detailedDescription = 'Processes basic features and edge detection.';
                } else {
                    role = 'Hidden Neuron (Late Layer)';
                    detailedDescription = 'Learns complex, high-level patterns and abstractions.';
                }
                
                // Create unique movement patterns for each node
                const breathingSpeed = 0.3 + Math.random() * 0.7;
                const breathingPhase = Math.random() * Math.PI * 2;
                const floatSpeed = 0.1 + Math.random() * 0.3;
                const floatPhase = Math.random() * Math.PI * 2;
                const pulseSpeed = 0.5 + Math.random() * 0.5;
                const pulsePhase = Math.random() * Math.PI * 2;
                
                layer.push({
                    x: layerX,
                    y: baseY,
                    baseX: layerX,
                    baseY: baseY,
                    startX: layerX,
                    startY: baseY,
                    expansionDelay: 0,
                    baseRadius: 5,
                    currentRadius: 0,
                    targetRadius: 0,
                    activation: 0,
                    targetActivation: 0,
                    visible: false,
                    
                    // Aesthetic movement properties
                    breathingAmplitude: 0.5 + Math.random() * 1.0,
                    breathingSpeed: breathingSpeed,
                    breathingPhase: breathingPhase,
                    breathing: true,
                    
                    floatAmplitude: 0.3 + Math.random() * 0.7,
                    floatSpeed: floatSpeed,
                    floatPhase: floatPhase,
                    floatDirectionX: Math.random() * Math.PI * 2,
                    floatDirectionY: Math.random() * Math.PI * 2,
                    
                    pulseAmplitude: 0.1 + Math.random() * 0.2,
                    pulseSpeed: pulseSpeed,
                    pulsePhase: pulsePhase,
                    
                    // Gentle sway properties
                    swayAmplitude: 0.2 + Math.random() * 0.4,
                    swaySpeed: 0.2 + Math.random() * 0.3,
                    swayPhase: Math.random() * Math.PI * 2,
                    
                    // Node metadata
                    layerIndex: i,
                    nodeIndex: j,
                    layerSize: nodesInLayer,
                    totalLayers: layerCount,
                    role: role,
                    description: detailedDescription,
                    connectionsIn: i > 0 ? layerSizes[i - 1] : 0,
                    connectionsOut: i < layerCount - 1 ? layerSizes[i + 1] : 0,
                    
                    inputValue: 0,
                    weightedSum: 0,
                    outputValue: 0,
                    incomingWeights: [],
                    outgoingWeights: [],
                    averageIncomingWeight: 0.5
                });
            }
            
            this.layers.push(layer);
            
            // Create connections with aesthetic flow properties
            if (i < layerCount - 1) {
                const nextLayer = layerSizes[i + 1];
                for (let j = 0; j < layer.length; j++) {
                    const node = layer[j];
                    const fromIndex = j;
                    for (let k = 0; k < nextLayer; k++) {
                        const weight = (Math.random() - 0.5) * 2.0;
                        const weightMagnitude = Math.abs(weight);
                        
                        // Create unique movement patterns for each connection
                        const waveAmplitude = 0.5 + Math.random() * 1.0;
                        const waveSpeed = 0.3 + Math.random() * 0.7;
                        const wavePhase = Math.random() * Math.PI * 2;
                        const flowSpeed = 0.01 + Math.random() * 0.02;
                        const flowPhase = Math.random() * Math.PI * 2;
                        
                        this.connections.push({
                            from: node,
                            to: null,
                            fromLayer: i,
                            toLayer: i + 1,
                            toIndex: k,
                            weight: weight,
                            weightMagnitude: weightMagnitude,
                            strength: weightMagnitude,
                            pulse: 0,
                            dataFlow: 0,
                            dataFlowSpeed: flowSpeed,
                            dataFlowPhase: flowPhase,
                            growthProgress: 0,
                            growthSpeed: 0.003 + Math.random() * 0.005,
                            isGrowing: false,
                            growthStartDelay: (fromIndex / Math.max(1, layerSizes[i])) * 1.2 + Math.random() * 0.5,
                            hasStartedGrowing: false,
                            sourceNodeIndex: fromIndex,
                            
                            // Aesthetic movement properties
                            waveAmplitude: waveAmplitude,
                            waveSpeed: waveSpeed,
                            wavePhase: wavePhase,
                            waveEnabled: true,
                            
                            flowParticles: [],
                            flowSpeed: 0.2 + Math.random() * 0.3,
                            flowPhase: Math.random() * Math.PI * 2,
                            
                            // Connection reveal animation
                            revealProgress: 0,
                            revealStartTime: 0,
                            revealDelay: (fromIndex * nextLayer + k) * this.connectionRevealDelay,
                            revealSpeed: 0.5 + Math.random() * 0.5,
                            hasStartedRevealing: false,
                            
                            // Sinuous movement
                            sinuousAmplitude: 0.8 + Math.random() * 1.2,
                            sinuousSpeed: 0.2 + Math.random() * 0.3,
                            sinuousPhase: Math.random() * Math.PI * 2,
                            sinuousWavelength: 20 + Math.random() * 40
                        });
                    }
                }
            }
        }
        
        // Link connections
        for (let conn of this.connections) {
            conn.to = this.layers[conn.fromLayer + 1][conn.toIndex];
            
            conn.to.incomingWeights.push({
                connection: conn,
                weight: conn.weight,
                weightMagnitude: conn.weightMagnitude,
                fromNode: conn.from
            });
            
            conn.from.outgoingWeights.push({
                connection: conn,
                weight: conn.weight,
                weightMagnitude: conn.weightMagnitude,
                toNode: conn.to
            });
        }
        
        // Calculate average weights
        for (let layer of this.layers) {
            for (let node of layer) {
                if (node.incomingWeights.length > 0) {
                    const sum = node.incomingWeights.reduce((acc, w) => acc + w.weightMagnitude, 0);
                    node.averageIncomingWeight = sum / node.incomingWeights.length;
                } else {
                    node.averageIncomingWeight = 0.5;
                }
            }
        }
    }

    startTraining() {
        // Ensure network is created before starting
        // Don't stop validation animation here - let it run until progress reaches 5%
        // Validation animation will be stopped in the progress handler
        
        if (!this.layers || this.layers.length === 0) {
            this.createNetwork();
        }
        
        // Clear canvas
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.isTraining = true;
        this.isExpanding = false; // No longer using expansion animation
        this.expansionProgress = 0;
        this.expansionStartTime = 0;
        this.trainingProgress = 0.01; // Start with minimal progress to show first layer
        this.currentLoss = 1.0;
        this.currentAccuracy = 0.0;
        this.learningQuality = 0;
        this.trainingComplete = false;
        this.visualizationComplete = false;
        this.trainingStoppedEarly = false; // Reset flag when starting new training
        
        // Reset all nodes to center and invisible
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        for (let layer of this.layers) {
            for (let node of layer) {
                node.x = centerX;
                node.y = centerY;
                node.startX = centerX;
                node.startY = centerY;
                node.visible = false;
                node.currentRadius = 0;
                node.targetRadius = 0;
                node.activation = 0;
                node.targetActivation = 0;
            }
        }
        
        // Reset all connections
        for (let conn of this.connections) {
            conn.isGrowing = false;
            conn.growthProgress = 0;
            conn.pulse = 0.5;
            conn.dataFlow = 0;
        }
    }

    stopTraining() {
        this.isTraining = false;
        this.isExpanding = false;
        this.expansionProgress = 1;
        this.trainingComplete = true;
        this.completeVisualization();
    }
    
    completeVisualization() {
        this.visualizationComplete = true;
        this.trainingComplete = true;
        
        // Ensure all nodes are visible and positioned
        if (this.layers && this.layers.length > 0) {
            for (let layer of this.layers) {
                for (let node of layer) {
                    node.visible = true;
                    node.x = node.baseX || node.x;
                    node.y = node.baseY || node.y;
                    node.currentRadius = this.maxRadius;
                    node.targetRadius = this.maxRadius;
                    node.activation = 0.7;
                    node.targetActivation = 0.7;
                }
            }
        }
        
        if (this.connections && this.connections.length > 0) {
            for (let conn of this.connections) {
                if (conn) {
                    conn.revealProgress = 1.0;
                    conn.hasStartedRevealing = true;
                    conn.isGrowing = true;
                    conn.hasStartedGrowing = true;
                    if (conn.growthProgress !== undefined && conn.growthProgress < 1.0) {
                        conn.growthProgress = 1.0;
                    }
                }
            }
        }
    }

    startValidation() {
        console.log('[NeuralNetwork] Starting validation animation...');
        
        // Ensure network is created first
        if (!this.layers || this.layers.length === 0) {
            console.log('[NeuralNetwork] Network not created yet, creating network for validation animation...');
            this.createNetwork();
        }
        
        // Make all nodes visible for validation animation
        if (this.layers && this.layers.length > 0) {
            for (let layer of this.layers) {
                for (let node of layer) {
                    node.visible = true;
                    // Ensure nodes are at their base positions
                    if (node.baseX && node.baseY) {
                        node.x = node.baseX;
                        node.y = node.baseY;
                    }
                    node.currentRadius = this.maxRadius;
                    node.activation = 0.7;
                }
            }
        }
        
        // Make all connections visible and fully grown for validation animation
        if (this.connections && this.connections.length > 0) {
            for (let conn of this.connections) {
                conn.isGrowing = true;
                conn.growthProgress = 1.0;
                conn.revealProgress = 1.0;
                conn.hasStartedRevealing = true;
                conn.hasStartedGrowing = true;
            }
        }
        
        this.isValidating = true;
        this.validationStartTime = this.globalTime;
        this.validationParticles = [];
        this.validationPulseWaves = [];
        this.validationPathwaySequence = 0;
        
        // Add validation class to visualization container
        const container = this.canvas.parentElement;
        if (container) {
            container.classList.add('validation-active');
            console.log('[NeuralNetwork] Added validation-active class to container');
        }
        
        // Don't show validation message - we use 3D orbiting neuron on canvas instead
        // Hide validation message if it exists
        const validationMsg = document.getElementById('validationMessage');
        if (validationMsg) {
            validationMsg.style.display = 'none';
        }
        
        // Initialize 3D orbit
        this.orbitAngle = 0;
        this.orbitRadius = 60;
        
        console.log('[NeuralNetwork] Validation animation started successfully - Layers:', this.layers ? this.layers.length : 0, 'Connections:', this.connections ? this.connections.length : 0);
    }
    
    stopValidation() {
        console.log('[NeuralNetwork] Stopping validation animation...');
        this.isValidating = false;
        this.validationParticles = [];
        this.validationPulseWaves = [];
        
        // Remove validation class
        const container = this.canvas.parentElement;
        if (container) {
            container.classList.remove('validation-active');
            console.log('[NeuralNetwork] Removed validation-active class from container');
        }
        
        // Hide validation message
        const validationMsg = document.getElementById('validationMessage');
        if (validationMsg) {
            validationMsg.style.display = 'none';
            console.log('[NeuralNetwork] Validation message hidden');
        }
        
        console.log('[NeuralNetwork] Validation animation stopped');
    }
    
    drawOrbitingNeuron() {
        if (!this.isValidating) return;
        
        const ctx = this.ctx;
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Calculate orbit position
        const orbitX = centerX + Math.cos(this.orbitAngle) * this.orbitRadius;
        const orbitY = centerY + Math.sin(this.orbitAngle) * this.orbitRadius;
        
        // Draw 3D neuron (sphere with gradient)
        ctx.save();
        
        const neuronRadius = 8 * this.neuron3DScale;
        
        // Outer glow
        const glowGradient = ctx.createRadialGradient(
            orbitX, orbitY, 0,
            orbitX, orbitY, neuronRadius * 2
        );
        glowGradient.addColorStop(0, 'rgba(147, 51, 234, 0.6)');
        glowGradient.addColorStop(0.5, 'rgba(168, 85, 247, 0.3)');
        glowGradient.addColorStop(1, 'rgba(147, 51, 234, 0)');
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(orbitX, orbitY, neuronRadius * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Main neuron sphere with 3D gradient
        const neuronGradient = ctx.createRadialGradient(
            orbitX - neuronRadius * 0.3,
            orbitY - neuronRadius * 0.3,
            0,
            orbitX, orbitY, neuronRadius
        );
        neuronGradient.addColorStop(0, 'rgba(200, 150, 255, 1)');
        neuronGradient.addColorStop(0.5, 'rgba(147, 51, 234, 0.9)');
        neuronGradient.addColorStop(1, 'rgba(100, 30, 180, 0.8)');
        ctx.fillStyle = neuronGradient;
        ctx.beginPath();
        ctx.arc(orbitX, orbitY, neuronRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.beginPath();
        ctx.arc(orbitX - neuronRadius * 0.3, orbitY - neuronRadius * 0.3, neuronRadius * 0.4, 0, Math.PI * 2);
        ctx.fill();
        
        // Orbit trail
        ctx.strokeStyle = 'rgba(147, 51, 234, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(centerX, centerY, this.orbitRadius, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.restore();
    }
    
    createValidationParticle(conn) {
        if (!conn.from.visible || !conn.to.visible) return null;
        
        return {
            connection: conn,
            progress: 0,
            speed: 0.01 + Math.random() * 0.02,
            size: 2 + Math.random() * 2,
            opacity: 0.6 + Math.random() * 0.4,
            delay: Math.random() * 0.5
        };
    }
    
    updateValidationParticles() {
        if (!this.isValidating) return;
        
        // Create new particles periodically
        if (Math.random() < 0.15 && this.connections.length > 0) {
            const randomConn = this.connections[Math.floor(Math.random() * this.connections.length)];
            if (randomConn && randomConn.growthProgress >= 0.8) {
                const particle = this.createValidationParticle(randomConn);
                if (particle) {
                    this.validationParticles.push(particle);
                }
            }
        }
        
        // Update existing particles
        this.validationParticles = this.validationParticles.filter(particle => {
            particle.progress += particle.speed;
            return particle.progress < 1.0;
        });
    }
    
    createValidationPulseWave(node) {
        if (!node.visible) return null;
        
        return {
            node: node,
            radius: node.currentRadius || 5,
            maxRadius: (node.currentRadius || 5) * 4,
            progress: 0,
            speed: 0.03,
            opacity: 0.6
        };
    }
    
    updateValidationPulseWaves() {
        if (!this.isValidating) return;
        
        // Create new pulse waves periodically from random active nodes
        if (Math.random() < 0.08 && this.layers.length > 0) {
            const allNodes = this.layers.flat();
            const visibleNodes = allNodes.filter(n => n.visible && (n.activation || 0) > 0.3);
            if (visibleNodes.length > 0) {
                const randomNode = visibleNodes[Math.floor(Math.random() * visibleNodes.length)];
                const wave = this.createValidationPulseWave(randomNode);
                if (wave) {
                    this.validationPulseWaves.push(wave);
                }
            }
        }
        
        // Update existing waves
        this.validationPulseWaves = this.validationPulseWaves.filter(wave => {
            wave.progress += wave.speed;
            wave.radius = wave.node.currentRadius + (wave.maxRadius - wave.node.currentRadius) * wave.progress;
            wave.opacity = 0.6 * (1 - wave.progress);
            return wave.progress < 1.0;
        });
    }

    reset() {
        this.isTraining = false;
        this.isValidating = false;
        this.isExpanding = false;
        this.expansionProgress = 0;
        this.trainingProgress = 0;
        this.currentLoss = 1.0;
        this.currentAccuracy = 0.0;
        this.learningQuality = 0;
        this.currentHeartbeatIntensity = 0;
        this.validationParticles = [];
        this.validationPulseWaves = [];
        this.heartbeatTime = 0;
        this.lastUpdateTime = null;
        this.lastHeartbeatSound = 0;
        this.trainingComplete = false;
        this.visualizationComplete = false;
        
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex++) {
            const layer = this.layers[layerIndex];
            for (let nodeIndex = 0; nodeIndex < layer.length; nodeIndex++) {
                const node = layer[nodeIndex];
                node.x = centerX;
                node.y = centerY;
                node.startX = centerX;
                node.startY = centerY;
                node.activation = 0;
                node.targetActivation = 0;
                node.visible = false;
                node.currentRadius = 0;
            }
        }
        
        if (this.audioContext) {
            try {
                this.audioContext.close();
            } catch (e) {
                // Ignore errors
            }
            this.audioContext = null;
            this.initHeartbeatAudio();
        }
        
        this.draw();
    }

    updateTrainingMetrics(loss, accuracy, epoch, totalEpochs, smoothProgress = null) {
        this.currentLoss = loss;
        this.currentAccuracy = accuracy;
        
        // Validate epoch and totalEpochs to prevent invalid progress
        const validEpoch = Math.max(0, Math.min(epoch, totalEpochs || 1));
        const validTotalEpochs = Math.max(1, totalEpochs || 1);
        
        if (smoothProgress !== null && smoothProgress !== undefined) {
            this.trainingProgress = Math.max(0, Math.min(1, smoothProgress));
        } else {
            // Ensure progress is at least 0.01 when training is active to show first layer
            const calculatedProgress = validEpoch / validTotalEpochs;
            this.trainingProgress = this.isTraining ? Math.max(0.01, calculatedProgress) : calculatedProgress;
        }
        
        // Stop validation animation when progress reaches 5% (0.05)
        if (this.isValidating && this.trainingProgress >= 0.05) {
            console.log('[NeuralNetwork] Progress reached 5% in updateTrainingMetrics, stopping validation...');
            this.stopValidation();
        }
        
        this.learningQuality = Math.min(1, accuracy / 100);
        
        // Only mark complete if we're actually at the end AND training wasn't stopped early
        if (!this.trainingStoppedEarly && (this.trainingProgress >= 0.99 || (validEpoch >= validTotalEpochs && validTotalEpochs > 0))) {
            if (!this.trainingComplete) {
                this.trainingComplete = true;
                // Delay completion animation to allow final updates
                setTimeout(() => {
                    if (!this.trainingStoppedEarly && this.trainingProgress >= 0.98) {
                        this.completeVisualization();
                    }
                }, 500);
            }
        }
        
        if (this.isTraining) {
            this.computeRealisticActivations();
            this.visualizeLearningProgress();
        }
    }
    
    computeRealisticActivations() {
        const { modelType = 'machine_learning', modelPurpose = 'machine_learning' } = this.trainingSettings;
        const progress = this.trainingProgress || 0;
        const loss = this.currentLoss || 1.0;
        const accuracy = this.currentAccuracy || 0.0;
        
        const effectiveModelType = modelPurpose !== 'machine_learning' ? modelPurpose : modelType;
        
        switch(effectiveModelType) {
            case 'computer_vision':
                this.computeCVActivations(progress, loss, accuracy);
                break;
            case 'natural_language_processing':
                this.computeNLPActivations(progress, loss, accuracy);
                break;
            case 'machine_learning':
            default:
                this.computeMLActivations(progress, loss, accuracy);
                break;
        }
    }
    
    computeCVActivations(progress, loss, accuracy) {
        const learningQuality = accuracy / 100;
        const lossImprovement = 1 - Math.min(1, loss);
        
        this.layers.forEach((layer, layerIndex) => {
            const depth = layerIndex / Math.max(1, this.layers.length - 1);
            
            layer.forEach(node => {
                if (!node.visible) return;
                
                if (depth < 0.3) {
                    const edgePattern = Math.sin(progress * Math.PI * 4 + node.nodeIndex * 0.5) * 0.25;
                    node.realisticActivation = 0.35 + 
                        edgePattern +
                        lossImprovement * 0.25 +
                        learningQuality * 0.15;
                } 
                else if (depth < 0.7) {
                    const featurePattern = Math.sin(progress * Math.PI * 2 + node.nodeIndex * 0.3) * 0.2;
                    node.realisticActivation = 0.45 + 
                        featurePattern +
                        lossImprovement * 0.3 +
                        learningQuality * 0.25;
                }
                else {
                    const stability = Math.min(1, progress * 1.5);
                    node.realisticActivation = 0.55 + 
                        stability * 0.25 +
                        lossImprovement * 0.4 +
                        learningQuality * 0.3;
                }
                
                node.realisticActivation = Math.max(0.2, Math.min(0.95, node.realisticActivation));
                node.targetActivation = node.realisticActivation;
            });
        });
    }
    
    computeNLPActivations(progress, loss, accuracy) {
        const learningQuality = accuracy / 100;
        const lossImprovement = 1 - Math.min(1, loss);
        
        this.layers.forEach((layer, layerIndex) => {
            const depth = layerIndex / Math.max(1, this.layers.length - 1);
            const layerProgress = Math.min(1, progress * (1 - depth * 0.15));
            
            layer.forEach(node => {
                if (!node.visible) return;
                
                if (depth < 0.25) {
                    const embedPattern = Math.sin(progress * Math.PI * 3 + node.nodeIndex * 0.4) * 0.2;
                    node.realisticActivation = 0.4 + 
                        embedPattern +
                        lossImprovement * 0.2 +
                        learningQuality * 0.2;
                }
                else if (depth < 0.75) {
                    const attentionPattern = Math.sin(progress * Math.PI * 1.5 + node.nodeIndex * 0.2) * 0.15;
                    node.realisticActivation = 0.5 + 
                        attentionPattern +
                        lossImprovement * 0.3 +
                        learningQuality * 0.3;
                }
                else {
                    node.realisticActivation = 0.6 + 
                        layerProgress * 0.2 +
                        lossImprovement * 0.4 +
                        learningQuality * 0.35;
                }
                
                node.realisticActivation = Math.max(0.25, Math.min(0.95, node.realisticActivation));
                node.targetActivation = node.realisticActivation;
            });
        });
    }
    
    computeMLActivations(progress, loss, accuracy) {
        const learningQuality = accuracy / 100;
        const lossImprovement = 1 - Math.min(1, loss);
        
        this.layers.forEach((layer, layerIndex) => {
            const depth = layerIndex / Math.max(1, this.layers.length - 1);
            const layerProgress = Math.min(1, progress * (1 - depth * 0.1));
            
            layer.forEach(node => {
                if (!node.visible) return;
                
                const baseActivation = 0.4 + layerProgress * 0.3;
                const weightContribution = node.averageIncomingWeight * 0.2;
                const learningContribution = lossImprovement * 0.25 + learningQuality * 0.25;
                const variation = Math.sin(progress * Math.PI + node.nodeIndex * 0.1) * 0.1;
                
                node.realisticActivation = baseActivation + 
                    weightContribution + 
                    learningContribution + 
                    variation;
                
                node.realisticActivation = Math.max(0.3, Math.min(0.9, node.realisticActivation));
                node.targetActivation = node.realisticActivation;
            });
        });
    }
    
    visualizeLearningProgress() {
        const progress = this.trainingProgress || 0;
        const loss = this.currentLoss || 1.0;
        const accuracy = this.currentAccuracy || 0.0;
        const learningQuality = accuracy / 100;
        const lossImprovement = 1 - Math.min(1, loss);
        
        this.connections.forEach((conn, connIndex) => {
            if (!conn.from.visible || !conn.to.visible) return;
            
            const learningRate = this.trainingSettings.learningRate || 0.001;
            const weightUpdateAmount = Math.sin(progress * Math.PI * 2 + connIndex * 0.1) * 
                                       learningRate * 50 * learningQuality * 0.1;
            
            if (conn.originalWeight === undefined) {
                conn.originalWeight = conn.weight;
            }
            
            conn.weight = conn.originalWeight + weightUpdateAmount;
            conn.weight = Math.max(-1.0, Math.min(1.0, conn.weight));
            conn.weightMagnitude = Math.abs(conn.weight);
            
            conn.dataFlowSpeed = 0.02 + (learningQuality * 0.03);
            
            const gradientFlow = lossImprovement * conn.weightMagnitude;
            conn.pulse = Math.max(0.2, Math.min(0.95, 0.3 + gradientFlow * 0.65));
            
            if (conn.to.incomingWeights) {
                const weightRef = conn.to.incomingWeights.find(w => w.connection === conn);
                if (weightRef) {
                    weightRef.weight = conn.weight;
                    weightRef.weightMagnitude = conn.weightMagnitude;
                }
            }
        });
        
        this.layers.forEach(layer => {
            layer.forEach(node => {
                if (node.incomingWeights && node.incomingWeights.length > 0) {
                    const sum = node.incomingWeights.reduce((acc, w) => acc + w.weightMagnitude, 0);
                    node.averageIncomingWeight = sum / node.incomingWeights.length;
                }
            });
        });
        
        this.layers.forEach((layer, layerIndex) => {
            const layerLearningRate = progress * (1 - (layerIndex / Math.max(1, this.layers.length - 1)) * 0.2);
            
            layer.forEach(node => {
                if (!node.visible) return;
                
                node.learningProgress = layerLearningRate;
                node.confidence = learningQuality;
                node.lossImprovement = lossImprovement;
            });
        });
    }

    update() {
        // Update global time for animations
        const now = Date.now() / 1000;
        if (!this.lastUpdateTime) {
            this.lastUpdateTime = now;
        }
        const deltaTime = now - this.lastUpdateTime;
        this.lastUpdateTime = now;
        
        this.globalTime += deltaTime;
        this.breathingPhase += deltaTime * 0.3;
        this.sinuousPhase += deltaTime * 0.2;
        this.pulsePhase += deltaTime * 0.5;
        
        // Update heartbeat
        if (this.isTraining) {
            this.heartbeatTime += deltaTime;
            
            const heartbeatPhase = (this.heartbeatTime % 2.0) / 2.0;
            
            let heartbeatIntensity = 0;
            if (heartbeatPhase < 0.15) {
                heartbeatIntensity = Math.sin((heartbeatPhase / 0.15) * Math.PI) * 0.08;
            } else if (heartbeatPhase < 0.25) {
                heartbeatIntensity = Math.sin(((heartbeatPhase - 0.15) / 0.1) * Math.PI) * 0.06;
            } else {
                heartbeatIntensity = 0;
            }
            
            if ((heartbeatPhase < 0.15 && heartbeatPhase > 0.14) || 
                (heartbeatPhase < 0.25 && heartbeatPhase > 0.24)) {
                this.playHeartbeatSound();
            }
            
            this.currentHeartbeatIntensity = heartbeatIntensity;
        } else {
            this.currentHeartbeatIntensity = 0;
        }
        
        // Update validation animations
        if (this.isValidating) {
            this.updateValidationParticles();
            this.updateValidationPulseWaves();
            this.validationPathwaySequence += deltaTime * 0.5;
            // Update 3D orbiting neuron
            this.orbitAngle += this.orbitSpeed;
            this.neuron3DScale = 0.8 + Math.sin(this.globalTime * 2) * 0.2;
        }
        
        if (!this.isTraining) {
            // If visualization is complete, show the final state
            if (this.visualizationComplete || this.trainingComplete) {
                // Ensure all nodes are visible and positioned
                for (let layer of this.layers) {
                    for (let node of layer) {
                        if (!node.visible) {
                            node.visible = true;
                        }
                        node.x = node.baseX || node.x;
                        node.y = node.baseY || node.y;
                        node.currentRadius = this.maxRadius;
                        node.activation = 0.7;
                    }
                }
                return;
            }
            
            if (this.trainingProgress > 0.1) {
                // Gentle breathing animation after training
                const pulseTime = now * 0.8;
                const heartbeatPhase = (pulseTime * 0.8) % 2;
                let heartbeatIntensity = 0;
                
                if (heartbeatPhase < 0.3) {
                    heartbeatIntensity = Math.sin(heartbeatPhase * Math.PI / 0.3) * 0.6 + 0.4;
                } else if (heartbeatPhase < 0.5) {
                    heartbeatIntensity = Math.sin((heartbeatPhase - 0.3) * Math.PI / 0.2) * 0.5 + 0.45;
                } else {
                    heartbeatIntensity = 0.35 + Math.sin(heartbeatPhase * Math.PI) * 0.1;
                }
                
                for (let layer of this.layers) {
                    for (let node of layer) {
                        if (node.visible && this.useSubtleMovements) {
                            // Subtle breathing movement
                            const breathing = Math.sin(this.globalTime * node.breathingSpeed + node.breathingPhase) * 
                                            node.breathingAmplitude * this.breathingAmplitude * 0.5;
                            
                            // Gentle floating
                            const floatX = Math.sin(this.globalTime * node.floatSpeed + node.floatDirectionX) * 
                                         node.floatAmplitude * 0.3;
                            const floatY = Math.cos(this.globalTime * node.floatSpeed + node.floatDirectionY) * 
                                         node.floatAmplitude * 0.3;
                            
                            // Pulse animation
                            const pulse = Math.sin(this.globalTime * node.pulseSpeed + node.pulsePhase) * 
                                        node.pulseAmplitude;
                            
                            // Sway animation
                            const sway = Math.sin(this.globalTime * node.swaySpeed + node.swayPhase) * 
                                       node.swayAmplitude;
                            
                            node.targetX = node.baseX + floatX + sway * 0.5;
                            node.targetY = node.baseY + floatY + sway;
                            
                            const nodeOffset = (node.layerIndex * 0.1 + node.nodeIndex * 0.02);
                            const nodeHeartbeat = heartbeatIntensity + Math.sin(pulseTime * 0.3 + nodeOffset) * 0.15;
                            node.targetActivation = Math.max(0.35, Math.min(0.85, nodeHeartbeat + pulse * 0.1));
                            node.activation += (node.targetActivation - node.activation) * 0.08;
                            
                            const radiusPulse = 0.9 + (heartbeatIntensity * 0.2) + breathing * 0.1;
                            node.targetRadius = this.maxRadius * radiusPulse;
                            node.currentRadius += (node.targetRadius - node.currentRadius) * 0.15;
                            
                            // Smooth movement towards target position
                            node.x += (node.targetX - node.x) * 0.1;
                            node.y += (node.targetY - node.y) * 0.1;
                        }
                    }
                }
                
                for (let conn of this.connections) {
                    if (conn.from.visible && conn.to.visible) {
                        const weightPulse = conn.weightMagnitude || 0.5;
                        const sourceActivation = conn.from.activation || 0;
                        const basePulse = sourceActivation * weightPulse;
                        const heartbeatPulse = basePulse * (0.7 + heartbeatIntensity * 0.3);
                        conn.pulse = Math.max(0.2, Math.min(0.9, heartbeatPulse));
                        
                        // Update flow animation
                        conn.dataFlow += conn.dataFlowSpeed * this.connectionFlowSpeed;
                        if (conn.dataFlow >= 1) {
                            conn.dataFlow = 0;
                        }
                    }
                }
                return;
            }
            
            // Only fade out if training hasn't started or was reset
            // Don't fade out if we had progress before stopping
            if (this.trainingProgress < 0.01) {
                this.trainingProgress *= 0.95;
                
                for (let layer of this.layers) {
                    for (let node of layer) {
                        node.targetActivation = 0;
                        node.activation *= 0.95;
                        node.targetRadius = 0;
                        node.currentRadius += (node.targetRadius - node.currentRadius) * 0.1;
                        if (node.currentRadius < 0.1) {
                            node.visible = false;
                        }
                    }
                }
                for (let conn of this.connections) {
                    conn.pulse *= 0.95;
                }
            }
            return;
        }

        // Training animation - layer-by-layer reveal based on training progress
        if (this.isTraining) {
            // Use actual progress (can be 0-1, input layer hidden until 0.05)
            const progress = Math.max(0, Math.min(1, this.trainingProgress || 0));
            const layersCount = this.layers.length;
            const learningQuality = this.learningQuality || 0;
            const lossImprovement = 1 - Math.min(1, this.currentLoss || 1);
            const combinedQuality = Math.max(0.5, (learningQuality + lossImprovement) / 2);

            // Calculate which layers should be visible based on progress
            // Each layer appears gradually as training progresses
            // Hide input layer (first layer) until 5% progress (progress >= 0.05)
            const totalLayers = this.layers.length;
            const visibleLayers = progress >= 0.05 
                ? Math.max(1, Math.min(totalLayers, Math.floor((progress - 0.05) * totalLayers * 1.2) + 1))
                : 0; // Hide all layers (including input) until 5% progress
            
            // Smooth transition between layers appearing
            for (let i = 0; i < this.layers.length; i++) {
                const layer = this.layers[i];
                const layerVisibility = i < visibleLayers ? 1.0 : 
                                      i === visibleLayers ? (progress * totalLayers * 1.2) - i : 0;
                
                for (let node of layer) {
                    if (layerVisibility > 0) {
                        if (!node.visible) {
                            node.visible = true;
                            // Start from center if not already positioned
                            if (Math.abs(node.x - this.canvas.width / 2) < 1) {
                                node.x = this.canvas.width / 2;
                                node.y = this.canvas.height / 2;
                            }
                        }
                        
                        // Calculate heartbeat pulse effect on position
                        const heartbeatPulse = this.currentHeartbeatIntensity || 0;
                        const pulseOffsetX = Math.cos(this.globalTime * 2) * heartbeatPulse * 2;
                        const pulseOffsetY = Math.sin(this.globalTime * 2) * heartbeatPulse * 2;
                        
                        // Animate nodes to their positions with heartbeat pulse
                        const moveSpeed = 0.08 + (layerVisibility * 0.07);
                        const targetX = node.baseX + pulseOffsetX;
                        const targetY = node.baseY + pulseOffsetY;
                        node.x += (targetX - node.x) * moveSpeed;
                        node.y += (targetY - node.y) * moveSpeed;
                        
                        // Calculate activation based on training progress with heartbeat
                        const baseActivation = 0.3 + (progress * 0.4) + (lossImprovement * 0.2);
                        const heartbeatActivation = baseActivation + (heartbeatPulse * 0.1);
                        const nodeActivation = Math.max(0.2, Math.min(0.9, 
                            heartbeatActivation + 
                            (node.averageIncomingWeight * 0.2) +
                            Math.sin(this.globalTime + i * 0.3) * 0.05
                        ));
                        
                        node.activation += (nodeActivation - node.activation) * 0.05;
                        
                        // Animate radius with heartbeat pulse
                        const heartbeatRadiusMultiplier = 1 + (heartbeatPulse * 0.4);
                        const targetRadius = this.maxRadius * layerVisibility * heartbeatRadiusMultiplier;
                        node.currentRadius += (targetRadius - node.currentRadius) * 0.1;
                    } else if (node.visible) {
                        // Fade out nodes in layers that are no longer visible
                        node.activation *= 0.95;
                        node.currentRadius *= 0.95;
                        if (node.currentRadius < 0.1) {
                            node.visible = false;
                        }
                    }
                }
            }

            // Update connections between visible layers
            for (let conn of this.connections) {
                const fromLayer = conn.fromLayer;
                const toLayer = conn.toLayer;
                const fromVisible = fromLayer < visibleLayers || 
                                  (fromLayer === visibleLayers && (progress * totalLayers * 1.2) - fromLayer > 0.3);
                const toVisible = toLayer < visibleLayers || 
                                (toLayer === visibleLayers && (progress * totalLayers * 1.2) - toLayer > 0.3);
                
                if (fromVisible && toVisible && conn.from.visible && conn.to.visible) {
                    // Calculate distance between nodes
                    const dx = conn.to.x - conn.from.x;
                    const dy = conn.to.y - conn.from.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const maxDistance = Math.abs(conn.to.baseX - conn.from.baseX) * 1.5;
                    
                    // Only start growing connection when nodes are reasonably close to their positions
                    const nodesReady = distance < maxDistance * 1.2;
                    
                    if (nodesReady) {
                        if (!conn.hasStartedGrowing) {
                            conn.hasStartedGrowing = true;
                        }
                        
                        if (conn.hasStartedGrowing && !conn.isGrowing) {
                            conn.isGrowing = true;
                        }
                        
                        if (conn.isGrowing && conn.growthProgress < 1.0) {
                            // Growth speed depends on training progress and connection strength
                            const growthRate = 0.01 * (0.5 + conn.weightMagnitude * 0.5) * 
                                             (1 + progress * 0.5) * this.animationSpeed;
                            conn.growthProgress = Math.min(1.0, conn.growthProgress + growthRate);
                        }
                        
                        // Update connection pulse based on training metrics
                        const weightPulse = conn.weightMagnitude || 0.5;
                        const sourceActivation = conn.from.activation || 0.3;
                        const progressPulse = 0.3 + (progress * 0.4);
                        conn.pulse = Math.max(0.2, Math.min(0.95, 
                            weightPulse * sourceActivation * progressPulse * conn.growthProgress
                        ));
                        
                        // Data flow animation
                        if (conn.growthProgress > 0.3) {
                            conn.dataFlow += conn.dataFlowSpeed * this.connectionFlowSpeed * 
                                           (0.5 + conn.weightMagnitude * 0.5);
                            if (conn.dataFlow >= 1) conn.dataFlow = 0;
                        }
                    }
                } else {
                    conn.isGrowing = false;
                    if (conn.growthProgress > 0) {
                        conn.growthProgress *= 0.95;
                    }
                }
            }
        }
        
        // Force complete visualization when training is complete
        if (this.trainingComplete || this.visualizationComplete) {
            for (let conn of this.connections) {
                conn.growthProgress = 1.0;
            }
            
            // Ensure all nodes are visible and properly positioned
            for (let layer of this.layers) {
                for (let node of layer) {
                    if (!node.visible) {
                        node.visible = true;
                    }
                    node.x = node.baseX;
                    node.y = node.baseY;
                    node.currentRadius = this.maxRadius;
                    node.activation = 0.7;
                }
            }
        }
    }
    
    drawValidationParticles() {
        if (!this.isValidating || this.validationParticles.length === 0) return;
        
        for (let particle of this.validationParticles) {
            const conn = particle.connection;
            if (!conn.from.visible || !conn.to.visible) continue;
            
            const fromX = conn.from.x;
            const fromY = conn.from.y;
            const toX = conn.to.x;
            const toY = conn.to.y;
            
            // Calculate particle position along connection
            const t = particle.progress;
            const x = fromX + (toX - fromX) * t;
            const y = fromY + (toY - fromY) * t;
            
            // Draw particle with purple/blue color for validation
            const alpha = particle.opacity * (1 - t * 0.5);
            this.ctx.globalAlpha = alpha;
            this.ctx.fillStyle = `rgba(147, 51, 234, ${alpha})`; // Purple for validation
            this.ctx.beginPath();
            this.ctx.arc(x, y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Add glow
            const glowRadius = particle.size * 2;
            const glowGradient = this.ctx.createRadialGradient(x, y, 0, x, y, glowRadius);
            glowGradient.addColorStop(0, `rgba(147, 51, 234, ${alpha * 0.5})`);
            glowGradient.addColorStop(1, 'rgba(147, 51, 234, 0)');
            this.ctx.fillStyle = glowGradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
            this.ctx.fill();
        }
        this.ctx.globalAlpha = 1;
    }
    
    drawValidationPulseWaves() {
        if (!this.isValidating || this.validationPulseWaves.length === 0) return;
        
        for (let wave of this.validationPulseWaves) {
            if (!wave.node.visible) continue;
            
            const alpha = wave.opacity * (1 - wave.progress);
            this.ctx.globalAlpha = alpha;
            this.ctx.strokeStyle = `rgba(147, 51, 234, ${alpha})`; // Purple for validation
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(wave.node.x, wave.node.y, wave.radius, 0, Math.PI * 2);
            this.ctx.stroke();
        }
        this.ctx.globalAlpha = 1;
    }
    
    drawValidationPathwayGlow(conn) {
        if (!this.isValidating || !conn.from.visible || !conn.to.visible) return false;
        
        // Sequential pathway lighting based on layer index
        const layerSequence = (this.validationPathwaySequence + conn.fromLayer * 0.5) % (Math.PI * 2);
        const glowIntensity = (Math.sin(layerSequence) + 1) / 2;
        
        if (glowIntensity > 0.3) {
            const fromX = conn.from.x;
            const fromY = conn.from.y;
            const toX = conn.to.x;
            const toY = conn.to.y;
            
            // Draw glowing pathway
            const gradient = this.ctx.createLinearGradient(fromX, fromY, toX, toY);
            gradient.addColorStop(0, `rgba(147, 51, 234, ${0.2 + glowIntensity * 0.4})`);
            gradient.addColorStop(1, `rgba(59, 130, 246, ${0.2 + glowIntensity * 0.4})`);
            
            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = 2 + glowIntensity * 2;
            this.ctx.beginPath();
            this.ctx.moveTo(fromX, fromY);
            this.ctx.lineTo(toX, toY);
            this.ctx.stroke();
            return true;
        }
        return false;
    }

    draw() {
        // Debug logging (can be removed later)
        if (this._drawCallCount === undefined) {
            this._drawCallCount = 0;
        }
        this._drawCallCount++;
        if (this._drawCallCount % 60 === 0) { // Log every 60 frames (~1 second at 60fps)
            console.log('[NeuralNetwork] draw() called', this._drawCallCount, 'times | Canvas:', 
                this.canvas ? `${this.canvas.width}x${this.canvas.height}` : 'null',
                '| isTraining:', this.isTraining,
                '| layers:', this.layers ? this.layers.length : 0);
        }
        
        if (!this.canvas) {
            console.error('[NeuralNetwork] draw() called but canvas is null!');
            return;
        }
        
        if (this.canvas.width === 0 || this.canvas.height === 0) {
            console.warn('[NeuralNetwork] Canvas has zero dimensions, calling resize()');
            this.resize();
            if (this.canvas.width === 0 || this.canvas.height === 0) {
                console.error('[NeuralNetwork] Canvas still has zero dimensions after resize!');
                return;
            }
        }
        
        // Clear with subtle gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        gradient.addColorStop(0, '#F7F3ED');
        gradient.addColorStop(1, '#F0EAE0');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Empty canvas on start - only show idle when truly idle
        if (!this.isTraining && this.trainingProgress < 0.1 && !this.isExpanding) {
            // Only show idle grid when NOT training AND no nodes visible
            const anyNodesVisible = this.layers.some(layer => 
                layer.some(node => node.visible && node.currentRadius > 0.1)
            );
            
            if (!anyNodesVisible) {
                this.drawIdleGrid();
                this.ctx.globalAlpha = 1;
                return;
            }
        }
        this.ctx.globalAlpha = 1;

        this.ctx.lineWidth = 1;
        
        // Draw connections in LAYER ORDER
        for (let fromLayerIndex = 0; fromLayerIndex < this.layers.length - 1; fromLayerIndex++) {
            const layerConnections = this.connections.filter(conn => conn.fromLayer === fromLayerIndex);
            
            for (let conn of layerConnections) {
                if (!conn.isGrowing || conn.growthProgress <= 0) continue;
                if (!conn.from.visible || !conn.to.visible) continue;
                
                const fromX = conn.from.x;
                const fromY = conn.from.y;
                const toX = conn.to.x;
                const toY = conn.to.y;
                
                const drawToX = fromX + (toX - fromX) * conn.growthProgress;
                const drawToY = fromY + (toY - fromY) * conn.growthProgress;
                
                // Draw validation pathway glow if in validation mode
                let drawnGlow = false;
                if (this.isValidating) {
                    drawnGlow = this.drawValidationPathwayGlow(conn);
                }
                
                // Draw connection line (if not already drawn by glow)
                if (!drawnGlow) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(fromX, fromY);
                    this.ctx.lineTo(drawToX, drawToY);
                    
                    const opacity = 0.15 + (conn.weightMagnitude * 0.2);
                    const lineWidth = 1 + (conn.weightMagnitude * 1.5);
                    
                    // Color shift to purple during validation
                    if (this.isValidating) {
                        this.ctx.strokeStyle = `rgba(147, 51, 234, ${opacity * 1.2})`;
                    } else if (conn.weight > 0) {
                        this.ctx.strokeStyle = `rgba(100, 150, 100, ${opacity})`;
                    } else {
                        this.ctx.strokeStyle = `rgba(150, 100, 100, ${opacity})`;
                    }
                    
                    this.ctx.lineWidth = lineWidth;
                    this.ctx.stroke();
                }
            }
        }
        
        // Draw validation effects (particles, pulse waves, and orbiting neuron)
        if (this.isValidating) {
            this.drawValidationParticles();
            this.drawValidationPulseWaves();
            this.drawOrbitingNeuron();
        }
        
        this.ctx.globalAlpha = 1;
        
        // Draw nodes with aesthetic effects
        for (let layer of this.layers) {
            for (let node of layer) {
                const shouldDraw = node.visible && (node.currentRadius > 0.05 || this.trainingProgress > 0.05);
                if (!shouldDraw) {
                    continue;
                }
                
                const alpha = node.activation || 0.7;
                const radius = Math.max(0.5, node.currentRadius);
                
                if (alpha > 0.01) {
                    const isHovered = this.hoveredNode === node;
                    const nodeRadius = isHovered ? radius * 1.3 : radius;
                    const nodeAlpha = isHovered ? Math.min(1, alpha * 1.2) : alpha;
                    
                    const weightMagnitude = node.averageIncomingWeight || 0.5;
                    const outputBrightness = Math.min(1, (node.outputValue || 0) / 2);
                    const combinedBrightness = (weightMagnitude * 0.7) + (outputBrightness * 0.3);
                    
                    // Color based on brightness with warm tones
                    let nodeR, nodeG, nodeB;
                    if (combinedBrightness < 0.33) {
                        const t = combinedBrightness / 0.33;
                        nodeR = Math.round(130 + t * 25);
                        nodeG = Math.round(110 + t * 20);
                        nodeB = Math.round(95 + t * 15);
                    } else if (combinedBrightness < 0.66) {
                        const t = (combinedBrightness - 0.33) / 0.33;
                        nodeR = Math.round(155 + t * 45);
                        nodeG = Math.round(135 + t * 40);
                        nodeB = Math.round(115 + t * 35);
                    } else {
                        const t = (combinedBrightness - 0.66) / 0.34;
                        nodeR = Math.round(200 + t * 55);
                        nodeG = Math.round(175 + t * 50);
                        nodeB = Math.round(150 + t * 45);
                    }
                    
                    const drawAlpha = Math.max(0.4, nodeAlpha);
                    const heartbeatGlow = this.currentHeartbeatIntensity || 0;
                    
                    // Base glow
                    const pulseGlowRadius = nodeRadius * (1.0 + heartbeatGlow * 0.3);
                    
                    // Draw outer glow for active nodes
                    if (nodeAlpha > 0.4) {
                        const glowRadius = pulseGlowRadius * 1.8;
                        const glowAlpha = (nodeAlpha - 0.4) * 0.2;
                        this.ctx.globalAlpha = glowAlpha;
                        this.ctx.beginPath();
                        this.ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
                        this.ctx.fillStyle = `rgba(${nodeR}, ${nodeG}, ${nodeB}, 0.3)`;
                        this.ctx.fill();
                    }
                    
                    // Draw node with subtle gradient
                    const gradient = this.ctx.createRadialGradient(
                        node.x, node.y, 0,
                        node.x, node.y, pulseGlowRadius
                    );
                    
                    if (isHovered) {
                        gradient.addColorStop(0, `rgba(${nodeR + 30}, ${nodeG + 30}, ${nodeB + 30}, ${drawAlpha})`);
                        gradient.addColorStop(1, `rgba(${nodeR}, ${nodeG}, ${nodeB}, ${drawAlpha * 0.7})`);
                    } else {
                        gradient.addColorStop(0, `rgba(${nodeR + 15}, ${nodeG + 15}, ${nodeB + 15}, ${drawAlpha})`);
                        gradient.addColorStop(1, `rgba(${nodeR}, ${nodeG}, ${nodeB}, ${drawAlpha * 0.6})`);
                    }
                    
                    this.ctx.globalAlpha = 1;
                    this.ctx.fillStyle = gradient;
                    this.ctx.beginPath();
                    this.ctx.arc(node.x, node.y, pulseGlowRadius, 0, Math.PI * 2);
                    this.ctx.fill();
                    
                    // Add inner highlight
                    if (nodeAlpha > 0.5) {
                        const highlightRadius = pulseGlowRadius * 0.4;
                        this.ctx.globalAlpha = 0.3;
                        this.ctx.beginPath();
                        this.ctx.arc(node.x - highlightRadius * 0.3, node.y - highlightRadius * 0.3, 
                                   highlightRadius, 0, Math.PI * 2);
                        this.ctx.fillStyle = `rgba(255, 255, 255, 0.4)`;
                        this.ctx.fill();
                    }
                    
                    // Add subtle pulse effect
                    if (node.pulseAmplitude && this.useSubtleMovements) {
                        const pulse = Math.sin(this.globalTime * node.pulseSpeed + node.pulsePhase) * 
                                    node.pulseAmplitude * 0.5;
                        const pulseRadius = pulseGlowRadius * (1 + pulse * 0.1);
                        
                        this.ctx.globalAlpha = 0.1;
                        this.ctx.beginPath();
                        this.ctx.arc(node.x, node.y, pulseRadius, 0, Math.PI * 2);
                        this.ctx.strokeStyle = `rgba(${nodeR}, ${nodeG}, ${nodeB}, 0.3)`;
                        this.ctx.lineWidth = 1;
                        this.ctx.stroke();
                    }
                    
                    // Hover effect
                    if (isHovered) {
                        this.ctx.strokeStyle = `rgba(${nodeR}, ${nodeG}, ${nodeB}, 0.8)`;
                        this.ctx.lineWidth = 2;
                        this.ctx.globalAlpha = 0.8;
                        this.ctx.beginPath();
                        this.ctx.arc(node.x, node.y, nodeRadius + 6, 0, Math.PI * 2);
                        this.ctx.stroke();
                        
                        // Additional inner ring for hover
                        this.ctx.strokeStyle = `rgba(${nodeR + 30}, ${nodeG + 30}, ${nodeB + 30}, 0.6)`;
                        this.ctx.lineWidth = 1;
                        this.ctx.beginPath();
                        this.ctx.arc(node.x, node.y, nodeRadius + 3, 0, Math.PI * 2);
                        this.ctx.stroke();
                    }
                }
            }
        }
        this.ctx.globalAlpha = 1;
    }

    drawIdleGrid() {
        if (this.trainingProgress > 0.1 || this.isExpanding) {
            return;
        }
        this.drawSingleNeuron();
    }

    drawSingleNeuron() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        if (!centerX || !centerY || isNaN(centerX) || isNaN(centerY)) {
            return;
        }
        
        // Draw a single breathing neuron
        const time = this.globalTime;
        const pulse = 0.95 + Math.sin(time * 1.5) * 0.05;
        const radius = 7 * pulse;
        
        // Subtle floating movement
        const floatOffset = Math.sin(time * 0.8) * 10;
        const floatX = Math.sin(time * 0.5) * 3;
        const y = centerY + floatOffset;
        const x = centerX + floatX;
        
        // Draw with gradient
        const gradient = this.ctx.createRadialGradient(
            x, y, 0,
            x, y, radius
        );
        gradient.addColorStop(0, 'rgba(160, 160, 160, 0.9)');
        gradient.addColorStop(1, 'rgba(120, 120, 120, 0.7)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Add subtle glow
        this.ctx.globalAlpha = 0.3;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius * 1.5, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(140, 140, 140, 0.2)';
        this.ctx.fill();
        this.ctx.globalAlpha = 1;
        
        // Add highlight
        this.ctx.beginPath();
        this.ctx.arc(x - radius * 0.3, y - radius * 0.3, radius * 0.4, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        this.ctx.fill();
        
        // Draw text prompt
        this.ctx.font = '14px Arial, sans-serif';
        this.ctx.fillStyle = 'rgba(100, 100, 100, 0.7)';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Click "Start Training" to begin', centerX, centerY + 50);
    }

    animate() {
        if (!this.canvas) {
            console.error('[NeuralNetwork] animate() called but canvas is null!');
            return;
        }
        
        this.update();
        this.draw();
        if (this.hoveredNode) {
            this.updateTooltip();
        }
        
        // Continue animation loop
        if (!this._stopAnimation) {
            this.animationId = requestAnimationFrame(() => this.animate());
        }
    }
}

// Initialize neural network
let neuralNetwork;

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('neuralCanvas');
    if (canvas) {
        neuralNetwork = new NeuralNetworkVisualization('neuralCanvas');
        window.neuralNetwork = neuralNetwork;
    }
});

if (document.readyState !== 'loading') {
    const canvas = document.getElementById('neuralCanvas');
    if (canvas && !neuralNetwork) {
        neuralNetwork = new NeuralNetworkVisualization('neuralCanvas');
        window.neuralNetwork = neuralNetwork;
    }
}
