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

/**
 * Cloud Training Handler
 * Implements complete cloud training workflow:
 * 1. Instance IP retrieval after launch
 * 2. Remote training script execution
 * 3. Progress streaming over SSH
 * 4. Model download automation
 */

const CloudSSHUtils = require('./cloud-ssh-utils');
const path = require('path');
const fs = require('fs');
const os = require('os');

class CloudTrainingHandler {
    constructor(canopywaveClient, mainWindow) {
        this.client = canopywaveClient;
        this.mainWindow = mainWindow;
        this.sshConnection = null;
        this.instanceId = null;
        this.instanceIP = null; // Store IP for later use
        this.cloudConfig = null;
        this.progressCheckInterval = null;
        this.isTraining = false;
        this.usingTmux = false; // Track if using tmux or nohup
        this.tmuxSessionName = null; // Store tmux session name if using tmux
        this.ultralyticsSaveDir = null; // Store actual save directory from Ultralytics results
        this.maxTrainingTimeout = null; // Timeout to auto-terminate after maxTrainingHours
        this.instanceStartTime = null; // Track when instance was launched
        this.costWarningInterval = null; // Interval for periodic cost warnings
    }

    /**
     * Mask IPv4 address for privacy (show only last octet)
     * @param {string} ip - IP address (can be null/undefined/IPv6)
     * @returns {string} Masked IP (e.g., "x.x.x.123")
     */
    maskIPv4(ip) {
        if (!ip || typeof ip !== 'string') return 'x.x.x.x';
        const parts = ip.split('.');
        return parts.length === 4 ? `x.x.x.${parts[3]}` : 'x.x.x.x';
    }

    /**
     * Main cloud training workflow
     * @param {object} config - Training configuration
     * @param {string} config.project - CanopyWave project
     * @param {string} config.region - CanopyWave region
     * @param {string} config.flavor - GPU type (e.g., "H100-4")
     * @param {string} config.image - OS image (e.g., "GPU-Ubuntu.22.04")
     * @param {string} config.password - SSH password
     * @param {string} config.datasetPath - Local dataset path
     * @param {object} config.trainingSettings - Training parameters
     * @returns {Promise<object>} Training result
     */
    async startCloudTraining(config) {
        try {
            // Store full config including apiKey for emergency termination
            this.cloudConfig = {
                ...config,
                apiKey: config.apiKey // Ensure apiKey is stored for emergency termination
            };
            this.isTraining = true;

            // Step 1: Launch instance
            this.sendStatus('Launching cloud GPU instance...');
            const instance = await this.launchInstance(config);
            this.instanceId = instance.id;
            this.instanceStartTime = Date.now(); // Track start time for cost monitoring
            
            console.log('[CloudTraining] Instance launched:', instance);
            // Only show short prefix of instance ID (privacy/security)
            const shortId = (instance.id || '').toString().slice(0, 8);
            this.sendLog(`Instance launched (id: ${shortId}…)`);
            
            // CRITICAL: Set up auto-termination timeout based on maxTrainingHours
            // This ensures instances are ALWAYS terminated, even if training hangs or app crashes
            const maxHours = config.maxTrainingHours || 24;
            const maxMs = maxHours * 60 * 60 * 1000; // Convert hours to milliseconds
            this.maxTrainingTimeout = setTimeout(async () => {
                console.warn(`[CloudTraining] MAX TRAINING TIME REACHED (${maxHours} hours). Auto-terminating instance to prevent overcharging.`);
                this.sendLog(`⚠️ Maximum training time (${maxHours} hours) reached. Terminating instance to prevent additional charges.`, 'warning');
                try {
                    await this.terminateInstance();
                } catch (timeoutError) {
                    console.error('[CloudTraining] Failed to auto-terminate on timeout:', timeoutError);
                    // Try again with a new API client if needed
                    if (this.instanceId && config.apiKey) {
                        try {
                            const emergencyClient = new (require('./canopywave-api'))(config.apiKey);
                            await emergencyClient.terminateInstance(this.instanceId, config.project, config.region);
                            console.log('[CloudTraining] Emergency termination succeeded');
                        } catch (emergencyError) {
                            console.error('[CloudTraining] Emergency termination also failed:', emergencyError);
                        }
                    }
                }
            }, maxMs);
            
            console.log(`[CloudTraining] Auto-termination scheduled for ${maxHours} hours (${maxMs}ms from now)`);

            // Step 1.5: Try to associate floating IP (if needed)
            this.sendStatus('Checking for public IP...');
            let instanceIP = null;
            try {
                instanceIP = await this.ensurePublicIP(instance.id, config.project, config.region);
                if (instanceIP) {
                    console.log('[CloudTraining] Successfully got/associated IP:', instanceIP);
                    this.sendLog(`Public IP assigned: ${this.maskIPv4(instanceIP)}`);
                }
            } catch (error) {
                console.warn('[CloudTraining] Could not auto-associate IP:', error.message);
                this.sendLog('Note: Could not auto-assign IP. You may need to manually associate a public IP in CanopyWave dashboard.', 'warning');
            }

            // Step 2: Wait for instance to be ready with IP
            this.sendStatus('Waiting for instance to be ready and get IP address...');
            this.sendLog('If instance is taking long, check CanopyWave dashboard and manually associate a public IP if needed.');
            const finalIP = await this.waitForInstanceReady(instance.id, config.project, config.region, instanceIP);
            
            // Save IP for later use
            this.instanceIP = finalIP;

            // Step 3: Connect via SSH (use finalIP, not instanceIP)
            this.sendStatus('Connecting to instance via SSH...');
            await this.connectSSH(finalIP, config.password);

            // Step 4: Setup environment on remote instance
            this.sendStatus('Setting up training environment...');
            await this.setupRemoteEnvironment();

            // Step 5: Upload dataset and training scripts
            this.sendStatus('Uploading dataset and training scripts...');
            await this.uploadTrainingFiles(config.datasetPath, config.trainingSettings);

            // Step 6: Start training and stream progress
            this.sendStatus('Starting training...');
            const trainingResult = await this.executeRemoteTraining(config.trainingSettings);

            // Check if training actually succeeded before trying to download
            if (!trainingResult || !trainingResult.success) {
                throw new Error(`Training failed: ${trainingResult?.error || 'Unknown error'}`);
            }

            // Write run summary artifact for debugging (before download/terminate)
            try {
                await this.writeRunSummary(config, trainingResult);
            } catch (summaryError) {
                // Don't fail the whole job if summary write fails
                console.warn('[CloudTraining] Failed to write run summary:', summaryError.message);
            }

            // Step 7: Download trained model
            this.sendStatus('Downloading trained model...');
            const modelPath = await this.downloadModel(config.trainingSettings);

            // Step 8: Cleanup - terminate instance (CRITICAL: Always terminate to prevent charges)
            this.sendStatus('Cleaning up cloud resources...');
            // Clear timeout since we're terminating normally
            if (this.maxTrainingTimeout) {
                clearTimeout(this.maxTrainingTimeout);
                this.maxTrainingTimeout = null;
            }
            await this.terminateInstance();

            this.isTraining = false;
            return {
                success: true,
                modelPath: modelPath,
                trainingResult: trainingResult
            };

        } catch (error) {
            console.error('[CloudTraining] Error:', error);
            console.error('[CloudTraining] Error stack:', error.stack);
            
            // Check for billing/payment errors
            const errorMessage = error.message || '';
            const isBillingError = error.statusCode === 402 || 
                errorMessage.toLowerCase().includes('payment') || 
                errorMessage.toLowerCase().includes('billing') || 
                errorMessage.toLowerCase().includes('authorization') ||
                errorMessage.toLowerCase().includes('insufficient funds') ||
                errorMessage.toLowerCase().includes('credit');
            
            if (isBillingError) {
                const billingErrorMsg = 'Cloud job failed to start due to billing authorization.\n' +
                    'Selected GPU requires a 24-hour upfront charge.\n' +
                    'Please update your payment method or choose a lower-cost GPU.';
                this.sendLog(billingErrorMsg, 'error');
                console.error('[CloudTraining] Billing authorization error detected');
            }
            
            // Log full CanopyWave API error details if available
            if (error.apiResponse) {
                console.error('[CloudTraining] CanopyWave API Error Details:', JSON.stringify(error.apiResponse, null, 2));
                if (!isBillingError) {
                    this.sendLog(`CanopyWave API Error (${error.statusCode || 'unknown'}):\n${JSON.stringify(error.apiResponse, null, 2)}`, 'error');
                }
            } else if (error.statusCode) {
                console.error('[CloudTraining] HTTP Status:', error.statusCode);
                console.error('[CloudTraining] Endpoint:', error.endpoint || 'unknown');
                if (error.rawResponse) {
                    console.error('[CloudTraining] Raw Response:', error.rawResponse);
                    if (!isBillingError) {
                        this.sendLog(`CanopyWave API Error (${error.statusCode}):\n${error.rawResponse.substring(0, 500)}`, 'error');
                    }
                }
            } else if (error.originalError) {
                console.error('[CloudTraining] Original Error:', error.originalError);
                if (!isBillingError) {
                    this.sendLog(`CanopyWave API Error:\n${error.message}\n\nOriginal: ${error.originalError.message || error.originalError}`, 'error');
                }
            }
            
            this.isTraining = false;
            
            // Write run summary even on failure (for debugging)
            try {
                if (this.sshConnection && this.instanceId) {
                    await this.writeRunSummary(this.cloudConfig || {}, { success: false, error: error.message });
                }
            } catch (summaryError) {
                console.warn('[CloudTraining] Failed to write run summary on error:', summaryError.message);
            }
            
            // CRITICAL: Always terminate instance on error to prevent charges
            // Clear timeout first
            if (this.maxTrainingTimeout) {
                clearTimeout(this.maxTrainingTimeout);
                this.maxTrainingTimeout = null;
            }
            
            // Cleanup on error - MUST terminate instance
            try {
                if (this.sshConnection) {
                    this.sshConnection.close();
                }
                if (this.instanceId) {
                    console.warn('[CloudTraining] Terminating instance due to error to prevent charges');
                    await this.terminateInstance();
                }
            } catch (cleanupError) {
                console.error('[CloudTraining] Cleanup error:', cleanupError);
                // If terminateInstance fails, it will have already tried retry logic
                // But we should still log the instance ID for manual termination
                if (this.instanceId) {
                    const shortId = (this.instanceId || '').toString().slice(0, 8);
                    this.sendLog(`⚠️ WARNING: Instance (id: ${shortId}…) may still be running. Check CanopyWave dashboard and terminate manually if needed.`, 'error');
                }
            }

            throw error;
        }
    }

    /**
     * FEATURE 1: Instance IP Retrieval After Launch
     * Launches instance and retrieves its IP address
     */
    async launchInstance(config) {
        const instanceConfig = {
            project: config.project,
            region: config.region,
            name: config.name || `unitrainer-${Date.now()}`,
            flavor: config.flavor,
            image: config.image,
            password: config.password,
            is_monitoring: true
        };

        console.log('[CloudTraining] Launching instance with config:', instanceConfig);
        
        try {
            const instance = await this.client.launchInstance(instanceConfig);
            console.log('[CloudTraining] Instance launched successfully:', instance);
            return instance;
        } catch (error) {
            console.error('[CloudTraining] Failed to launch instance:', error.message);
            console.error('[CloudTraining] Error details:', error);
            
            // Provide specific error messages for common issues
            let errorMessage = error.message;
            
            if (errorMessage.includes('Project network not found')) {
                errorMessage = `Project network not found. Please ensure your CanopyWave project "${config.project}" is properly configured:\n\n` +
                    `1. Go to CanopyWave Dashboard (https://cloud.canopywave.io)\n` +
                    `2. Navigate to your project: "${config.project}"\n` +
                    `3. Check "Networks" section and ensure a network exists\n` +
                    `4. If no network exists, create one or contact CanopyWave support\n` +
                    `5. Verify your API key has permission to launch instances\n\n` +
                    `Alternative: Try selecting a different project in the Cloud Configuration.`;
            } else if (errorMessage.includes('no default payment method')) {
                errorMessage = 'No payment method configured. Please add a payment method in your CanopyWave dashboard before launching instances.';
            } else if (error.statusCode === 402 || errorMessage.toLowerCase().includes('payment') || 
                       errorMessage.toLowerCase().includes('billing') || 
                       errorMessage.toLowerCase().includes('authorization') ||
                       errorMessage.toLowerCase().includes('insufficient funds') ||
                       errorMessage.toLowerCase().includes('credit')) {
                // Billing/payment authorization error
                errorMessage = 'Cloud job failed to start due to billing authorization.\n' +
                    'Selected GPU requires a 24-hour upfront charge.\n' +
                    'Please update your payment method or choose a lower-cost GPU.';
                this.sendLog(errorMessage, 'error');
            } else if (errorMessage.includes('403') || errorMessage.toLowerCase().includes('forbidden')) {
                errorMessage = `Access forbidden. Your API key may not have permission to launch instances in project "${config.project}". Check your CanopyWave account permissions.`;
            } else if (errorMessage.includes('insufficient') || errorMessage.toLowerCase().includes('quota')) {
                errorMessage = 'Insufficient quota or resources. Check your CanopyWave account limits and available balance.';
            } else if (errorMessage.includes('flavor') || errorMessage.includes('not available')) {
                errorMessage = `The selected GPU type "${config.flavor}" may not be available in region "${config.region}". Try a different GPU or region.`;
            }
            
            throw new Error(`Failed to launch instance: ${errorMessage}`);
        }
    }

    /**
     * Ensure instance has a public IP (associate floating IP if needed)
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<string>} Public IP address
     */
    /**
     * Ensure instance has a public IP using CanopyWave's 3-step process:
     * 1. Check if instance already has an IP
     * 2. If not, allocate a new public IP (or find an available one)
     * 3. Associate the IP with the instance
     * 
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<string|null>} Public IP address or null if failed
     */
    async ensurePublicIP(instanceId, project, region) {
        try {
            console.log('[CloudTraining] Ensuring instance has public IP...');
            
            // Step 1: Check if instance already has an IP
            const instanceDetails = await this.client.getInstance(instanceId, project, region);
            const existingIP = instanceDetails.publicIp 
                || instanceDetails.public_ip 
                || instanceDetails.accessIPv4;
            
            if (existingIP && existingIP !== 'null' && existingIP !== '') {
                console.log('[CloudTraining] Instance already has IP:', existingIP);
                return existingIP;
            }

            console.log('[CloudTraining] No IP found, will allocate and associate one...');
            this.sendStatus('Allocating public IP address...');

            // Step 2: List existing public IPs to find an available one
            const publicIPsResponse = await this.client.listPublicIPs(project, region);
            const publicIPs = publicIPsResponse.data || publicIPsResponse;
            console.log('[CloudTraining] Existing public IPs:', publicIPs);

            // Find an unassociated IP (status: "DOWN" and no server)
            let availableIP = null;
            if (Array.isArray(publicIPs)) {
                availableIP = publicIPs.find(ip => 
                    !ip.server && !ip.serverId && ip.status === 'DOWN'
                );
            }

            // If no available IP, allocate a new one (API #22)
            if (!availableIP) {
                console.log('[CloudTraining] No available public IP, allocating new one...');
                this.sendStatus('Allocating new public IP address...');
                const allocateResponse = await this.client.allocatePublicIP(project, region);
                availableIP = allocateResponse.data || allocateResponse;
                console.log('[CloudTraining] Allocated new public IP:', availableIP);
            } else {
                console.log('[CloudTraining] Found available public IP:', availableIP);
            }

            // Step 3: Associate the public IP with the instance (API #23)
            const ipId = availableIP.id;
            const ipAddress = availableIP.ip;
            
            console.log('[CloudTraining] Associating IP', ipAddress, '(ID:', ipId, ') with instance', instanceId);
            this.sendStatus(`Associating IP ${this.maskIPv4(ipAddress)} to instance...`);
            
            const associateResponse = await this.client.associatePublicIP(ipId, instanceId, project, region);
            console.log('[CloudTraining] Public IP associated successfully:', associateResponse);
            this.sendLog(`✓ Public IP ${this.maskIPv4(ipAddress)} associated successfully`);
            
            return ipAddress;

        } catch (error) {
            console.error('[CloudTraining] Error ensuring public IP:', error);
            this.sendLog(`Warning: Could not auto-associate IP: ${error.message}`, 'warning');
            
            // Don't fail completely - provide manual instructions
            this.sendLog('You may need to manually associate a public IP in CanopyWave dashboard:', 'warning');
            this.sendLog('1. Go to https://cloud.canopywave.io', 'info');
            this.sendLog('2. Navigate to "Public IPs" section', 'info');
            this.sendLog('3. Click "Associate" next to an available IP', 'info');
            const shortId = (instanceId || '').toString().slice(0, 8);
            this.sendLog(`4. Select instance: ${shortId}…`, 'info');
            
            return null;
        }
    }

    /**
     * FEATURE 1 (continued): Wait for instance to be ready and get IP
     * Polls instance status until it's ACTIVE and has an IP address
     */
    async waitForInstanceReady(instanceId, project, region, knownIP = null, maxWaitTime = 900000) {
        const startTime = Date.now();
        const pollInterval = 5000; // Check every 5 seconds
        let attemptCount = 0;
        let hasShownManualInstructions = false;

        // If we already have an IP, just wait for SSH
        if (knownIP) {
            console.log('[CloudTraining] Using known IP:', knownIP);
            this.sendStatus(`Waiting for instance to be ready at ${this.maskIPv4(knownIP)}...`);
            
            // Wait a bit for instance to fully start
            await this.sleep(10000); // 10 seconds
            
            // Wait for SSH to be ready
            this.sendStatus('Waiting for SSH to be ready...');
            const sshReady = await this.waitForSSHPort(knownIP, 22, 120000); // 2 min max
            if (sshReady) {
                return knownIP;
            }
            // If SSH not ready, fall through to polling
        }

        this.sendStatus('Waiting for instance to start and get IP address...');
        this.sendLog('⏳ This may take 1-3 minutes. Please be patient...');

        while (Date.now() - startTime < maxWaitTime) {
            try {
                attemptCount++;
                const instanceDetails = await this.client.getInstance(instanceId, project, region);
                
                // Log full instance details for debugging
                console.log('[CloudTraining] Instance check #' + attemptCount);
                console.log('[CloudTraining] Instance status:', instanceDetails.status);
                console.log('[CloudTraining] Full instance data:', JSON.stringify(instanceDetails, null, 2));

                // Check if instance is active and has IP
                if (instanceDetails.status === 'ACTIVE' || instanceDetails.status === 'active') {
                    // Try to get IP address from various possible fields
                    const ip = knownIP
                        || instanceDetails.ip 
                        || instanceDetails.floating_ip 
                        || instanceDetails.public_ip
                        || instanceDetails.accessIPv4
                        || instanceDetails.access_ip
                        || instanceDetails.ipv4
                        || (instanceDetails.addresses && this.extractIPFromAddresses(instanceDetails.addresses))
                        || (instanceDetails.networks && this.extractIPFromNetworks(instanceDetails.networks));

                    if (ip) {
                        console.log('[CloudTraining] Instance ready with IP:', ip);
                        this.sendStatus(`Instance ready at ${this.maskIPv4(ip)}`);
                        
                        // REFINEMENT 1: TCP Ping on port 22 instead of blind wait
                        this.sendStatus('Waiting for SSH to be ready...');
                        const sshReady = await this.waitForSSHPort(ip, 22, 120000); // 2 min max
                        if (sshReady) {
                            return ip;
                        } else {
                            console.warn('[CloudTraining] SSH port not ready, continuing to poll...');
                        }
                    } else {
                        // Instance is ACTIVE but no IP yet
                        console.log('[CloudTraining] Instance is ACTIVE but no IP address found yet. Continuing to wait...');
                        this.sendStatus(`Instance active, waiting for IP address... (attempt ${attemptCount})`);
                        
                        // After 30 seconds (6 attempts), show manual instructions
                        if (attemptCount === 6 && !hasShownManualInstructions) {
                            hasShownManualInstructions = true;
                            this.sendLog('⚠️ Instance is active but no IP detected yet.', 'warning');
                            this.sendLog('📋 Manual step needed:', 'warning');
                            this.sendLog('1. Go to CanopyWave Dashboard (https://cloud.canopywave.io)', 'warning');
                            this.sendLog('2. Find your instance (unitrainer-' + instanceId.substring(0, 8) + '...)', 'warning');
                            this.sendLog('3. Click "Associate a Public IP" or "Public IP" button', 'warning');
                            this.sendLog('4. Select or create a floating IP and associate it', 'warning');
                            this.sendLog('5. Training will continue automatically once IP is assigned', 'warning');
                            this.sendLog('⏳ Continuing to wait for IP...', 'warning');
                        }
                    }
                } else {
                    // Instance not active yet
                    this.sendStatus(`Instance starting... Status: ${instanceDetails.status} (attempt ${attemptCount})`);
                }

                // Wait before next poll
                await this.sleep(pollInterval);

            } catch (error) {
                console.error('[CloudTraining] Error checking instance status:', error);
                this.sendStatus(`Checking instance status... (attempt ${attemptCount})`);
                // Continue polling unless max wait time exceeded
                await this.sleep(pollInterval);
            }
        }

        throw new Error(`Timeout waiting for instance to be ready. The instance may need more time to start, or there may be an issue with IP assignment. Check CanopyWave dashboard for instance status.`);
    }

    /**
     * Extract IP from complex addresses object
     */
    extractIPFromAddresses(addresses) {
        // addresses might be like: { "network-name": [{ "addr": "1.2.3.4", "version": 4 }] }
        for (const network in addresses) {
            const addrs = addresses[network];
            if (Array.isArray(addrs)) {
                for (const addr of addrs) {
                    if (addr.version === 4 && addr.addr) {
                        return addr.addr;
                    }
                }
            }
        }
        return null;
    }

    /**
     * Extract IP from networks object (alternative format)
     */
    extractIPFromNetworks(networks) {
        // networks might be an array or object
        if (Array.isArray(networks)) {
            for (const network of networks) {
                if (network.ip || network.addr || network.address) {
                    return network.ip || network.addr || network.address;
                }
            }
        } else if (typeof networks === 'object') {
            // Try common field names
            for (const key in networks) {
                const network = networks[key];
                if (typeof network === 'string') {
                    // Direct IP string
                    return network;
                } else if (network && (network.ip || network.addr || network.address)) {
                    return network.ip || network.addr || network.address;
                }
            }
        }
        return null;
    }

    /**
     * REFINEMENT 1: TCP Ping - Wait for SSH port to be accepting connections
     * More reliable than blind 30-second wait
     */
    async waitForSSHPort(ip, port, maxWaitTime = 120000) {
        const net = require('net');
        const startTime = Date.now();
        const retryInterval = 3000; // Check every 3 seconds

        console.log(`[CloudTraining] Waiting for SSH port ${port} on ${ip}...`);

        while (Date.now() - startTime < maxWaitTime) {
            try {
                const isOpen = await new Promise((resolve) => {
                    const socket = new net.Socket();
                    const timeout = 5000; // 5 second connection timeout

                    socket.setTimeout(timeout);
                    
                    socket.on('connect', () => {
                        socket.destroy();
                        resolve(true);
                    });

                    socket.on('timeout', () => {
                        socket.destroy();
                        resolve(false);
                    });

                    socket.on('error', () => {
                        socket.destroy();
                        resolve(false);
                    });

                    socket.connect(port, ip);
                });

                if (isOpen) {
                    console.log(`[CloudTraining] SSH port ${port} is accepting connections`);
                    this.sendStatus('SSH service ready');
                    return true;
                }

            } catch (error) {
                // Continue trying
            }

            await this.sleep(retryInterval);
        }

        console.warn(`[CloudTraining] SSH port ${port} not ready after ${maxWaitTime}ms`);
        return false;
    }

    /**
     * FEATURE 1 (continued): Connect to instance via SSH
     */
    async connectSSH(ip, password, username = 'ubuntu', maxRetries = 5) {
        let lastError = null;

        for (let i = 0; i < maxRetries; i++) {
            try {
                console.log(`[CloudTraining] Attempting SSH connection (attempt ${i + 1}/${maxRetries})...`);
                
                this.sshConnection = new CloudSSHUtils(ip, username, password);
                await this.sshConnection.connect();
                
                console.log('[CloudTraining] SSH connection established');
                return;

            } catch (error) {
                lastError = error;
                console.error(`[CloudTraining] SSH connection attempt ${i + 1} failed:`, error.message);
                
                if (i < maxRetries - 1) {
                    // Wait 10 seconds before retry
                    await this.sleep(10000);
                }
            }
        }

        throw new Error(`Failed to connect via SSH after ${maxRetries} attempts: ${lastError.message}`);
    }

    /**
     * FEATURE 2: Remote Training Script Execution
     * Setup Python environment and dependencies on remote instance
     */
    async setupRemoteEnvironment() {
        console.log('[CloudTraining] Setting up remote environment...');

        // Update system and install Python dependencies
        const commands = [
            // Update package lists (critical - must succeed before install)
            'sudo apt-get update',
            
            // Enable universe repository if available (needed for some packages on Ubuntu)
            'sudo apt-get install -y software-properties-common 2>/dev/null || true',
            'sudo add-apt-repository universe -y 2>/dev/null || true',
            'sudo apt-get update 2>/dev/null || true',
            
            // Install core packages (python3-pip may not exist as package on some Ubuntu versions)
            // We'll install pip separately using ensurepip or get-pip.py
            'sudo apt-get install -y python3 python3-venv python3-distutils python3-dev curl',
            
            // Install tmux (usually available, but not critical - we have nohup fallback)
            'sudo apt-get install -y tmux || echo "tmux install failed, will use nohup fallback"',
            
            // Install unzip (CRITICAL - needed for dataset extraction)
            // Try multiple methods: check if installed, try apt-get, try alternative package names
            'which unzip && echo "unzip already installed" || (sudo apt-get install -y unzip || sudo apt-get install -y zip unzip || (echo "ERROR: unzip installation failed" && exit 1))',
            
            // Install pip using ensurepip (works on Python 3.4+)
            // If ensurepip fails, try get-pip.py as fallback
            'python3 -m ensurepip --upgrade 2>/dev/null || python3 -m ensurepip 2>/dev/null || (curl -sS https://bootstrap.pypa.io/get-pip.py | python3)',
            
            // Verify pip is available (use python3 -m pip for reliability)
            'python3 -m pip --version || (echo "ERROR: pip not available" && exit 1)',
            
            // Install PyTorch with CUDA support (use cu121 for H100 compatibility)
            // Use python3 -m pip instead of pip3 for reliability across different Ubuntu versions
            // First upgrade pip, then install CUDA wheels (not CPU-only wheels from default PyPI)
            // Use --no-cache-dir for faster installs and less disk usage
            'python3 -m pip install --upgrade pip --no-cache-dir',
            
            // Install PyTorch with CUDA support if not already installed
            // Install PyTorch with retry/timeout flags and longer timeout (handled in loop below)
            'python3 -m pip uninstall -y torch torchvision torchaudio || true',
            'python3 -m pip install --no-cache-dir --retries 10 --timeout 120 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
            
            // Install Ultralytics YOLO
            'python3 -m pip install --no-cache-dir ultralytics',
            
            // Install other ML libraries
            'python3 -m pip install --no-cache-dir scikit-learn xgboost lightgbm pandas numpy pillow',
            
            // Create working directory
            'mkdir -p ~/training',
            'mkdir -p ~/training/dataset',
            'mkdir -p ~/training/output'
        ];

        // Check if PyTorch CUDA is already installed before reinstalling
        let torchAlreadyInstalled = false;
        try {
            this.sendStatus('Checking if PyTorch CUDA is already installed...');
            const torchCheckCmd = 'python3 -c "import torch; assert torch.cuda.is_available(); print(\'torch_cuda_ok\')" 2>/dev/null || echo "torch_not_installed_or_no_cuda"';
            const torchCheckResult = await this.sshConnection.executeCommand(torchCheckCmd, 10000);
            if (torchCheckResult.stdout && torchCheckResult.stdout.includes('torch_cuda_ok')) {
                torchAlreadyInstalled = true;
                console.log('[CloudTraining] ✓ PyTorch CUDA already installed and working, skipping reinstall');
                this.sendLog('✓ PyTorch CUDA already installed, skipping reinstall', 'success');
            }
        } catch (e) {
            console.log('[CloudTraining] Torch check failed or not installed, proceeding with install...');
        }

        for (const cmd of commands) {
            try {
                // Skip torch uninstall/install if already installed and working
                if (torchAlreadyInstalled && cmd.includes('torch') && (cmd.includes('uninstall') || (cmd.includes('pip') && cmd.includes('install') && cmd.includes('torch')))) {
                    console.log(`[CloudTraining] Skipping command (torch already installed): ${cmd}`);
                    continue;
                }
                
                console.log(`[CloudTraining] Executing: ${cmd}`);
                
                // Detect torch install command and use longer timeout (30 minutes)
                // Also check for pip installs that might take longer (ultralytics, etc.)
                const isTorchInstall = (cmd.includes('pip') && cmd.includes('torch') && cmd.includes('download.pytorch.org')) ||
                                      (cmd.includes('pip') && cmd.includes('torch') && cmd.includes('install'));
                const isLargeMLInstall = cmd.includes('pip') && (cmd.includes('ultralytics') || cmd.includes('lightgbm') || cmd.includes('xgboost'));
                const timeout = isTorchInstall ? 1800000 : (isLargeMLInstall ? 900000 : 300000); // 30 min for torch, 15 min for large ML packages, 5 min otherwise
                
                const result = await this.sshConnection.executeCommand(cmd, timeout);
                
                if (result.code !== 0) {
                    // Critical commands must succeed (foundational tools + PyTorch/Ultralytics)
                    const isCritical = 
                        cmd.startsWith('sudo apt-get update') ||
                        (cmd.startsWith('sudo apt-get install') && (cmd.includes('unzip') || cmd.includes('python3'))) ||
                        (cmd.includes('pip --version') && (cmd.includes('ERROR') || result.code !== 0)) ||
                        ((cmd.includes('python3 -m pip install') || cmd.includes('pip3 install')) && (cmd.includes('torch') || cmd.includes('ultralytics')));
                    
                    if (isCritical) {
                        // Include both stdout and stderr for better debugging
                        const errorDetails = [
                            `Command: ${cmd}`,
                            `Exit code: ${result.code}`,
                            `stdout: ${result.stdout || '(empty)'}`,
                            `stderr: ${result.stderr || '(empty)'}`
                        ].join('\n');
                        
                        const errorMsg = `Critical command failed:\n${errorDetails}`;
                        console.error(`[CloudTraining] ${errorMsg}`);
                        this.sendLog(errorMsg, 'error');
                        throw new Error(`Failed to install required dependencies: ${cmd}\n\n${errorDetails}`);
                    } else {
                        console.warn(`[CloudTraining] Command warning (code ${result.code}):`, cmd);
                        console.warn('stdout:', result.stdout);
                        console.warn('stderr:', result.stderr);
                    }
                } else {
                    // Log success for critical commands
                    if (cmd.startsWith('sudo apt-get install') || cmd.includes('pip3 install')) {
                        console.log(`[CloudTraining] ✓ Command succeeded: ${cmd}`);
                        if (result.stdout) {
                            console.log(`[CloudTraining] Output: ${result.stdout.substring(0, 200)}...`);
                        }
                    }
                }
            } catch (error) {
                // If it's a critical command error, re-throw it with full details
                if (error.message && error.message.includes('Failed to install required dependencies')) {
                    throw error;
                }
                // For non-critical commands, check if it's a timeout or connection error
                if (error.message && (error.message.includes('timeout') || error.message.includes('Connection'))) {
                    console.error(`[CloudTraining] Connection/timeout error for: ${cmd}`, error);
                    // Re-throw connection errors as they're critical
                    throw new Error(`Connection error while executing: ${cmd}\n${error.message}`);
                }
                console.error(`[CloudTraining] Command failed: ${cmd}`, error);
                // Continue with other non-critical commands
            }
        }

        // Preflight check: Verify PyTorch CUDA and Ultralytics are working
        // CRITICAL: Fail early if CUDA is not available (don't waste money on CPU training)
        this.sendStatus('Verifying PyTorch CUDA and Ultralytics...');
        try {
            // Check Python version, PyTorch CUDA, Ultralytics, and GPU
            // Remove || true to get proper error output if commands fail
            const preflightCheck = await this.sshConnection.executeCommand(
                'python3 -c "import sys; print(\'python\', sys.version); import torch; print(\'torch\', torch.__version__, \'cuda\', torch.cuda.is_available()); import ultralytics; print(\'ultralytics ok\')" && nvidia-smi --query-gpu=name --format=csv,noheader | head -1',
                30000
            );
            console.log('[CloudTraining] Preflight check output:', preflightCheck.stdout);
            console.log('[CloudTraining] Preflight check stderr:', preflightCheck.stderr);
            console.log('[CloudTraining] Preflight check exit code:', preflightCheck.code);
            this.sendLog(`Preflight check:\n${preflightCheck.stdout}`);
            
            // Check if command failed (non-zero exit code)
            if (preflightCheck.code !== 0) {
                const errorMsg = `Preflight check command failed (exit code ${preflightCheck.code}).\n\nOutput: ${preflightCheck.stdout || 'No output'}\nErrors: ${preflightCheck.stderr || 'No errors'}\n\nCannot proceed with training.`;
                console.error('[CloudTraining] CRITICAL: Preflight command failed');
                this.sendLog(errorMsg, 'error');
                throw new Error(errorMsg);
            }
            
            if (preflightCheck.stdout && preflightCheck.stdout.includes('cuda True')) {
                this.sendLog('✓ PyTorch CUDA is available', 'success');
            } else {
                // CUDA not available - this is a critical failure for GPU training
                const errorMsg = `PyTorch CUDA is not available. Training requires GPU support.\n\nPreflight output: ${preflightCheck.stdout || 'No output'}\nPreflight errors: ${preflightCheck.stderr || 'No errors'}\n\nPlease check PyTorch installation or contact support.`;
                console.error('[CloudTraining] CRITICAL: CUDA not available');
                this.sendLog(errorMsg, 'error');
                throw new Error(errorMsg);
            }
        } catch (preflightError) {
            console.error('[CloudTraining] Preflight check failed:', preflightError);
            // If executeCommand failed, it might have stderr in the error object
            // Also check if we got a result but with non-zero exit code
            let errorDetails = '';
            if (preflightError.stderr) {
                errorDetails = `\n\nstderr: ${preflightError.stderr}`;
            } else if (preflightError.stdout) {
                errorDetails = `\n\noutput: ${preflightError.stdout}`;
            }
            const errorMsg = `Preflight check failed: ${preflightError.message}${errorDetails}. Cannot proceed with training.`;
            throw new Error(errorMsg);
        }

        console.log('[CloudTraining] Remote environment setup complete');
    }

    /**
     * FEATURE 2 (continued): Upload dataset and training scripts
     * REFINEMENT 2: Compress datasets before upload for 80% faster transfer
     */
    async uploadTrainingFiles(datasetPath, trainingSettings) {
        console.log('[CloudTraining] Uploading training files...');

        // Upload dataset
        if (datasetPath && fs.existsSync(datasetPath)) {
            const stat = fs.statSync(datasetPath);
            
            if (stat.isDirectory()) {
                // REFINEMENT 2: Compress directory before upload
                this.sendStatus('Compressing dataset for faster upload...');
                const zipPath = await this.compressDirectory(datasetPath);
                
                try {
                    this.sendStatus('Uploading compressed dataset...');
                    await this.sshConnection.uploadFile(zipPath, '~/training/dataset.zip');
                    
                    // Extract on remote
                    this.sendStatus('Extracting dataset on remote instance...');
                    // Extract to temp location first, then move contents to dataset directory
                    // This handles the case where zip contains a parent directory (e.g., PEOPLE_yolo/)
                    const extractCmd = `
                        cd ~/training && 
                        rm -rf dataset_temp && 
                        mkdir -p dataset_temp && 
                        unzip -q dataset.zip -d dataset_temp && 
                        # Find the actual dataset contents (may be in a subdirectory)
                        if [ -d dataset_temp/*/ ]; then
                            # Zip contains a parent directory, move its contents
                            SUBDIR=$(ls -d dataset_temp/*/ | head -1)
                            echo "Found subdirectory: $SUBDIR"
                            # Move all contents including hidden files
                            shopt -s dotglob
                            mv "$SUBDIR"* dataset/ 2>/dev/null || true
                            shopt -u dotglob
                        else
                            # Zip contents are at root level
                            echo "No subdirectory found, moving root contents"
                            shopt -s dotglob
                            mv dataset_temp/* dataset/ 2>/dev/null || true
                            shopt -u dotglob
                        fi && 
                        rm -rf dataset_temp && 
                        rm dataset.zip &&
                        # Verify data.yaml exists and show directory structure
                        echo "=== Dataset directory contents ===" && 
                        ls -la dataset/ | head -20 && 
                        echo "=== Checking for data.yaml ===" && 
                        (test -f dataset/data.yaml && echo "data.yaml FOUND" || echo "data.yaml MISSING")
                    `;
                    const extractResult = await this.sshConnection.executeCommand(extractCmd, 600000); // 10 min timeout
                    console.log('[CloudTraining] Extraction output:', extractResult.stdout);
                    this.sendLog(`Extraction result:\n${extractResult.stdout}`);
                    
                } finally {
                    // Cleanup local zip
                    if (fs.existsSync(zipPath)) {
                        fs.unlinkSync(zipPath);
                    }
                }
            } else {
                // Upload single file
                this.sendStatus('Uploading dataset file...');
                const filename = path.basename(datasetPath);
                await this.sshConnection.uploadFile(datasetPath, `~/training/dataset/${filename}`);
            }
        }

        // Create training script based on framework
        const trainingScript = this.generateTrainingScript(trainingSettings);
        const localScriptPath = path.join(os.tmpdir(), 'remote_training.py');
        fs.writeFileSync(localScriptPath, trainingScript);

        // Upload training script
        this.sendStatus('Uploading training script...');
        try {
            await this.sshConnection.uploadFile(localScriptPath, '~/training/train.py');
            
            // Verify upload succeeded
            const verifyScript = await this.sshConnection.executeCommand(
                'test -f ~/training/train.py && echo "EXISTS" || echo "MISSING"',
                10000
            );
            
            if (!verifyScript.stdout || !verifyScript.stdout.includes('EXISTS')) {
                throw new Error('Training script upload failed - file not found on remote instance');
            }
            
            // Make script executable
            await this.sshConnection.executeCommand('chmod +x ~/training/train.py', 5000);
            
            console.log('[CloudTraining] Training script uploaded and verified');
            this.sendLog('✓ Training script uploaded successfully');
        } catch (uploadError) {
            console.error('[CloudTraining] Failed to upload training script:', uploadError);
            throw new Error(`Failed to upload training script: ${uploadError.message}`);
        } finally {
            // Cleanup local temp file
            if (fs.existsSync(localScriptPath)) {
                fs.unlinkSync(localScriptPath);
            }
        }
        
        // Verify training directory structure and data.yaml exists
        this.sendStatus('Verifying remote directory structure...');
        const dirCheck = await this.sshConnection.executeCommand(
            'ls -la ~/training/ && echo "---" && ls -la ~/training/dataset/ 2>&1 | head -20',
            15000
        );
        console.log('[CloudTraining] Remote directory contents:', dirCheck.stdout);
        this.sendLog(`Remote directory check:\n${dirCheck.stdout}`);
        
        // Verify data.yaml exists (critical for YOLO training)
        if (trainingSettings.framework === 'yolo') {
            const yamlCheck = await this.sshConnection.executeCommand(
                'test -f ~/training/dataset/data.yaml && echo "EXISTS" || echo "MISSING"',
                10000
            );
            if (!yamlCheck.stdout || !yamlCheck.stdout.includes('EXISTS')) {
                // Try to find data.yaml in subdirectories
                const findYaml = await this.sshConnection.executeCommand(
                    'find ~/training/dataset -name "data.yaml" -type f 2>/dev/null | head -1',
                    10000
                );
                if (findYaml.stdout && findYaml.stdout.trim()) {
                    const yamlPath = findYaml.stdout.trim();
                    this.sendLog(`Found data.yaml at: ${yamlPath}`, 'warning');
                    // Copy to expected location
                    await this.sshConnection.executeCommand(
                        `cp "${yamlPath}" ~/training/dataset/data.yaml`,
                        10000
                    );
                    this.sendLog('Copied data.yaml to expected location', 'success');
                } else {
                    // List what files are actually in the dataset directory for debugging
                    const listFiles = await this.sshConnection.executeCommand(
                        'ls -la ~/training/dataset/ 2>&1',
                        10000
                    );
                    const fileList = listFiles.stdout || 'Unable to list files';
                    throw new Error(`data.yaml not found in dataset after upload. Please ensure your dataset is in YOLO format with data.yaml in the root directory.\n\nFiles found in dataset directory:\n${fileList}\n\nIf you see a subdirectory (e.g., PEOPLE_yolo/), the extraction may have failed. Please check the extraction logs above.`);
                }
            } else {
                this.sendLog('✓ data.yaml found at expected location', 'success');
            }
        }

        console.log('[CloudTraining] Training files uploaded');
    }

    /**
     * REFINEMENT 2: Compress directory to zip for faster upload
     * Reduces upload time by up to 80% for datasets with many small files
     */
    async compressDirectory(dirPath) {
        const archiver = require('archiver');
        const zipPath = path.join(os.tmpdir(), `dataset-${Date.now()}.zip`);
        
        return new Promise((resolve, reject) => {
            const output = fs.createWriteStream(zipPath);
            const archive = archiver('zip', { zlib: { level: 6 } }); // Balanced compression

            output.on('close', () => {
                const sizeMB = (archive.pointer() / 1024 / 1024).toFixed(2);
                console.log(`[CloudTraining] Dataset compressed: ${sizeMB} MB`);
                resolve(zipPath);
            });

            archive.on('error', (err) => {
                reject(err);
            });

            archive.pipe(output);
            // Use false to preserve structure but strip parent directory
            // This ensures data.yaml ends up at ~/training/dataset/data.yaml, not ~/training/dataset/PEOPLE_yolo/data.yaml
            archive.directory(dirPath, false);
            archive.finalize();
        });
    }

    /**
     * Generate training script based on framework
     */
    generateTrainingScript(settings) {
        const framework = settings.framework || 'yolo';
        
        if (framework === 'yolo') {
            return this.generateYOLOScript(settings);
        } else if (framework === 'pytorch') {
            return this.generatePyTorchScript(settings);
        } else {
            return this.generateGenericScript(settings);
        }
    }

    /**
     * Generate YOLO training script
     */
    generateYOLOScript(settings) {
        // Extract settings with defaults
        const modelVariant = settings.modelVariant || 'yolov11n';
        const epochs = settings.epochs || 10;
        const batchSize = settings.batchSize || 16;
        const imageSize = settings.imageSize || 640;
        
        return `#!/usr/bin/env python3
"""
Remote YOLO Training Script
Generated by Uni Trainer
"""

import sys
import json
import os
from ultralytics import YOLO
import traceback

def send_progress(epoch, total_epochs, metrics):
    """Send progress to main process"""
    progress_data = {
        "type": "progress",
        "data": {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "metrics": metrics,
            "status": "training"
        }
    }
    print(json.dumps(progress_data))
    sys.stdout.flush()

def main():
    try:
        # Check dataset exists
        data_yaml = os.path.expanduser("~/training/dataset/data.yaml")
        dataset_dir = os.path.expanduser("~/training/dataset")
        
        print(f"Checking dataset at: {dataset_dir}")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        if not os.path.exists(data_yaml):
            # List what's actually in the dataset directory
            files = os.listdir(dataset_dir)
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Files in dataset directory: {files}")
        
        print(f"Dataset found: {data_yaml}")
        
        # Initialize model
        model_variant = "${modelVariant}"
        print(f"Loading model: {model_variant}")
        model = YOLO(f"{model_variant}.pt")
        
        # Training parameters
        epochs = ${epochs}
        batch_size = ${batchSize}
        imgsz = ${imageSize}
        
        # Start training
        # Ultralytics will print progress to stdout automatically with verbose=True
        print(f"Starting training: epochs={epochs}, batch={batch_size}, imgsz={imgsz}")
        print(f"Using data.yaml: {data_yaml}")
        print("Training output will be streamed in real-time...")
        sys.stdout.flush()
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=0,  # Use GPU 0
            project=os.path.expanduser("~/training/output"),
            name="training_run",
            verbose=True  # This enables progress output to stdout
        )
        
        # Output the actual save directory for deterministic model location FIRST
        # (before "complete" so Node.js can capture it before download step)
        try:
            save_dir = getattr(results, "save_dir", None)
            if save_dir:
                print(json.dumps({"type": "artifact", "data": {"save_dir": str(save_dir)}}))
                sys.stdout.flush()
        except Exception as e:
            print(f"Warning: Could not extract save_dir: {e}", file=sys.stderr)
        
        # Write completion flag for deterministic detection (especially for tmux mode)
        try:
            flag_path = os.path.expanduser("~/training/done.flag")
            with open(flag_path, 'w') as f:
                f.write("training_completed")
        except Exception as e:
            print(f"Warning: Could not write done flag: {e}", file=sys.stderr)
        
        print("Training completed successfully!")
        print(json.dumps({"type": "complete", "data": {"status": "success"}}))
        
    except Exception as e:
        # Write error flag for deterministic detection
        try:
            flag_path = os.path.expanduser("~/training/error.flag")
            with open(flag_path, 'w') as f:
                f.write(f"training_failed: {str(e)}")
        except Exception:
            pass
        
        error_msg = f"Training error: {str(e)}"
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"type": "error", "data": {"error": str(e)}}))
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
`;
    }

    /**
     * Generate PyTorch training script
     */
    generatePyTorchScript(settings) {
        return `#!/usr/bin/env python3
"""
Remote PyTorch Training Script
Generated by Uni Trainer
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Training implementation here
print("PyTorch training not yet implemented")
sys.exit(1)
`;
    }

    /**
     * Generate generic training script
     */
    generateGenericScript(settings) {
        return `#!/usr/bin/env python3
print("Generic training not yet implemented")
sys.exit(1)
`;
    }

    /**
     * FEATURE 3: Progress Streaming Over SSH
     * Execute training and stream progress in real-time
     * REFINEMENT 3: Use tmux for persistence (survives SSH disconnects)
     */
    async executeRemoteTraining(settings) {
        console.log('[CloudTraining] Starting remote training execution...');

        // REFINEMENT 3: Start training in tmux session for persistence
        const sessionName = `training-${Date.now()}`;
        let usingTmux = false;
        
        // Clear old completion flags from previous runs
        await this.sshConnection.executeCommand('rm -f ~/training/done.flag ~/training/error.flag', 5000);
        
        // Create tmux session and start training
        const startCommand = `tmux new-session -d -s ${sessionName} 'cd ~/training && python3 train.py 2>&1 | tee training.log'`;
        
        try {
            await this.sshConnection.executeCommand(startCommand);
            // Verify tmux session actually started
            const verifyTmux = await this.sshConnection.executeCommand(
                `tmux has-session -t ${sessionName} 2>/dev/null && echo "OK" || echo "FAIL"`,
                5000
            );
            if (verifyTmux.stdout && verifyTmux.stdout.includes('OK')) {
                usingTmux = true;
                console.log(`[CloudTraining] Training started in tmux session: ${sessionName}`);
                this.sendStatus('Training running in persistent session');
            } else {
                throw new Error('tmux session verification failed');
            }
        } catch (error) {
            console.error('[CloudTraining] Failed to start tmux session, falling back to nohup:', error.message);
            usingTmux = false;
            // Fallback to nohup if tmux not available
            // Clear flags again (in case tmux partially started)
            await this.sshConnection.executeCommand('rm -f ~/training/done.flag ~/training/error.flag', 5000);
            // Write PID to file for reliable process checking
            await this.sshConnection.executeCommand(
                'cd ~/training && nohup python3 train.py > training.log 2>&1 & echo $! > training.pid'
            );
            this.sendStatus('Training running in background');
        }
        
        // Store tmux usage for later checks
        this.usingTmux = usingTmux;
        this.tmuxSessionName = usingTmux ? sessionName : null;

        // Wait a moment for log file to be created
        await this.sleep(3000);
        
        // Verify training process is actually running
        try {
            if (this.usingTmux) {
                const processCheck = await this.sshConnection.executeCommand(
                    `tmux list-panes -t ${sessionName} -F "#{pane_pid}" 2>/dev/null | head -1 || ps aux | grep "python3 train.py" | grep -v grep | awk '{print $2}' | head -1 || echo "NOT_FOUND"`,
                    5000
                );
                if (processCheck.stdout && !processCheck.stdout.includes('NOT_FOUND')) {
                    this.sendLog(`Training process verified (PID: ${processCheck.stdout.trim()})`, 'info');
                } else {
                    this.sendLog('Warning: Training process not found. Checking log file...', 'warning');
                }
            } else {
                const pidCheck = await this.sshConnection.executeCommand(
                    'test -f ~/training/training.pid && cat ~/training/training.pid || echo "NO_PID"',
                    5000
                );
                if (pidCheck.stdout && !pidCheck.stdout.includes('NO_PID')) {
                    const pid = pidCheck.stdout.trim();
                    const processCheck = await this.sshConnection.executeCommand(
                        `kill -0 ${pid} 2>/dev/null && echo "RUNNING" || echo "NOT_RUNNING"`,
                        5000
                    );
                    if (processCheck.stdout && processCheck.stdout.includes('RUNNING')) {
                        this.sendLog(`Training process verified (PID: ${pid})`, 'info');
                    } else {
                        this.sendLog('Warning: Training process PID found but process not running', 'warning');
                    }
                }
            }
        } catch (e) {
            console.warn('[CloudTraining] Process verification failed:', e.message);
        }
        
        // Stream logs from the training session
        return new Promise((resolve, reject) => {
            // First check if log file exists and has content
            let logFileReady = false;
            const initialCheck = setInterval(async () => {
                try {
                    const checkLog = await this.sshConnection.executeCommand('test -f ~/training/training.log && wc -l ~/training/training.log | cut -d" " -f1 || echo "0"', 5000);
                    const lineCount = parseInt(checkLog.stdout || '0');
                    if (lineCount > 0) {
                        logFileReady = true;
                        clearInterval(initialCheck);
                        this.sendLog(`Training log file detected (${lineCount} lines), streaming output...`, 'info');
                        
                        // Show first few lines of log to confirm training started
                        const headLog = await this.sshConnection.executeCommand('head -10 ~/training/training.log', 5000);
                        if (headLog.stdout) {
                            this.sendLog(`Initial log output:\n${headLog.stdout}`, 'info');
                        }
                    } else {
                        // Log file exists but is empty - training might not have started
                        const fileExists = await this.sshConnection.executeCommand('test -f ~/training/training.log && echo "EXISTS" || echo "NOT_EXISTS"', 5000);
                        if (fileExists.stdout && fileExists.stdout.includes('EXISTS')) {
                            this.sendLog('Log file exists but is empty. Training may be starting...', 'info');
                        }
                    }
                } catch (e) {
                    // Continue waiting
                }
            }, 2000);
            
            // Timeout after 30 seconds if log never appears
            setTimeout(() => {
                if (!logFileReady) {
                    clearInterval(initialCheck);
                    this.sendLog('Warning: Training log file not detected. Training may have failed to start.', 'warning');
                }
            }, 30000);
            
            // Store stream reference for cleanup
            let stream = null;
            let streamCleanup = () => {
                if (stream) {
                    try {
                        // Close the stream (ssh2 exec streams don't support signal like local processes)
                        if (stream.end) stream.end();
                    } catch (e) {}
                    try {
                        if (stream.close) stream.close();
                    } catch (e) {}
                    stream = null;
                }
            };
            
            // Tail the log file for real-time updates
            // For nohup mode: use tail --pid to auto-exit when training process ends
            // For tmux mode: use tail with unique marker for reliable cleanup
            let tailCommand;
            if (this.usingTmux) {
                // Use unique marker so pkill only targets our tail process
                tailCommand = `bash -lc 'UNITRAINER_TAIL=1 tail -f ~/training/training.log'`;
            } else {
                // For nohup: guard against PID=0 edge case, use heredoc for readability and quote-safety
                tailCommand = `bash -lc 'set -e
TRAIN_PID=$(cat ~/training/training.pid 2>/dev/null || true)
if [ -n "$TRAIN_PID" ] && [ "$TRAIN_PID" != "0" ]; then
  tail --pid="$TRAIN_PID" -f ~/training/training.log
else
  tail -f ~/training/training.log
fi'`;
            }
            
            this.sshConnection.conn.exec(tailCommand, (err, execStream) => {
                if (err) {
                    clearInterval(initialCheck);
                    reject(err);
                    return;
                }
                
                stream = execStream; // Store for cleanup

                let stdout = '';
                let lastOutputTime = Date.now();
                let ultralyticsSaveDir = null; // Track actual save directory from Ultralytics
                const trainingCheckInterval = setInterval(async () => {
                    try {
                        // Check if training process is still running (handle tmux vs nohup)
                        // For tmux: check flag files for deterministic detection (more reliable than session existence)
                        // For nohup: check PID file and process
                        let checkCmd;
                        if (this.usingTmux) {
                            // Check for completion/error flags first (deterministic), then fall back to session check
                            checkCmd = `if [ -f ~/training/done.flag ] || [ -f ~/training/error.flag ]; then echo "stopped"; elif tmux has-session -t ${this.tmuxSessionName} 2>/dev/null; then echo "running"; else echo "stopped"; fi`;
                        } else {
                            checkCmd = `test -f ~/training/training.pid && kill -0 "$(cat ~/training/training.pid)" 2>/dev/null && echo "running" || echo "stopped"`;
                        }
                        const result = await this.sshConnection.executeCommand(checkCmd, 10000);
                        
                        // Log status periodically for debugging (every 5 checks = ~50 seconds)
                        if (Math.random() < 0.2) { // 20% chance = roughly every 5 checks
                            const status = result.stdout.includes('running') ? 'running' : 'stopped';
                            console.log(`[CloudTraining] Training status check: ${status}`);
                            if (status === 'running' && stdout.length > 0) {
                                const lastLines = stdout.split('\n').slice(-3).join('\n');
                                console.log(`[CloudTraining] Recent output (last 3 lines):\n${lastLines}`);
                            }
                        }
                        
                        if (result.stdout.includes('stopped')) {
                            clearInterval(trainingCheckInterval);
                            clearInterval(initialCheck);
                            if (this.costWarningInterval) {
                                clearInterval(this.costWarningInterval);
                                this.costWarningInterval = null;
                            }
                            
                            // Kill remote tail process (for tmux mode, tail -f keeps running)
                            // Use unique marker to only kill our specific tail process
                            if (this.usingTmux) {
                                try {
                                    await this.sshConnection.executeCommand('pkill -f "UNITRAINER_TAIL=1" || true', 5000);
                                } catch (e) {
                                    // Ignore kill errors
                                }
                            }
                            
                            streamCleanup();
                            
                            // Get final log content - try multiple ways
                            let logContent = '';
                            try {
                                const finalLog = await this.sshConnection.executeCommand('cat ~/training/training.log 2>/dev/null || echo ""');
                                logContent = finalLog.stdout || '';
                                
                                // If log is empty, check if training.py even ran
                                if (!logContent || logContent.trim() === '') {
                                    // Check if script exists and is executable
                                    const scriptCheck = await this.sshConnection.executeCommand('ls -la ~/training/train.py 2>/dev/null && head -5 ~/training/train.py || echo "SCRIPT_NOT_FOUND"');
                                    if (scriptCheck.stdout && scriptCheck.stdout.includes('SCRIPT_NOT_FOUND')) {
                                        throw new Error('Training script (train.py) not found on remote instance. File upload may have failed.');
                                    }
                                    
                                    // Check Python process errors
                                    const pyErrorCheck = await this.sshConnection.executeCommand('ls -la ~/training/*.log 2>/dev/null | head -5 || echo ""');
                                    this.sendLog(`Warning: Training log is empty. Files in training directory:\n${pyErrorCheck.stdout}`, 'warning');
                                    
                                    // Try to get stderr from process
                                    const stderrCheck = await this.sshConnection.executeCommand('cat ~/training/training.log.err 2>/dev/null || python3 ~/training/train.py 2>&1 | head -20 || echo ""');
                                    logContent = stderrCheck.stdout || '';
                                }
                            } catch (logError) {
                                console.error('[CloudTraining] Failed to read training log:', logError);
                                logContent = `Failed to read log: ${logError.message}`;
                            }
                            
                            stdout = logContent;
                            
                            // Check if training actually completed successfully (whitespace-tolerant JSON detection)
                            const trainingSucceeded = 
                                stdout.includes('Training completed successfully') || 
                                stdout.toLowerCase().includes('training completed') ||
                                /"type"\s*:\s*"complete"/.test(stdout);
                            
                            const trainingFailed = 
                                stdout.includes('Training error:') ||
                                stdout.includes('Training failed') ||
                                /"type"\s*:\s*"error"/.test(stdout) ||
                                /Traceback/i.test(stdout) ||
                                stdout.includes('Exception:');
                            
                            // Determine exit code based on log markers (simpler and more reliable)
                            // Don't try to infer from tmux session state - session is already stopped when we get here
                            const exitCode = trainingFailed ? 1 : 0;
                            
                            console.log('[CloudTraining] Training session stopped. Success:', trainingSucceeded, 'Failed:', trainingFailed);
                            
                            if (trainingFailed || exitCode !== 0) {
                                const errorMsg = stdout.split('\n').reverse().find(line => 
                                    line.includes('Error') || line.includes('Exception') || line.includes('Failed')
                                ) || 'Training failed - check logs for details';
                                reject(new Error(`Training failed: ${errorMsg}`));
                            } else if (trainingSucceeded) {
                                console.log('[CloudTraining] Training completed successfully');
                                resolve({
                                    success: true,
                                    stdout: stdout,
                                    stderr: ''
                                });
                            } else {
                                // Unknown state - log is empty or unclear
                                if (!stdout || stdout.trim() === '') {
                                    // Empty log usually means script failed to run
                                    reject(new Error('Training log is empty. The training script likely failed to start. Possible causes:\n' +
                                        '1. Python script syntax error\n' +
                                        '2. Missing dependencies (ultralytics not installed)\n' +
                                        '3. Dataset path incorrect (data.yaml not found)\n' +
                                        '4. Permission errors\n\n' +
                                        'Check the remote instance logs or try running the training script manually via SSH.'));
                                } else {
                                    // Wait a bit longer and check again
                                    console.warn('[CloudTraining] Training state unclear, checking logs again...');
                                    await this.sleep(2000);
                                    const recheckLog = await this.sshConnection.executeCommand('cat ~/training/training.log 2>/dev/null || echo ""');
                                    const recheckStdout = recheckLog.stdout || '';
                                    if (recheckStdout.includes('Training completed successfully')) {
                                        resolve({
                                            success: true,
                                            stdout: recheckStdout,
                                            stderr: ''
                                        });
                                    } else {
                                        // If no clear success but no failure either, reject to be safe
                                        reject(new Error(`Training status unclear. Log content (last 500 chars):\n${stdout.slice(-500)}\n\nTraining may have failed or is still in progress.`));
                                    }
                                }
                            }
                        }
                    } catch (error) {
                        // Continue checking
                    }
                }, 10000); // Check every 10 seconds
                
                // Store cost warning interval for cleanup
                this.costWarningInterval = costWarningInterval;

                // FEATURE 3: Stream stdout and parse progress
                execStream.on('data', (data) => {
                    const output = data.toString();
                    stdout += output;
                    lastOutputTime = Date.now();
                    
                    console.log('[CloudTraining] Stream output:', output.trim());
                    
                    // If we're getting output, the stream is working
                    if (output.trim().length > 0) {
                        // This confirms the tail -f is working and training is producing output
                    }
                    
                    // Parse artifact messages (save_dir from Ultralytics)
                    try {
                        const lines = output.split('\n');
                        for (const line of lines) {
                            if (line.includes('"type"') && line.includes('"artifact"')) {
                                try {
                                    const artifactData = JSON.parse(line);
                                    if (artifactData.type === 'artifact' && artifactData.data && artifactData.data.save_dir) {
                                        ultralyticsSaveDir = artifactData.data.save_dir;
                                        this.ultralyticsSaveDir = ultralyticsSaveDir; // Store for downloadModel to use
                                        console.log('[CloudTraining] Ultralytics save_dir detected:', ultralyticsSaveDir);
                                        this.sendLog(`Model will be saved to: ${ultralyticsSaveDir}`, 'info');
                                    }
                                } catch (e) {
                                    // Not JSON, continue
                                }
                            }
                        }
                    } catch (e) {
                        // Ignore parsing errors
                    }
                    
                    // Parse progress from output
                    this.parseAndSendProgress(output);
                    
                    // Send raw output to UI (SECURITY: sanitize sensitive data)
                    this.sendLog(this.sanitizeLog(output));
                });

                execStream.stderr.on('data', (data) => {
                    const output = data.toString();
                    console.error('[CloudTraining] Error output:', output.trim());
                    this.sendLog(this.sanitizeLog(output), 'error');
                });

                execStream.on('close', () => {
                    clearInterval(trainingCheckInterval);
                    clearInterval(initialCheck);
                    streamCleanup();
                });

                execStream.on('error', (error) => {
                    clearInterval(trainingCheckInterval);
                    clearInterval(initialCheck);
                    streamCleanup();
                    console.error('[CloudTraining] Stream error:', error);
                    reject(error);
                });
            });
        });
    }

    /**
     * FEATURE 3 (continued): Parse training progress from output
     */
    parseAndSendProgress(output) {
        try {
            // Try to parse JSON progress messages
            const lines = output.split('\n');
            for (const line of lines) {
                // Parse JSON progress messages from send_progress()
                if (line.includes('"type"') && line.includes('"progress"')) {
                    try {
                        const progressData = JSON.parse(line);
                        if (progressData.type === 'progress') {
                            this.sendProgress(progressData.data);
                            return; // Found JSON progress, done
                        }
                    } catch (e) {
                        // Not valid JSON, continue
                    }
                }
                
                // Parse Ultralytics YOLO progress output formats:
                // Format 1: "Epoch 1/10: ..." or "Epoch   1/10: ..."
                // Format 2: "train: epoch=1/10 ..."
                // Format 3: "      Epoch   1/10       ..."
                let epochMatch = line.match(/Epoch[:\s]+(\d+)\/(\d+)/i) ||
                                line.match(/epoch[=:\s]+(\d+)\/(\d+)/i) ||
                                line.match(/^\s+Epoch\s+(\d+)\/(\d+)/i);
                
                if (epochMatch) {
                    const epoch = parseInt(epochMatch[1]);
                    const totalEpochs = parseInt(epochMatch[2]);
                    
                    // Extract metrics if present (loss, mAP, etc.)
                    const metrics = {};
                    const lossMatch = line.match(/loss[=:\s]+([\d.]+)/i);
                    if (lossMatch) metrics.loss = parseFloat(lossMatch[1]);
                    
                    const mapMatch = line.match(/mAP[=:\s]+([\d.]+)/i);
                    if (mapMatch) metrics.map = parseFloat(mapMatch[1]);
                    
                    this.sendProgress({
                        epoch: epoch,
                        total_epochs: totalEpochs,
                        progress: epoch / totalEpochs,
                        status: 'training',
                        metrics: metrics
                    });
                }
            }
        } catch (error) {
            // Ignore parsing errors, but log for debugging
            console.warn('[CloudTraining] Progress parsing error:', error.message);
        }
    }

    /**
     * Write run summary artifact for debugging
     * @param {object} config - Training configuration
     * @param {object} trainingResult - Training result
     * @returns {Promise<void>}
     */
    async writeRunSummary(config, trainingResult) {
        const summary = {
            framework: config.trainingSettings?.framework || 'unknown',
            instance_id: this.instanceId || 'unknown',
            region: config.region || 'unknown',
            save_dir: this.ultralyticsSaveDir || 'unknown',
            status: trainingResult?.success ? 'success' : 'error',
            timestamp: new Date().toISOString()
        };

        // Write summary as JSON to remote instance
        // Use explicit newline join for heredoc safety
        const summaryJson = JSON.stringify(summary, null, 2);
        const summaryCommand = [
            "cat > ~/training/run_summary.json <<'EOF'",
            summaryJson,
            "EOF"
        ].join("\n");

        await this.sshConnection.executeCommand(summaryCommand, 5000);
        // Log without exposing instance_id (security/privacy)
        console.log('[CloudTraining] Run summary written (framework:', summary.framework + ', status:', summary.status + ')');
    }

    /**
     * FEATURE 4: Model Download Automation
     * Download trained model from remote instance
     */
    async downloadModel(settings) {
        console.log('[CloudTraining] Downloading trained model...');

        // Wait a moment for model file to be fully written after training completes
        await this.sleep(3000);

        // Normalize remote paths helper (SFTP doesn't always expand ~)
        // Converts any path (relative, ~/, or absolute) to absolute path
        const normalizeRemotePath = (p) => {
            if (!p) return p;
            if (p.startsWith('~/')) return p.replace('~/', '/home/ubuntu/');
            if (p.startsWith('/')) return p;
            // relative path - check if it looks like a training-related path
            // If it doesn't start with training/, runs/, or output/, assume it's under /home/ubuntu/training/
            if (!p.startsWith('training/') && !p.startsWith('runs/') && !p.startsWith('output/')) {
                return `/home/ubuntu/training/${p.replace(/^\.?\//, '')}`;
            }
            // Already has training/runs/output prefix, just prepend /home/ubuntu/
            return `/home/ubuntu/${p.replace(/^\.?\//, '')}`;
        };

        // Determine model location based on framework
        const framework = settings.framework || 'yolo';
        let remoteModelPath = '';
        let localModelDir = '';
        
        if (framework === 'yolo') {
            // YOLO saves to runs/detect/train/weights/best.pt
            // Also check standard YOLO output locations
            remoteModelPath = '/home/ubuntu/training/output/training_run/weights/best.pt'; // Use absolute path
            localModelDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'models');
        } else {
            remoteModelPath = '/home/ubuntu/training/output/model.pth'; // Use absolute path
            localModelDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'models');
        }

        // Create local directory if it doesn't exist
        if (!fs.existsSync(localModelDir)) {
            fs.mkdirSync(localModelDir, { recursive: true });
        }

        // Generate local filename
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const modelFilename = `model_${framework}_${timestamp}.pt`;
        const localModelPath = path.join(localModelDir, modelFilename);

        // REFINEMENT 4: Try expected path first (fast-path), then find, then fallback
        console.log('[CloudTraining] Locating model file...');
        
        // Fast-path 1: Try the expected remoteModelPath directly
        try {
            const verifyResult = await this.sshConnection.executeCommand(
                `test -f "${remoteModelPath}" && echo "exists" || echo "not_found"`,
                5000
            );
            if (verifyResult.stdout && verifyResult.stdout.includes('exists')) {
                await this.sshConnection.downloadFile(remoteModelPath, localModelPath);
                console.log('[CloudTraining] Model downloaded from expected path (fast-path)');
                this.sendStatus(`Model downloaded: ${modelFilename}`);
                return localModelPath;
            }
        } catch (e) {
            console.warn('[CloudTraining] Fast-path check failed, trying other methods:', e.message);
        }
        
        // Fast-path 2: If we have the actual save_dir from Ultralytics, search there first
        if (this.ultralyticsSaveDir) {
            const saveDir = normalizeRemotePath(this.ultralyticsSaveDir);
            console.log('[CloudTraining] Using Ultralytics save_dir:', saveDir);
            try {
                const saveDirFind = await this.sshConnection.executeCommand(
                    `find "${saveDir}" -type f \\( -name "*.pt" -o -name "*.pth" \\) 2>/dev/null | head -5`,
                    30000
                );
                if (saveDirFind.stdout && saveDirFind.stdout.trim()) {
                    const paths = saveDirFind.stdout.trim().split('\n')
                        .map(line => line.trim())
                        .filter(p => p && p.includes('.'));
                    const foundPath = paths.find(p => p.includes('best.pt')) ||
                                     paths.find(p => p.includes('last.pt')) ||
                                     paths.find(p => p.includes('.pt')) ||
                                     paths[0];
                    if (foundPath) {
                        const absPath = normalizeRemotePath(foundPath);
                        const verifyResult = await this.sshConnection.executeCommand(
                            `test -f "${absPath}" && echo "exists" || echo "not_found"`,
                            5000
                        );
                        if (verifyResult.stdout && verifyResult.stdout.includes('exists')) {
                            await this.sshConnection.downloadFile(absPath, localModelPath);
                            console.log('[CloudTraining] Model downloaded from Ultralytics save_dir');
                            this.sendStatus(`Model downloaded: ${modelFilename}`);
                            return localModelPath;
                        }
                    }
                }
            } catch (e) {
                console.warn('[CloudTraining] Failed to find model in save_dir, trying other locations:', e.message);
            }
        }
        
        try {
            // Search in multiple locations
            const findCommands = [
                'find ~/training -type f \\( -name "*.pt" -o -name "*.pth" \\) 2>/dev/null | head -5',
                'find ~/training/runs -type f \\( -name "*.pt" -o -name "*.pth" \\) 2>/dev/null | head -5',
                'ls -la ~/training/output/training_run/weights/*.pt 2>/dev/null | head -5'
            ];
            
            let foundPath = null;
            for (const findCmd of findCommands) {
                try {
                    const findResult = await this.sshConnection.executeCommand(findCmd, 30000);
                    
                    if (findResult.stdout && findResult.stdout.trim()) {
                        // Get first valid path (prefer best.pt over last.pt)
                        const paths = findResult.stdout.trim().split('\n')
                            .map(line => {
                                // Extract path from ls output or use line as-is
                                // Handle both absolute paths and relative paths
                                const match = line.match(/(\/[^\s]+\.(pt|pth)|[^\s]+\.(pt|pth))/);
                                return match ? match[0] : line.trim();
                            })
                            .filter(p => p && p.includes('.'));
                        
                        // Prefer best.pt, then last.pt, then any .pt
                        foundPath = paths.find(p => p.includes('best.pt')) ||
                                   paths.find(p => p.includes('last.pt')) ||
                                   paths.find(p => p.includes('.pt')) ||
                                   paths[0];
                        
                        if (foundPath) {
                            // Force to absolute path (handles relative, ~/, and absolute)
                            foundPath = normalizeRemotePath(foundPath);
                            console.log('[CloudTraining] Found model at:', foundPath);
                            break;
                        }
                    }
                } catch (findError) {
                    console.warn('[CloudTraining] Find command failed:', findCmd, findError.message);
                }
            }
            
            if (foundPath) {
                try {
                    // Verify file exists before trying to download
                    const verifyResult = await this.sshConnection.executeCommand(
                        `test -f "${foundPath}" && echo "exists" || echo "not_found"`,
                        5000
                    );
                    
                    if (verifyResult.stdout && verifyResult.stdout.includes('exists')) {
                        await this.sshConnection.downloadFile(foundPath, localModelPath);
                        console.log('[CloudTraining] Model downloaded from found path');
                        this.sendStatus(`Model downloaded: ${modelFilename}`);
                        return localModelPath;
                    }
                } catch (downloadError) {
                    console.warn('[CloudTraining] Failed to download from found path:', downloadError.message);
                }
            }
        } catch (findError) {
            console.warn('[CloudTraining] Find command search failed:', findError.message);
        }

        // Fallback: Try the expected paths directly
        // (normalizeRemotePath already defined above)
        const alternativePaths = [
            '~/training/output/training_run/weights/best.pt',
            '~/training/output/training_run/weights/last.pt',
            '~/training/runs/detect/train/weights/best.pt',
            '~/training/runs/detect/train/weights/last.pt',
            '~/training/output/model.pt',
            '~/training/output/best.pth'
        ];

        for (const altPath of alternativePaths) {
            try {
                const absPath = normalizeRemotePath(altPath);
                // Verify file exists before trying to download
                const verifyResult = await this.sshConnection.executeCommand(
                    `test -f "${absPath}" && echo "exists" || echo "not_found"`,
                    5000
                );
                
                if (verifyResult.stdout && verifyResult.stdout.includes('exists')) {
                    await this.sshConnection.downloadFile(absPath, localModelPath);
                    console.log('[CloudTraining] Model downloaded from alternative path:', absPath);
                    this.sendStatus(`Model downloaded: ${modelFilename}`);
                    return localModelPath;
                }
            } catch (e) {
                // Continue trying
                console.warn(`[CloudTraining] Failed to download from ${altPath}:`, e.message);
            }
        }

        // Last resort: Check what actually exists in the output directory
        try {
            const dirCheck = await this.sshConnection.executeCommand(
                'ls -laR ~/training/output ~/training/runs 2>/dev/null | grep -E "\\.(pt|pth)$" | head -10',
                30000
            );
            const directoryListing = dirCheck.stdout || 'No output found';
            
            throw new Error(`Failed to download model from any known location.\n\nTraining may not have completed successfully, or model was saved to an unexpected location.\n\nDirectory contents:\n${directoryListing}\n\nCheck the training logs for errors.`);
        } catch (finalError) {
            throw finalError; // Re-throw if it's our custom error
        }
    }

    /**
     * Terminate cloud instance and cleanup
     * CRITICAL: This must ALWAYS succeed to prevent overcharging
     */
    async terminateInstance() {
        if (!this.instanceId) {
            console.warn('[CloudTraining] No instance ID to terminate');
            return;
        }

        // Clear auto-termination timeout if it exists
        if (this.maxTrainingTimeout) {
            clearTimeout(this.maxTrainingTimeout);
            this.maxTrainingTimeout = null;
        }

        const instanceId = this.instanceId;
        const project = this.cloudConfig?.project;
        const region = this.cloudConfig?.region;
        const apiKey = this.cloudConfig?.apiKey;

        // Calculate runtime and cost for logging
        let runtimeHours = 0;
        let estimatedCost = 0;
        if (this.instanceStartTime) {
            const runtimeMs = Date.now() - this.instanceStartTime;
            runtimeHours = runtimeMs / (1000 * 60 * 60);
            
            // Estimate cost if we have GPU info
            if (this.cloudConfig?.gpuName) {
                const gpuText = this.cloudConfig.gpuName;
                const priceMatch = gpuText.match(/\$([\d.]+)/);
                const gpuCountMatch = gpuText.match(/(\d+)x/i);
                const gpuCount = gpuCountMatch ? parseInt(gpuCountMatch[1]) : 1;
                
                if (priceMatch) {
                    const pricePerGpuPerHour = parseFloat(priceMatch[1]);
                    estimatedCost = pricePerGpuPerHour * gpuCount * runtimeHours;
                }
            }
        }

        // Store instance info before clearing (for retry attempts)
        const instanceInfo = {
            instanceId: instanceId,
            project: project,
            region: region,
            apiKey: apiKey
        };
        
        let terminationSucceeded = false;

        try {
            console.log(`[CloudTraining] Terminating instance: ${instanceId}`);
            console.log(`[CloudTraining] Project: ${project}, Region: ${region}`);
            console.log(`[CloudTraining] Instance runtime: ${runtimeHours.toFixed(2)} hours`);
            if (estimatedCost > 0) {
                console.log(`[CloudTraining] Estimated cost: $${estimatedCost.toFixed(2)}`);
                this.sendLog(`Instance runtime: ${runtimeHours.toFixed(2)} hours. Estimated cost: $${estimatedCost.toFixed(2)}`, 'info');
            }
            
            // Validate we have all required info
            if (!instanceId) {
                throw new Error('Instance ID is missing');
            }
            if (!project) {
                throw new Error('Project is missing');
            }
            if (!region) {
                throw new Error('Region is missing');
            }
            if (!apiKey) {
                throw new Error('API key is missing');
            }
            
            // Try to terminate with current client first
            let terminationSucceeded = false;
            if (this.client && project && region) {
                try {
                    console.log('[CloudTraining] Attempting termination with current client...');
                    const result = await this.client.terminateInstance(instanceId, project, region);
                    console.log('[CloudTraining] Termination API response:', JSON.stringify(result));
                    terminationSucceeded = true;
                    console.log('[CloudTraining] Instance terminated successfully');
                    this.sendStatus('Cloud instance terminated');
                    this.sendLog('✓ Cloud instance terminated successfully', 'success');
                } catch (clientError) {
                    console.error('[CloudTraining] Termination with current client failed:', clientError);
                    console.error('[CloudTraining] Error details:', {
                        message: clientError.message,
                        statusCode: clientError.statusCode,
                        apiResponse: clientError.apiResponse
                    });
                    // Fall through to emergency client
                }
            }
            
            // Fallback: Create new client if current one failed or is missing
            if (!terminationSucceeded && apiKey && project && region) {
                try {
                    console.warn('[CloudTraining] Client missing or failed, creating new one for termination');
                    const CanopyWaveAPI = require('./canopywave-api');
                    const emergencyClient = new CanopyWaveAPI(apiKey);
                    const result = await emergencyClient.terminateInstance(instanceId, project, region);
                    console.log('[CloudTraining] Termination API response (emergency):', JSON.stringify(result));
                    terminationSucceeded = true;
                    console.log('[CloudTraining] Instance terminated via emergency client');
                    this.sendStatus('Cloud instance terminated');
                    this.sendLog('✓ Cloud instance terminated successfully (emergency client)', 'success');
                } catch (emergencyError) {
                    console.error('[CloudTraining] Emergency termination failed:', emergencyError);
                    console.error('[CloudTraining] Emergency error details:', {
                        message: emergencyError.message,
                        statusCode: emergencyError.statusCode,
                        apiResponse: emergencyError.apiResponse
                    });
                }
            }
            
            // CRITICAL: Final retry attempt with fresh client
            if (!terminationSucceeded && apiKey && project && region) {
                try {
                    console.log('[CloudTraining] Final retry: Creating fresh client for termination...');
                    const CanopyWaveAPI = require('./canopywave-api');
                    const retryClient = new CanopyWaveAPI(apiKey);
                    // Add a small delay before retry
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    const result = await retryClient.terminateInstance(instanceId, project, region);
                    console.log('[CloudTraining] Termination API response (retry):', JSON.stringify(result));
                    terminationSucceeded = true;
                    console.log('[CloudTraining] Retry termination succeeded');
                    this.sendStatus('Cloud instance terminated (retry)');
                    this.sendLog('✓ Cloud instance terminated successfully (retry)', 'success');
                } catch (retryError) {
                    console.error('[CloudTraining] Final retry termination failed:', retryError);
                    console.error('[CloudTraining] Retry error details:', {
                        message: retryError.message,
                        statusCode: retryError.statusCode,
                        apiResponse: retryError.apiResponse,
                        rawResponse: retryError.rawResponse
                    });
                    
                    // Log instance ID so user can manually terminate
                    const shortId = (instanceId || '').toString().slice(0, 8);
                    const errorDetails = `Instance ID: ${shortId}…\nProject: ${project}\nRegion: ${region}\nError: ${retryError.message}`;
                    this.sendLog(`❌ CRITICAL: Could not terminate instance after multiple attempts.\n${errorDetails}\n\nPlease terminate manually in CanopyWave dashboard immediately to prevent charges.`, 'error');
                    
                    // Don't throw - we've tried everything, just log the error
                    console.error('[CloudTraining] All termination attempts failed. Instance may still be running.');
                }
            }
            
            // Verify termination succeeded by checking instance status
            if (terminationSucceeded && instanceInfo.instanceId && instanceInfo.project && instanceInfo.region && instanceInfo.apiKey) {
                try {
                    console.log('[CloudTraining] Verifying instance termination...');
                    // Wait a moment for termination to process
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    
                    // Use a fresh client for verification
                    const CanopyWaveAPI = require('./canopywave-api');
                    const verifyClient = this.client || new CanopyWaveAPI(instanceInfo.apiKey);
                    
                    // Try to get instance status - if it fails or returns deleted/terminated, we're good
                    try {
                        const instanceStatus = await verifyClient.getInstance(instanceInfo.instanceId, instanceInfo.project, instanceInfo.region);
                        console.log('[CloudTraining] Instance status after termination:', instanceStatus.status);
                        
                        // If instance still exists and is not in a terminating state, log warning
                        if (instanceStatus.status && 
                            instanceStatus.status !== 'DELETED' && 
                            instanceStatus.status !== 'TERMINATED' &&
                            instanceStatus.status !== 'deleted' &&
                            instanceStatus.status !== 'terminated') {
                            console.warn(`[CloudTraining] WARNING: Instance status is "${instanceStatus.status}" after termination attempt. It may still be running.`);
                            this.sendLog(`⚠️ Warning: Instance status is "${instanceStatus.status}" after termination. Please verify in CanopyWave dashboard.`, 'warning');
                            terminationSucceeded = false; // Mark as failed if still running
                        } else {
                            console.log('[CloudTraining] ✓ Instance termination verified');
                        }
                    } catch (statusError) {
                        // If we can't get instance status, it might be deleted (which is good)
                        if (statusError.statusCode === 404) {
                            console.log('[CloudTraining] ✓ Instance not found (likely terminated successfully)');
                        } else {
                            console.warn('[CloudTraining] Could not verify termination status:', statusError.message);
                            // Don't mark as failed if we can't verify - API might be slow
                        }
                    }
                } catch (verifyError) {
                    console.warn('[CloudTraining] Error verifying termination:', verifyError);
                    // Don't fail the whole process if verification fails
                }
            }
            
            if (!terminationSucceeded) {
                throw new Error('All termination attempts failed');
            }
            
        } catch (error) {
            console.error('[CloudTraining] Fatal error in termination process:', error);
            console.error('[CloudTraining] Instance info:', instanceInfo);
            this.sendLog(`⚠️ Failed to terminate instance: ${error.message}. Please terminate manually in CanopyWave dashboard to prevent charges.`, 'error');
            
            // Log critical info for manual termination
            if (instanceInfo.instanceId) {
                const shortId = (instanceInfo.instanceId || '').toString().slice(0, 8);
                console.error(`[CloudTraining] CRITICAL: Instance ${shortId}… may still be running!`);
                console.error(`[CloudTraining] Project: ${instanceInfo.project}, Region: ${instanceInfo.region}`);
            }
        } finally {
            // Only clear instance tracking if termination succeeded or we've exhausted all retries
            // Keep instance info if termination failed so user can manually terminate
            if (terminationSucceeded !== false) {
                // Only clear if we successfully terminated OR if we don't have the info to retry
                if (terminationSucceeded || !instanceInfo.instanceId || !instanceInfo.apiKey) {
                    this.instanceId = null;
                    this.instanceStartTime = null;
                } else {
                    // Keep instance info for potential manual termination
                    console.warn('[CloudTraining] Keeping instance info for manual termination:', {
                        instanceId: instanceInfo.instanceId,
                        project: instanceInfo.project,
                        region: instanceInfo.region
                    });
                }
            }
            
            // Close SSH connection
            if (this.sshConnection) {
                try {
                    this.sshConnection.close();
                } catch (closeError) {
                    console.warn('[CloudTraining] Error closing SSH connection:', closeError);
                }
                this.sshConnection = null;
            }
        }
    }

    /**
     * Stop training (user-initiated)
     * CRITICAL: Must always terminate instance to prevent charges
     */
    async stopTraining() {
        this.isTraining = false;
        
        // Clear auto-termination timeout
        if (this.maxTrainingTimeout) {
            clearTimeout(this.maxTrainingTimeout);
            this.maxTrainingTimeout = null;
        }
        
        if (this.progressCheckInterval) {
            clearInterval(this.progressCheckInterval);
            this.progressCheckInterval = null;
        }
        
        if (this.costWarningInterval) {
            clearInterval(this.costWarningInterval);
            this.costWarningInterval = null;
        }

        // CRITICAL: Always terminate instance when user stops training
        console.log('[CloudTraining] User requested stop - terminating instance');
        this.sendLog('Stopping training and terminating instance...', 'warning');
        await this.terminateInstance();
    }

    // Helper methods

    sendStatus(message) {
        console.log('[CloudTraining] Status:', message);
        if (this.mainWindow) {
            this.mainWindow.webContents.send('cloud-training-status', { status: message });
        }
    }

    sendProgress(data) {
        if (this.mainWindow) {
            this.mainWindow.webContents.send('training-progress', data);
        }
    }

    sendLog(message, type = 'info') {
        // Log to console with appropriate level for debugging
        if (type === 'error') {
            console.error('[CloudTraining] ERROR:', message);
        } else if (type === 'warning') {
            console.warn('[CloudTraining] WARNING:', message);
        } else {
            console.log(`[CloudTraining] ${type.toUpperCase()}:`, message);
        }
        if (this.mainWindow) {
            this.mainWindow.webContents.send('cloud-training-log', { message, type });
        }
    }

    /**
     * SECURITY: Sanitize log output to prevent password leakage
     * Removes passwords, API keys, and sensitive information from logs
     */
    sanitizeLog(message) {
        if (!message || typeof message !== 'string') {
            return message;
        }

        let sanitized = message;

        // Remove passwords from SSH commands
        sanitized = sanitized.replace(/password[=:\s]+['"]?([^'"\s]+)['"]?/gi, 'password=***REDACTED***');
        
        // Remove API keys
        sanitized = sanitized.replace(/cw_[a-zA-Z0-9_-]+/g, 'cw_***REDACTED***');
        sanitized = sanitized.replace(/Bearer\s+[a-zA-Z0-9_-]+/gi, 'Bearer ***REDACTED***');
        
        // Remove IP addresses in sensitive contexts (keep for status messages)
        // sanitized = sanitized.replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, 'IP.REDACTED');
        
        // Remove authorization headers
        sanitized = sanitized.replace(/Authorization:\s*[^\s]+/gi, 'Authorization: ***REDACTED***');

        return sanitized;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = CloudTrainingHandler;
