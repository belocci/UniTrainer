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

// Global error handlers - add at the very top
console.log('\n' + '='.repeat(80));
console.log('APP STARTED - PID:', process.pid);
console.log('process.execPath:', process.execPath);
console.log('__dirname:', __dirname);
console.log('='.repeat(80));

process.on('uncaughtException', (error) => {
    console.error('\n!!! UNCAUGHT EXCEPTION !!!');
    console.error('Error:', error.message);
    console.error('Stack:', error.stack);
    console.error('Full error:', error);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('\n!!! UNHANDLED REJECTION !!!');
    console.error('Reason:', reason);
    console.error('Promise:', promise);
});

const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn, execFile } = require('child_process');
const https = require('https');
const CanopyWaveAPI = require('./canopywave-api');
const CloudTrainingHandler = require('./cloud-training-handler');

// Set up log file for debugging (use temp dir since app.getPath requires app to be ready)
const logFile = path.join(os.tmpdir(), 'uni-trainer-debug.log');
let logStream = null;

// Try to create log file
try {
    logStream = fs.createWriteStream(logFile, { flags: 'a' });
    console.log('Log file location:', logFile);
} catch (e) {
    console.error('Could not create log file:', e.message);
}

// Override console.log and console.error to also write to file
const originalLog = console.log;
const originalError = console.error;

console.log = function(...args) {
    const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)).join(' ');
    originalLog.apply(console, args);
    if (logStream) {
        try {
            logStream.write(`[LOG] ${new Date().toISOString()} - ${message}\n`);
        } catch (e) {
            // Ignore write errors
        }
    }
};

console.error = function(...args) {
    const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)).join(' ');
    originalError.apply(console, args);
    if (logStream) {
        try {
            logStream.write(`[ERROR] ${new Date().toISOString()} - ${message}\n`);
        } catch (e) {
            // Ignore write errors
        }
    }
};

function getLocalAppDataPath() {
    try {
        const localAppData = app.getPath('localAppData');
        return { path: localAppData, source: 'app.getPath', error: null };
    } catch (error) {
        const envLocalAppData = process.env.LOCALAPPDATA || '';
        if (envLocalAppData) {
            console.warn('[Path] app.getPath(localAppData) failed, using LOCALAPPDATA env var:', error.message);
            return { path: envLocalAppData, source: 'env.LOCALAPPDATA', error: error.message };
        }

        const userProfile = process.env.USERPROFILE || '';
        if (userProfile) {
            const fallbackPath = path.join(userProfile, 'AppData', 'Local');
            console.warn('[Path] app.getPath(localAppData) failed, using USERPROFILE fallback:', error.message);
            return { path: fallbackPath, source: 'USERPROFILE', error: error.message };
        }

        console.warn('[Path] app.getPath(localAppData) failed, no fallback available:', error.message);
        return { path: '', source: 'none', error: error.message };
    }
}

function getUserHomePath() {
    try {
        const homePath = app.getPath('home');
        return { path: homePath, source: 'app.getPath', error: null };
    } catch (error) {
        const osHome = os.homedir();
        if (osHome) {
            console.warn('[Path] app.getPath(home) failed, using os.homedir():', error.message);
            return { path: osHome, source: 'os.homedir', error: error.message };
        }

        const envUserProfile = process.env.USERPROFILE || '';
        if (envUserProfile) {
            console.warn('[Path] app.getPath(home) failed, using USERPROFILE env var:', error.message);
            return { path: envUserProfile, source: 'env.USERPROFILE', error: error.message };
        }

        console.warn('[Path] app.getPath(home) failed, no fallback available:', error.message);
        return { path: '', source: 'none', error: error.message };
    }
}

function findPythonInDirectory(rootDir, maxDepth = 3) {
    try {
        const entries = fs.readdirSync(rootDir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = path.join(rootDir, entry.name);
            if (entry.isFile() && entry.name.toLowerCase() === 'python.exe') {
                return fullPath;
            }
            if (entry.isDirectory() && maxDepth > 0) {
                const found = findPythonInDirectory(fullPath, maxDepth - 1);
                if (found) return found;
            }
        }
    } catch (error) {
        // Ignore unreadable directories
    }
    return null;
}

function resolvePythonFromPyLauncher() {
    try {
        const { execSync, execFileSync } = require('child_process');
        let resolved = '';
        try {
            resolved = execSync('py -3 -c "import sys; print(sys.executable)"', {
                encoding: 'utf8',
                stdio: ['ignore', 'pipe', 'ignore'],
                timeout: 5000
            }).trim();
        } catch (e) {
            const pyCandidates = ['C:\\Windows\\py.exe', 'C:\\Windows\\System32\\py.exe'];
            for (const pyPath of pyCandidates) {
                if (fs.existsSync(pyPath)) {
                    try {
                        resolved = execFileSync(pyPath, ['-3', '-c', 'import sys; print(sys.executable)'], {
                            encoding: 'utf8',
                            stdio: ['ignore', 'pipe', 'ignore'],
                            timeout: 5000
                        }).trim();
                        break;
                    } catch (inner) {
                        // continue
                    }
                }
            }
        }
        if (resolved) {
            const normalized = path.normalize(resolved);
            if (fs.existsSync(normalized) || normalized.toLowerCase().includes('windowsapps')) {
                return normalized;
            }
        }
    } catch (e) {
        // py launcher check failed, continue
    }
    return '';
}

function resolvePyLauncherExecutable() {
    const candidates = [
        'C:\\Windows\\py.exe',
        'C:\\Windows\\System32\\py.exe'
    ];
    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }
    return 'py';
}

function getRunnableTrainerPyPath() {
    const asarPath = path.join(__dirname, 'trainer.py');
    const resourcesPath = path.join(__dirname, 'resources', 'trainer.py');
    const source = fs.existsSync(asarPath)
        ? asarPath
        : (fs.existsSync(resourcesPath) ? resourcesPath : null);

    if (!source) {
        return null;
    }

    if (!app.isPackaged) {
        return source;
    }

    const tempDir = path.join(os.tmpdir(), 'uni-trainer-scripts');
    if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir, { recursive: true });
    }

    const dest = path.join(tempDir, 'trainer.py');
    fs.writeFileSync(dest, fs.readFileSync(source));
    return dest;
}

function runPythonInfer(config, options = {}) {
    const label = options.label || 'Inference';
    return new Promise(async (resolve, reject) => {
        try {
            let pythonCmd = await ensurePythonAvailable();
            console.log(`[${label}] ensurePythonAvailable() returned:`, pythonCmd);

            const execDirForBundled = path.dirname(process.execPath);
            const bundledPythonPath = path.resolve(execDirForBundled, 'resources', 'python', 'python.exe');
            if (fs.existsSync(bundledPythonPath)) {
                console.log(`[${label}] Bundled Python found, forcing:`, bundledPythonPath);
                pythonCmd = bundledPythonPath;
            }

            if (pythonCmd && pythonCmd.toLowerCase().includes('\\pymanager\\')) {
                const resolved = resolvePythonFromPyLauncher();
                if (resolved) {
                    console.warn(`[${label}] Replacing PyManager path with py -3:`, resolved);
                    pythonCmd = resolved;
                }
            }

            if (!pythonCmd || !isUsablePythonPath(pythonCmd)) {
                console.error(`[${label}] Python executable not found or unusable:`, pythonCmd);
                reject(new Error(`Python executable not found: ${pythonCmd || 'null'}`));
                return;
            }

            const scriptPath = getRunnableTrainerPyPath();
            if (!scriptPath) {
                reject(new Error('trainer.py not found'));
                return;
            }

            const normalizedPath = path.normalize(pythonCmd);
            const normalizedScript = path.normalize(scriptPath);
            const isWindowsAppsPython = normalizedPath.toLowerCase().includes('\\windowsapps\\');

            const pythonDir = path.dirname(normalizedPath);
            const pythonLibPath = path.join(pythonDir, 'Lib');
            const sitePackagesPath = path.join(pythonLibPath, 'site-packages');

            const localAppDataInfo = getLocalAppDataPath();
            const windowsAppsDir = localAppDataInfo.path
                ? path.join(localAppDataInfo.path, 'Microsoft', 'WindowsApps')
                : '';
            const basePath = process.env.PATH || '';

            const env = {
                ...process.env,
                PYTHONUNBUFFERED: '1',
                PYTHONHOME: isWindowsAppsPython ? undefined : pythonDir,
                PYTHONPATH: isWindowsAppsPython ? undefined : pythonLibPath,
                PATH: `${windowsAppsDir ? `${windowsAppsDir};` : ''}${pythonDir};${basePath}`
            };

            if (!isWindowsAppsPython && fs.existsSync(sitePackagesPath)) {
                env.PYTHONPATH = `${sitePackagesPath};${pythonLibPath}`;
            } else {
                console.warn(`[${label}] site-packages not found, using Lib only`);
            }

            const execDir = path.dirname(normalizedScript);
            const spawnCmd = isWindowsAppsPython ? resolvePyLauncherExecutable() : normalizedPath;
            const spawnArgs = isWindowsAppsPython ? ['-3', normalizedScript, 'infer'] : [normalizedScript, 'infer'];
            const inferenceProcess = spawn(spawnCmd, spawnArgs, {
                cwd: execDir,
                env,
                stdio: ['pipe', 'pipe', 'pipe'],
                windowsHide: true
            });

            const configJson = JSON.stringify(config) + '\n';
            inferenceProcess.stdin.write(configJson);
            inferenceProcess.stdin.end();

            let stdout = '';
            let stderr = '';
            let inferenceResult = null;

            inferenceProcess.stdout.on('data', (data) => {
                stdout += data.toString();
                const lines = data.toString().split('\n');
                lines.forEach(line => {
                    if (!line.trim()) return;
                    try {
                        const jsonData = JSON.parse(line.trim());
                        if (jsonData.type === 'result') {
                            inferenceResult = jsonData.data;
                        } else if (jsonData.type === 'log') {
                            mainWindow.webContents.send('inference-log', jsonData.data);
                        } else if (jsonData.type === 'error') {
                            const traceback = jsonData.data && jsonData.data.traceback
                                ? `\n${jsonData.data.traceback}`
                                : '';
                            mainWindow.webContents.send('inference-log', { message: `${jsonData.data.error}${traceback}`, level: 'error' });
                        }
                    } catch (e) {
                        mainWindow.webContents.send('inference-log', { message: line.trim(), level: 'log' });
                    }
                });
            });

            inferenceProcess.stderr.on('data', (data) => {
                stderr += data.toString();
                const lines = data.toString().split('\n');
                lines.forEach(line => {
                    if (line.trim()) {
                        mainWindow.webContents.send('inference-log', { message: line.trim(), level: 'error' });
                    }
                });
            });

            inferenceProcess.on('exit', (code) => {
                if (code === 0) return resolve(inferenceResult || { output: stdout });
                const trimmedErr = (stderr || '').trim();
                const trimmedOut = (stdout || '').trim();
                const detail = [
                    `Inference failed (exit ${code})`,
                    trimmedErr ? `--- STDERR ---\n${trimmedErr}` : '',
                    trimmedOut ? `--- STDOUT ---\n${trimmedOut}` : '',
                ].filter(Boolean).join('\n\n');
                reject(new Error(detail));
            });

            inferenceProcess.on('error', (e) => reject(new Error(`Failed to start inference: ${e.message}`)));
        } catch (error) {
            reject(error);
        }
    });
}

function isUsablePythonPath(pythonPath) {
    if (!pythonPath) return false;
    const normalized = path.normalize(pythonPath);
    if (fs.existsSync(normalized)) return true;
    const lower = normalized.toLowerCase();
    if (lower.includes('\\windowsapps\\pythonsoftwarefoundation.python.') || lower.endsWith('\\windowsapps\\python.exe')) {
        return true;
    }
    return false;
}

function downloadFile(url, destination) {
    return new Promise((resolve, reject) => {
        const request = https.get(url, (response) => {
            if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
                return resolve(downloadFile(response.headers.location, destination));
            }
            if (response.statusCode !== 200) {
                return reject(new Error(`Download failed with status ${response.statusCode}`));
            }

            const fileStream = fs.createWriteStream(destination);
            response.pipe(fileStream);
            fileStream.on('finish', () => fileStream.close(resolve));
            fileStream.on('error', reject);
        });
        request.on('error', reject);
    });
}

let pythonInstallPromise = null;
async function attemptAutoInstallPython() {
    try {
        const installerUrl = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe';
        const installerPath = path.join(os.tmpdir(), 'python-3.12.10-amd64.exe');
        const execDir = path.dirname(process.execPath);
        const targetDir = (app.isPackaged && process.resourcesPath)
            ? path.join(process.resourcesPath, 'python')
            : path.resolve(execDir, 'resources', 'python');

        if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true });
        }

        console.log('[Python Install] Downloading installer:', installerUrl);
        await downloadFile(installerUrl, installerPath);
        console.log('[Python Install] Installer downloaded:', installerPath);

        const { execFileSync } = require('child_process');
        execFileSync(installerPath, [
            '/quiet',
            'InstallAllUsers=0',
            'Include_pip=1',
            'PrependPath=0',
            `TargetDir=${targetDir}`
        ], { windowsHide: true, timeout: 300000 });

        const installedPython = path.join(targetDir, 'python.exe');
        if (fs.existsSync(installedPython)) {
            console.log('[Python Install] Python installed at:', installedPython);
            return true;
        }

        console.warn('[Python Install] Install completed but python.exe not found:', installedPython);
        return false;
    } catch (error) {
        console.error('[Python Install] Auto-install failed:', error.message);
        return false;
    }
}

async function ensurePythonAvailable() {
    let pythonCmd = findPythonPath();
    if (pythonCmd && isUsablePythonPath(pythonCmd)) {
        return pythonCmd;
    }

    const execDir = path.dirname(process.execPath);
    const appDirPython = findPythonInDirectory(execDir, 3);
    if (appDirPython && isUsablePythonPath(appDirPython)) {
        console.log('[Python Detection] Found python.exe inside app directory:', appDirPython);
        return appDirPython;
    }

    if (!pythonInstallPromise) {
        pythonInstallPromise = attemptAutoInstallPython();
    }
    await pythonInstallPromise;

    pythonCmd = findPythonPath();
    if (pythonCmd && isUsablePythonPath(pythonCmd)) {
        return pythonCmd;
    }

    const appDirPythonAfterInstall = findPythonInDirectory(execDir, 3);
    if (appDirPythonAfterInstall && isUsablePythonPath(appDirPythonAfterInstall)) {
        console.log('[Python Detection] Found python.exe inside app directory after install:', appDirPythonAfterInstall);
        return appDirPythonAfterInstall;
    }

    return null;
}

let mainWindow;
let trainingProcess = null;
let trainingManuallyStopped = false; // Track if user manually stopped training
let stopRequestedByUser = false; // Explicit flag for user-initiated stop
let lastProgressTime = Date.now(); // Track last progress update time

// Shared function to find Python executable - check bundled first, then system
// Store the expected total epochs from config to validate parsed progress
// Track last parsed epoch info for header line detection
let expectedTotalEpochs = null;
let lastParsedEpoch = 0;
let lastParsedTotalEpochs = 10;

// Parse YOLO training progress from stdout output

// Simple progress parser - less strict, finds epoch/total pattern anywhere
function parseSimpleYOLOProgress(line) {
  // Strip ANSI escape codes (e.g., [K, [34m, [1m, etc.)
  const ansiRegex = /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;
  const strippedLine = line.replace(ansiRegex, '').trim();
  
  // Strip progress bar characters (━, ─, etc.) and other non-ASCII progress indicators
  const progressBarRegex = /[━─═─┈┉┊┋┌┐└┘├┤┬┴┼│█▉▊▋▌▍▎▏]/g;
  const cleanLine = strippedLine.replace(progressBarRegex, '').trim();
  
  // Look for "epoch/total" pattern anywhere in the line (more flexible)
  const epochMatch = cleanLine.match(/(\d+)\/(\d+)/);
  
  if (epochMatch) {
    const epoch = parseInt(epochMatch[1]);
    const total = parseInt(epochMatch[2]);
    
    // Basic validation - epoch must be positive, total must be >= epoch, reasonable limit
    if (epoch > 0 && total >= epoch && total <= 1000 && epoch <= total) {
      // Extract losses if present (look for numbers after epoch/total)
      const numbers = cleanLine.match(/[\d\.]+/g);
      let boxLoss = null;
      let clsLoss = null;
      let dflLoss = null;
      let gpuMem = null;
      
      if (numbers && numbers.length >= 3) {
        // Skip epoch/total (first two numbers), next numbers are likely GPU mem, losses
        // Format varies, but typically: epoch/total GPU_mem box_loss cls_loss dfl_loss ...
        const numIndex = cleanLine.indexOf(epochMatch[0]) + epochMatch[0].length;
        const afterEpoch = cleanLine.substring(numIndex);
        const afterNumbers = afterEpoch.match(/[\d\.]+[GM]?/g);
        if (afterNumbers && afterNumbers.length >= 3) {
          gpuMem = afterNumbers[0];
          boxLoss = parseFloat(afterNumbers[1]) || null;
          clsLoss = parseFloat(afterNumbers[2]) || null;
          if (afterNumbers.length >= 4) {
            dflLoss = parseFloat(afterNumbers[3]) || null;
          }
        }
      }
      
      return {
        epoch: epoch,
        total_epochs: total,
        progress: epoch / total,
        status: "training",
        box_loss: boxLoss,
        cls_loss: clsLoss,
        dfl_loss: dflLoss,
        gpu_mem: gpuMem
      };
    }
  }
  
  return null;
}

function parseYOLOProgress(line) {
  // YOLO outputs progress in specific format:
  // Training progress line: "     1/10      1.2G     0.02345    0.01234    0.00567       1234       640"
  // Header line: "Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size"
  // Validation line: "                all        1234      0.234      0.567      0.123      0.456"
  // Speed line: "55.0s<2.7s" or "5.2it/s"
  // 
  // We need to be VERY careful - YOLO outputs model summary lines like:
  // "from  n    params  module"
  // "0                  -1  1       464  ultralytics.nn.modules.conv.Conv"
  // These have numbers that could be mistaken for epochs!
  
  // Strip ANSI escape codes first
  const ansiRegex = /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;
  const strippedLine = line.replace(ansiRegex, '');
  
  // Strip progress bar characters
  const progressBarRegex = /[━─═─┈┉┊┋┌┐└┘├┤┬┴┼│█▉▊▋▌▍▎▏]/g;
  const cleanLine = strippedLine.replace(progressBarRegex, '').trim();
  
  // Add debug logging
  if (cleanLine.length > 0 && cleanLine.length < 200) {
    console.log(`[PROGRESS PARSER] Checking line: "${cleanLine.substring(0, 100)}${cleanLine.length > 100 ? '...' : ''}"`);
  }
  
  // Check for epoch header line - indicates training has started
  const epochHeaderMatch = /^Epoch\s+GPU_mem\s+box_loss\s+cls_loss\s+dfl_loss/i.test(cleanLine);
  if (epochHeaderMatch) {
    // Training has started, send a progress update to trigger visualization
    // Use the last known epoch/total or defaults
    return {
      epoch: lastParsedEpoch || 0,
      total_epochs: lastParsedTotalEpochs || expectedTotalEpochs || 10,
      progress: Math.max(0.001, (lastParsedEpoch || 0) / (lastParsedTotalEpochs || expectedTotalEpochs || 10)),
      status: "training",
      training_started: true
    };
  }
  
  // Enhanced skip patterns for model summary lines
  const skipPatterns = [
    /from\s+n\s+params\s+module/i,
    /^\s*\d+\s+-1\s+\d+/,  // Model layer lines like "0 -1 1 464"
    /ultralytics\.nn\.modules\./i,
    /torch\.nn\.modules\./i,
    /summary:/i,
    /parameters:/i,
    /gradients:/i,
    /gflops:/i,
    /arguments\s*\[/i,
    /module\s+arguments/i,
    /^\s*[\d\-]+\s+\[[\d\s,\-]+\]\s+\d+/,  // Lines like "-1 [4, 5] 1 0"
    /^\s*\d+\s+\d+\s+\d+\s+\d+.*ultralytics/i  // Layer definition lines
  ];
  
  // Check if this is a model summary line (use original line for skip patterns)
  if (skipPatterns.some(pattern => pattern.test(line))) {
    return null;
  }
  
  // Training progress lines MUST:
  // 1. Start with spaces followed by epoch pattern: "     1/10"
  // 2. Have epoch/total format at the start
  // 3. Be followed by numbers (GPU memory, losses, etc.)
  // 4. NOT be part of model summary
  
  // Only match lines that start with spaces and epoch pattern (standard YOLO training output)
  // Use cleaned line (with ANSI codes removed) for matching
  const epochMatch = cleanLine.match(/^\s*(\d+)\/(\d+)\s+/);
  
  if (epochMatch) {
    const epoch = parseInt(epochMatch[1]);
    const totalEpochs = parseInt(epochMatch[2]);
    
    // Strict validation:
    // - Epoch must be > 0 and <= totalEpochs
    // - Total epochs must be reasonable (<= 1000)
    // - Epoch must not exceed expected total if we have it
    if (epoch > 0 && epoch <= totalEpochs && totalEpochs <= 1000) {
      const validatedTotalEpochs = expectedTotalEpochs || totalEpochs;
      
      // Critical validation: epoch must not exceed expected total
      if (expectedTotalEpochs) {
        if (epoch > expectedTotalEpochs || totalEpochs > expectedTotalEpochs * 2) {
          console.log(`[YOLO Progress] Rejecting invalid epoch: ${epoch}/${totalEpochs} (expected ${expectedTotalEpochs})`);
          return null;
        }
      }
      
      // Additional safety: if epoch is way too high, reject it
      if (epoch > 1000 || totalEpochs > 1000) {
        console.log(`[YOLO Progress] Rejecting unreasonable epoch: ${epoch}/${totalEpochs}`);
        return null;
      }
      
      // Extract all numeric values from the line
      // Format: "epoch/total  GPU_mem  box_loss  cls_loss  dfl_loss  instances  size"
      const parts = cleanLine.split(/\s+/);
      
      let gpuMemory = null;
      let boxLoss = null;
      let clsLoss = null;
      let dflLoss = null;
      let instances = null;
      let size = null;
      
      if (parts.length >= 7) {
        // Parse GPU memory (e.g., "1.2G" or "1024M")
        const gpuMemStr = parts[1];
        if (gpuMemStr && gpuMemStr.match(/^\d+\.?\d*[GM]?$/i)) {
          gpuMemory = gpuMemStr;
        }
        
        // Parse losses (floats)
        boxLoss = parseFloat(parts[2]) || null;
        clsLoss = parseFloat(parts[3]) || null;
        dflLoss = parseFloat(parts[4]) || null;
        
        // Parse instances and size (integers)
        instances = parseInt(parts[5]) || null;
        size = parseInt(parts[6]) || null;
      } else {
        // Fallback: try to extract numbers
        const numbers = line.match(/[\d\.]+/g);
        if (numbers && numbers.length >= 2) {
          boxLoss = parseFloat(numbers[1]);
        }
      }
      
      // Use box_loss as primary loss for backward compatibility
      const loss = boxLoss;
      
      console.log(`[YOLO Progress] Parsed: epoch ${epoch}/${validatedTotalEpochs}, box_loss: ${boxLoss}, cls_loss: ${clsLoss}, dfl_loss: ${dflLoss}`);
      
      // Store for header line detection
      lastParsedEpoch = epoch;
      lastParsedTotalEpochs = validatedTotalEpochs;
      
      return {
        epoch: epoch,
        total_epochs: validatedTotalEpochs,
        loss: loss,
        box_loss: boxLoss,
        cls_loss: clsLoss,
        dfl_loss: dflLoss,
        gpu_mem: gpuMemory,
        gpu_memory: gpuMemory, // Also include for backward compatibility
        instances: instances,
        size: size,
        accuracy: null,
        progress: epoch / validatedTotalEpochs,
        status: "training"
      };
    }
  }
  
  // Try to parse validation/metrics lines (mAP values)
  // Format: "                all        1234      0.234      0.567      0.123      0.456"
  const validationMatch = line.match(/^\s+all\s+/);
  if (validationMatch) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 6) {
      // Typically: "all", instances, mAP50, mAP50-95, precision, recall
      const map50 = parseFloat(parts[2]) || null;
      const map5095 = parseFloat(parts[3]) || null;
      const precision = parseFloat(parts[4]) || null;
      const recall = parseFloat(parts[5]) || null;
      
      if (map50 !== null || map5095 !== null) {
        return {
          type: "validation",
          map50: map50,
          map5095: map5095,
          precision: precision,
          recall: recall,
          f1: (precision !== null && recall !== null) ? (2 * precision * recall / (precision + recall)) : null
        };
      }
    }
  }
  
  // Try to parse speed/ETA lines (e.g., "55.0s<2.7s" or "5.2it/s")
  const speedMatch = line.match(/(\d+\.?\d*)(it\/s|s<[\d\.]+s)/);
  if (speedMatch) {
    return {
      type: "speed",
      speed: speedMatch[1] + (speedMatch[2].includes('it/s') ? ' it/s' : '')
    };
  }
  
  return null;
}

function findPythonPath() {
  // Priority 1: Check for bundled Python (for distribution)
  // In packaged apps, Python should be in resources/python/ (outside app.asar)
  
  // Get the actual app path - different in dev vs packaged
  // Use process.resourcesPath if available (electron-builder), otherwise use execDir/resources (electron-packager)
  const isPackaged = app.isPackaged;
  const execDir = path.dirname(process.execPath);  // Define execDir in outer scope
  let resourcesPythonPath;
  
  if (isPackaged && process.resourcesPath) {
    // electron-builder: resources are in process.resourcesPath
    resourcesPythonPath = path.join(process.resourcesPath, 'python', 'python.exe');
  } else {
    // electron-packager or dev: resources are in execDir/resources
    resourcesPythonPath = path.resolve(execDir, 'resources', 'python', 'python.exe');
  }
  
  const bundledPaths = [
    // For packaged Electron app - resources folder (outside app.asar)
    // This is the primary path for electron-packager builds
    resourcesPythonPath,
    
    // Development paths (when running from source)
    path.resolve(__dirname, 'python', 'python.exe'),
    path.resolve(__dirname, '..', 'python', 'python.exe'),
    
    // Fallback paths
    path.resolve(execDir, 'python', 'python.exe')
  ].filter(p => p !== null);
  
  // Debug logging
  console.log('=== Python Detection Debug ===');
  console.log('process.execPath:', process.execPath);
  console.log('execDir:', execDir);
  console.log('app.isPackaged:', app.isPackaged);
  console.log('Primary path to check:', resourcesPythonPath);
  
  // Prefer Python Launcher (py.exe) resolution for Windows Store installs
  const resolvedFromPy = resolvePythonFromPyLauncher();
  if (resolvedFromPy) {
    console.log('✓ Found Python via py -3:', resolvedFromPy);
    return resolvedFromPy;
  }

  for (const pyPath of bundledPaths) {
    if (pyPath) {
      const normalized = path.normalize(pyPath);
      const exists = fs.existsSync(normalized);
      console.log(`Checking: ${normalized} - ${exists ? 'EXISTS ✓' : 'NOT FOUND ✗'}`);
      if (exists) {
        console.log('✓ Found bundled Python at:', normalized);
        // Verify ultralytics is available in this Python
        const pythonLibPath = path.join(path.dirname(normalized), 'Lib');
        const sitePackagesPath = path.join(pythonLibPath, 'site-packages');
        const ultralyticsPath = path.join(sitePackagesPath, 'ultralytics');
        console.log('  Site-packages:', sitePackagesPath);
        console.log('  ultralytics exists:', fs.existsSync(ultralyticsPath));
        return normalized;
      }
    }
  }
  const appDirPython = findPythonInDirectory(execDir, 3);
  if (appDirPython) {
    console.log('Found python.exe inside app directory:', appDirPython);
    return appDirPython;
  }
  console.log('=== End Python Detection Debug ===');
  
  // Priority 2: Check system Python installations
  const userHomeInfo = getUserHomePath();
  const userHome = userHomeInfo.path || '';
  const userProfile = process.env.USERPROFILE || userHome || '';
  const username = process.env.USERNAME
    || (userProfile ? userProfile.split('\\').pop() : '')
    || (userHome ? userHome.split('\\').pop() : '');
  if (userHomeInfo.source !== 'app.getPath') {
    console.warn('[Python Detection] user home fallback used:', userHomeInfo.source, userHomeInfo.error || '');
  }
  const localAppDataInfo = getLocalAppDataPath();
  const localAppDataPath = localAppDataInfo.path || '';
  const windowsAppsDir = localAppDataPath
    ? path.join(localAppDataPath, 'Microsoft', 'WindowsApps')
    : '';
  
  // Expanded common Python installation paths on Windows
  const commonPaths = [
    // Root installations (most common for standard installs)
    'C:\\Python313\\python.exe',
    'C:\\Python312\\python.exe',
    'C:\\Python311\\python.exe',
    'C:\\Python310\\python.exe',
    'C:\\Python39\\python.exe',
    'C:\\Python38\\python.exe',
    'C:\\Python37\\python.exe',
    'C:\\Python36\\python.exe',
    
    // Program Files installations
    'C:\\Program Files\\Python313\\python.exe',
    'C:\\Program Files\\Python312\\python.exe',
    'C:\\Program Files\\Python311\\python.exe',
    'C:\\Program Files\\Python310\\python.exe',
    'C:\\Program Files\\Python39\\python.exe',
    'C:\\Program Files\\Python38\\python.exe',
    'C:\\Program Files\\Python37\\python.exe',
    'C:\\Program Files\\Python36\\python.exe',
    
    // Program Files (x86) for 32-bit installations
    'C:\\Program Files (x86)\\Python313\\python.exe',
    'C:\\Program Files (x86)\\Python312\\python.exe',
    'C:\\Program Files (x86)\\Python311\\python.exe',
    'C:\\Program Files (x86)\\Python310\\python.exe',
    'C:\\Program Files (x86)\\Python39\\python.exe',
    'C:\\Program Files (x86)\\Python38\\python.exe',
    
    // User AppData installations (common for newer Python installs from python.org)
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python313', 'python.exe'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python312', 'python.exe'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python311', 'python.exe'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python310', 'python.exe'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python39', 'python.exe'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python38', 'python.exe'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python37', 'python.exe'),
    path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', 'Python36', 'python.exe'),

    // LocalAppData-based installations (robust when USERPROFILE is missing)
    path.join(localAppDataPath, 'Programs', 'Python', 'Python313', 'python.exe'),
    path.join(localAppDataPath, 'Programs', 'Python', 'Python312', 'python.exe'),
    path.join(localAppDataPath, 'Programs', 'Python', 'Python311', 'python.exe'),
    path.join(localAppDataPath, 'Programs', 'Python', 'Python310', 'python.exe'),
    path.join(localAppDataPath, 'Programs', 'Python', 'Python39', 'python.exe'),
    path.join(localAppDataPath, 'Programs', 'Python', 'Python38', 'python.exe'),
    path.join(localAppDataPath, 'Programs', 'Python', 'Python37', 'python.exe'),
    path.join(localAppDataPath, 'Programs', 'Python', 'Python36', 'python.exe'),
    
    // Alternative user path format
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python313\\python.exe`,
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe`,
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python311\\python.exe`,
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python310\\python.exe`,
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python39\\python.exe`,
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python38\\python.exe`,
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python37\\python.exe`,
    `C:\\Users\\${username}\\AppData\\Local\\Programs\\Python\\Python36\\python.exe`,

    // Windows Store Python (LocalAppData-based)
    path.join(localAppDataPath, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0', 'python.exe'),
    path.join(localAppDataPath, 'Microsoft', 'WindowsApps', 'python.exe'),
  ];
  
  // Check common paths first
  for (const pythonPath of commonPaths) {
    if (pythonPath && fs.existsSync(pythonPath)) {
      console.log('Found Python at:', pythonPath);
      return pythonPath;
    }
  }
  
  // Try to find Python in PATH (inject WindowsApps for packaged apps)
  try {
    const { execSync } = require('child_process');
    const env = {
      ...process.env,
      PATH: `${windowsAppsDir ? `${windowsAppsDir};` : ''}${process.env.PATH || ''}`
    };
    const pythonPaths = execSync('where python', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], env }).trim().split(/[\r\n]+/);
    for (const pathStr of pythonPaths) {
      const trimmedPath = pathStr.trim();
      const lowerTrimmed = trimmedPath.toLowerCase();
      if (lowerTrimmed.includes('\\pymanager\\')) {
        continue;
      }
      // Skip WindowsApps stub, but check if path exists and is executable
      if (trimmedPath && !trimmedPath.includes('WindowsApps') && trimmedPath.length > 0 && fs.existsSync(trimmedPath)) {
        console.log('Found Python in PATH at:', trimmedPath);
        return trimmedPath;
      }
    }
  } catch (e) {
    // PATH check failed, continue
  }
  
  // Try python3 command as well
  try {
    const { execSync } = require('child_process');
    const env = {
      ...process.env,
      PATH: `${windowsAppsDir ? `${windowsAppsDir};` : ''}${process.env.PATH || ''}`
    };
    const python3Paths = execSync('where python3', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], env }).trim().split(/[\r\n]+/);
    for (const pathStr of python3Paths) {
      const trimmedPath = pathStr.trim();
      const lowerTrimmed = trimmedPath.toLowerCase();
      if (lowerTrimmed.includes('\\pymanager\\')) {
        continue;
      }
      if (trimmedPath && !trimmedPath.includes('WindowsApps') && trimmedPath.length > 0 && fs.existsSync(trimmedPath)) {
        console.log('Found Python3 in PATH at:', trimmedPath);
        return trimmedPath;
      }
    }
  } catch (e) {
    // python3 check failed, continue
  }

  // Try Python launcher (py.exe) as a last resort
  try {
    const { execSync } = require('child_process');
    const env = {
      ...process.env,
      PATH: `${windowsAppsDir ? `${windowsAppsDir};` : ''}${process.env.PATH || ''}`
    };
    const pyPaths = execSync('where py', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], env }).trim().split(/[\r\n]+/);
    for (const pathStr of pyPaths) {
      const trimmedPath = pathStr.trim();
      if (trimmedPath && fs.existsSync(trimmedPath)) {
        try {
          const resolved = execSync('py -3 -c "import sys; print(sys.executable)"', {
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore'],
            timeout: 5000,
            env
          }).trim();
          if (resolved && fs.existsSync(resolved)) {
            console.log('Found Python via py.exe:', resolved);
            return resolved;
          }
        } catch (e) {
          // py.exe found but failed to resolve
        }
      }
    }
  } catch (e) {
    // py launcher check failed, continue
  }

  // Directly try common py.exe locations (works even if PATH is stripped)
  const resolvedDirectPy = resolvePythonFromPyLauncher();
  if (resolvedDirectPy) {
    console.log('Found Python via direct py.exe:', resolvedDirectPy);
    return resolvedDirectPy;
  }
  
  return null;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    backgroundColor: '#000000',
    frame: false,
    transparent: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    show: false
  });

  mainWindow.loadFile('index.html');

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // DevTools disabled - removed Ctrl+Shift+I shortcut

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Handle window controls
ipcMain.on('close-window', () => {
  if (mainWindow) {
    mainWindow.close();
  }
});

// Handle model saving
ipcMain.on('save-model', (event, modelData, modelPurpose, framework, modelVariant, modelFormat) => {
  try {
    // Create models directory in user's documents
    const modelsDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'Models');
    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
    }
    
    // Generate filename with date, purpose, framework, and variant
    const now = new Date();
    const dateStr = now.toISOString().replace(/:/g, '-').split('.')[0]; // Format: YYYY-MM-DDTHH-MM-SS
    
    // Add checkpoint suffix if partial
    const checkpointSuffix = modelData.isPartial ? '_checkpoint' : '';
    
    // Get file extension based on format
    const formatExtensions = {
      'pt': '.pt',
      'pth': '.pth',
      'h5': '.h5',
      'onnx': '.onnx',
      'pkl': '.pkl',
      'joblib': '.joblib',
      'json': '.json',
      'txt': '.txt',
      'pb': '.pb',
      'torchscript': '.torchscript'
    };
    
    const purposeNames = {
      'computer_vision': 'cv',
      'machine_learning': 'ml',
      'natural_language_processing': 'nlp',
      'reinforcement_learning': 'rl',
      'time_series': 'ts',
      'generative': 'gen'
    };
    
    const purpose = purposeNames[modelPurpose] || 'model';
    const extension = formatExtensions[modelFormat] || '.pt';
    // Build filename: purpose_framework_variant_date[checkpoint].extension
    // Example: cv_yolo_yolov11n_2024-01-15T14-30-45_checkpoint.pt
    // checkpointSuffix is already defined above
    const filename = `${purpose}_${framework}_${modelVariant}_${dateStr}${checkpointSuffix}${extension}`;
    const filepath = path.join(modelsDir, filename);
    
    // Save model based on format
    if (modelFormat === 'pickle' || modelFormat === 'joblib') {
      // For pickle/joblib, save as binary (though in real app, this would be actual model weights)
      // Here we'll save metadata as JSON and simulate binary format
      const metadata = {
        purpose: modelPurpose,
        format: modelFormat,
        ...modelData
      };
      fs.writeFileSync(filepath, JSON.stringify(metadata, null, 2));
    } else {
      // For other formats, save structured data
      // In a real implementation, this would save actual model weights
      const modelFile = {
        format: modelFormat,
        purpose: modelPurpose,
        metadata: {
          created: new Date().toISOString(),
          trainingSettings: modelData.trainingSettings,
          finalMetrics: modelData.finalMetrics
        },
        trainingHistory: modelData.trainingHistory,
        // In real implementation, this would contain actual model weights/architecture
        modelWeights: 'BASE64_ENCODED_WEIGHTS_PLACEHOLDER',
        architecture: 'MODEL_ARCHITECTURE_PLACEHOLDER'
      };
      
      // Save as JSON for now (in real app, would use proper serialization)
      fs.writeFileSync(filepath, JSON.stringify(modelFile, null, 2));
    }
    
    // Send back the filepath and whether it's a checkpoint
    event.reply('model-saved', filepath, modelData.isPartial || false);
  } catch (error) {
    event.reply('model-save-error', error.message);
  }
});

// Handle opening file location
ipcMain.on('open-model-location', (event, filepath) => {
  try {
    if (!filepath) {
      console.error('No filepath provided to open-model-location');
      return;
    }
    
    // Normalize the path and check if it exists
    const normalizedPath = path.normalize(filepath);
    console.log('[Main] Opening model location:', normalizedPath);
    
    // If the path is to a file, show the file in its folder
    if (fs.existsSync(normalizedPath)) {
      console.log('[Main] File exists, showing in folder');
      shell.showItemInFolder(normalizedPath);
    } else {
      // If file doesn't exist, try to open the parent directory
      const parentDir = path.dirname(normalizedPath);
      console.log('[Main] File does not exist, trying parent directory:', parentDir);
      
      if (fs.existsSync(parentDir)) {
        console.log('[Main] Parent directory exists, opening it');
        shell.openPath(parentDir);
      } else {
        // Try opening the weights directory if it exists
        const weightsDir = path.join(parentDir, 'weights');
        if (fs.existsSync(weightsDir)) {
          console.log('[Main] Weights directory exists, opening it');
          shell.openPath(weightsDir);
        } else {
          // Try opening the models base directory
          const modelsDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'Models');
          if (fs.existsSync(modelsDir)) {
            console.log('[Main] Opening Models directory:', modelsDir);
            shell.openPath(modelsDir);
          } else {
            console.error('Cannot open model location - path does not exist:', normalizedPath);
            mainWindow.webContents.send('training-log', { 
              level: 'error', 
              message: `Cannot open model location - path does not exist: ${normalizedPath}` 
            });
          }
        }
      }
    }
  } catch (error) {
    console.error('Error opening model location:', error);
    mainWindow.webContents.send('training-log', { 
      level: 'error', 
      message: `Error opening model location: ${error.message}` 
    });
  }
});

// Handle listing saved models
// Recursive function to find artifact folders
function findArtifactFolders(rootDir, maxDepth = 3, currentDepth = 0) {
  const artifacts = [];
  
  if (currentDepth >= maxDepth || !fs.existsSync(rootDir)) {
    return artifacts;
  }
  
  try {
    const items = fs.readdirSync(rootDir);
    
    for (const item of items) {
      const itemPath = path.join(rootDir, item);
      
      try {
        const stats = fs.statSync(itemPath);
        
        if (stats.isDirectory()) {
          // Check if this directory is an artifact folder
          const metadataPath = path.join(itemPath, 'metadata.json');
          const schemaPath = path.join(itemPath, 'schema.json');
          const modelFiles = ['model.pkl', 'model.joblib', 'model.pth'];
          
          let hasModelFile = false;
          let foundModelFile = null;
          for (const mf of modelFiles) {
            if (fs.existsSync(path.join(itemPath, mf))) {
              hasModelFile = true;
              foundModelFile = mf;
              break;
            }
          }
          
          // This is an artifact folder if it has metadata.json, schema.json, and a model file
          if (fs.existsSync(metadataPath) && fs.existsSync(schemaPath) && hasModelFile) {
            console.log(`[Model Discovery] Found artifact folder: ${itemPath}`);
            console.log(`[Model Discovery]   - metadata.json: exists`);
            console.log(`[Model Discovery]   - schema.json: exists`);
            console.log(`[Model Discovery]   - model file: ${foundModelFile}`);
            
            try {
              const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
              const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
              
              console.log(`[Model Discovery]   - model_type: ${metadata.model_type || 'N/A'}`);
              console.log(`[Model Discovery]   - framework: ${metadata.framework || 'N/A'}`);
              console.log(`[Model Discovery]   - algorithm: ${metadata.algorithm || 'N/A'}`);
              
              artifacts.push({
                artifactDir: itemPath,
                metadataPath: metadataPath,
                schemaPath: schemaPath,
                modelPath: path.join(itemPath, foundModelFile),
                metadata: metadata,
                schema: schema,
                foundModelFile: foundModelFile
              });
            } catch (e) {
              console.error(`[Model Discovery] Error reading artifact ${itemPath}:`, e.message);
            }
          } else {
            // Not an artifact folder, recurse into it (but only if not too deep)
            if (currentDepth < maxDepth - 1) {
              const subArtifacts = findArtifactFolders(itemPath, maxDepth, currentDepth + 1);
              artifacts.push(...subArtifacts);
            }
          }
        }
      } catch (e) {
        // Skip items we can't read
        console.warn(`[Model Discovery] Skipping ${itemPath}: ${e.message}`);
      }
    }
  } catch (e) {
    console.error(`[Model Discovery] Error reading directory ${rootDir}:`, e.message);
  }
  
  return artifacts;
}

ipcMain.on('list-models', (event) => {
  try {
    const modelsDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'Models');
    console.log(`[Model Discovery] Scanning for models in: ${modelsDir}`);
    
    if (!fs.existsSync(modelsDir)) {
      console.log(`[Model Discovery] Models directory does not exist`);
      event.reply('models-list', []);
      return;
    }

    // Recursively find all artifact folders
    const artifactFolders = findArtifactFolders(modelsDir, 3);
    console.log(`[Model Discovery] Found ${artifactFolders.length} artifact folders`);
    
    const models = [];

    artifactFolders.forEach(artifact => {
      try {
        const { artifactDir, metadata, schema, foundModelFile } = artifact;
        
        // Determine if this is a tabular model
        const isTabular = 
          metadata.model_type === 'tabular' ||
          (metadata.model_type && metadata.model_type.startsWith('sklearn_')) ||
          (metadata.framework === 'sklearn' && metadata.algorithm);
        
        console.log(`[Model Discovery] Processing artifact: ${path.basename(artifactDir)}`);
        console.log(`[Model Discovery]   - isTabular: ${isTabular}`);
        
        if (!isTabular) {
          console.log(`[Model Discovery]   - Excluded: Not a tabular model (model_type=${metadata.model_type}, framework=${metadata.framework})`);
        }
        
        // Create model entry
        const modelEntry = {
          filename: path.basename(artifactDir),
          filepath: artifactDir, // Use artifact folder path, not parent
          isArtifact: true,
          isCheckpoint: false,
          currentEpoch: 0,
          totalEpochs: 0,
          accuracy: 0,
          loss: 0,
          timestamp: metadata.trained_at || fs.statSync(artifactDir).mtime.toISOString(),
          metadata: metadata,
          schema: schema,
          modelType: metadata.model_type,
          framework: metadata.framework,
          algorithm: metadata.algorithm,
          modelPath: artifact.modelPath,
          schemaPath: artifact.schemaPath,
          metadataPath: artifact.metadataPath
        };
        
        models.push(modelEntry);
        console.log(`[Model Discovery]   - Added to models list`);
      } catch (e) {
        console.error(`[Model Discovery] Error processing artifact ${artifact.artifactDir}:`, e.message);
      }
    });

    // Also check for legacy JSON model files in root
    try {
      const items = fs.readdirSync(modelsDir);
      items.forEach(item => {
        const itempath = path.join(modelsDir, item);
        const stats = fs.statSync(itempath);
        
        if (stats.isFile() && item.endsWith('.json')) {
          try {
            const content = fs.readFileSync(itempath, 'utf8');
            const modelData = JSON.parse(content);
            
            models.push({
              filename: item,
              filepath: itempath,
              isArtifact: false,
              isCheckpoint: modelData.isPartial || item.includes('_checkpoint'),
              currentEpoch: modelData.currentEpoch || (modelData.finalMetrics ? modelData.finalMetrics.epochs : 0),
              totalEpochs: modelData.totalEpochs || (modelData.finalMetrics ? modelData.finalMetrics.epochs : 0),
              accuracy: modelData.finalMetrics ? modelData.finalMetrics.accuracy : 0,
              loss: modelData.finalMetrics ? modelData.finalMetrics.loss : 0,
              timestamp: modelData.timestamp || stats.mtime.toISOString(),
              trainingHistory: modelData.trainingHistory || [],
              trainingSettings: modelData.trainingSettings || {},
              modelConfig: modelData.modelConfig || {}
            });
          } catch (e) {
            // Skip files that aren't valid JSON models
          }
        }
      });
    } catch (e) {
      console.warn(`[Model Discovery] Error reading legacy files:`, e.message);
    }

    // Sort by timestamp, newest first
    models.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    console.log(`[Model Discovery] Total models found: ${models.length}`);
    const tabularCount = models.filter(m => 
      m.isArtifact && (
        m.metadata?.model_type === 'tabular' ||
        (m.metadata?.model_type && m.metadata.model_type.startsWith('sklearn_')) ||
        (m.metadata?.framework === 'sklearn' && m.metadata?.algorithm)
      )
    ).length;
    console.log(`[Model Discovery] Tabular models: ${tabularCount}`);

    event.reply('models-list', models);
  } catch (error) {
    console.error(`[Model Discovery] Fatal error:`, error);
    event.reply('models-list-error', error.message);
  }
});

// Handle loading a model (supports both artifact folders and legacy single files)
ipcMain.on('load-model', async (event, filepath) => {
  try {
    const path = require('path');
    const stats = fs.statSync(filepath);
    
    // Check if it's an artifact folder (contains metadata.json)
    const metadataPath = path.join(filepath, 'metadata.json');
    const isArtifact = stats.isDirectory() && fs.existsSync(metadataPath);
    
    if (isArtifact) {
      // Load artifact: validate compatibility first
      try {
        const metadataContent = fs.readFileSync(metadataPath, 'utf8');
        const metadata = JSON.parse(metadataContent);
        
        // Validate compatibility (call Python validation function)
        // spawn is already declared at module level
        const pythonScript = path.join(__dirname, 'resources', 'trainer.py');
        
        // Create validation command
        const validationCmd = {
          type: 'validate_artifact',
          artifact_path: filepath
        };
        
        // For now, do basic validation in Node.js
        const pythonVersion = process.versions.python || 'unknown';
        const savedPyMajor = parseInt(metadata.python_version?.split('.')[0] || '0');
        const currentPyMajor = parseInt(pythonVersion.split('.')[0] || '0');
        
        let canLoad = true;
        let issues = [];
        let warnings = [];
        
        if (savedPyMajor !== currentPyMajor && savedPyMajor > 0 && currentPyMajor > 0) {
          issues.push(`Python major version mismatch: saved=${metadata.python_version}, current=${pythonVersion}`);
          canLoad = false;
        }
        
        if (!canLoad) {
          event.reply('model-load-error', {
            error: 'Version compatibility error',
            message: `Cannot load model: ${issues.join('; ')}`,
            issues: issues,
            warnings: warnings,
            metadata: metadata
          });
          return;
        }
        
        if (warnings.length > 0) {
          // Send warnings but allow load
          event.reply('model-load-warning', {
            warnings: warnings,
            metadata: metadata
          });
        }
        
        // Load schema if available
        const schemaPath = path.join(filepath, 'schema.json');
        let schema = null;
        if (fs.existsSync(schemaPath)) {
          schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
        }
        
        // Find model file (model.joblib or model.pkl)
        const modelFiles = ['model.joblib', 'model.pkl'];
        let modelFile = null;
        for (const mf of modelFiles) {
          const mfPath = path.join(filepath, mf);
          if (fs.existsSync(mfPath)) {
            modelFile = mfPath;
            break;
          }
        }
        
        if (!modelFile) {
          throw new Error('Model file not found in artifact folder');
        }
        
        // Return artifact info
        event.reply('model-loaded', {
          isArtifact: true,
          artifactPath: filepath,
          modelPath: modelFile,
          metadata: metadata,
          schema: schema
        }, filepath);
        
      } catch (error) {
        event.reply('model-load-error', {
          error: 'Failed to load artifact',
          message: error.message
        });
      }
    } else {
      // Legacy single file (JSON metadata)
      const content = fs.readFileSync(filepath, 'utf8');
      const modelData = JSON.parse(content);
      event.reply('model-loaded', modelData, filepath);
    }
  } catch (error) {
    event.reply('model-load-error', {
      error: 'Failed to load model',
      message: error.message
    });
  }
});

// Handle CV inference
ipcMain.handle('cv-infer', async (event, config) => {
  return runPythonInfer({ ...config, task: 'cv' }, { label: 'CV Inference' });
});

// Handle tabular inference
ipcMain.handle('tabular-infer', async (event, config) => {
  return runPythonInfer({ ...config, task: 'tabular' }, { label: 'Tabular Inference' });
});

// Handle listing CV models (YOLO runs)
ipcMain.handle('list-cv-models', async (event) => {
  try {
    const modelsDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'Models');
    const runsDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'runs');
    
    const models = [];

    const tryAddModel = (modelPath, nameHint) => {
      if (!fs.existsSync(modelPath)) return;
      const stats = fs.statSync(modelPath);
      models.push({
        name: nameHint,
        path: modelPath,
        timestamp: stats.mtime.toISOString(),
        type: 'yolo'
      });
    };

    const scanForBestWeights = (baseDir, namePrefix = '') => {
      if (!fs.existsSync(baseDir)) return;
      const entries = fs.readdirSync(baseDir);
      entries.forEach(entry => {
        const entryPath = path.join(baseDir, entry);
        let entryStat;
        try {
          entryStat = fs.statSync(entryPath);
        } catch {
          return;
        }
        if (!entryStat.isDirectory()) return;

        // Direct weights in this folder
        tryAddModel(path.join(entryPath, 'weights', 'best.pt'), `${namePrefix}${entry}`);

        // Train/exp style subfolders (train, train2, exp, exp2, etc.)
        const subEntries = fs.readdirSync(entryPath);
        subEntries.forEach(sub => {
          const subPath = path.join(entryPath, sub);
          try {
            if (!fs.statSync(subPath).isDirectory()) return;
          } catch {
            return;
          }
          if (sub.toLowerCase().startsWith('train') || sub.toLowerCase().startsWith('exp')) {
            tryAddModel(path.join(subPath, 'weights', 'best.pt'), `${namePrefix}${entry}/${sub}`);
          }
        });
      });
    };

    // Check Models directory (current training output location)
    scanForBestWeights(modelsDir);
    
    // Check runs directory for YOLO training outputs
    scanForBestWeights(runsDir);
    
    // Sort by timestamp, newest first
    models.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    return models;
  } catch (error) {
    return [];
  }
});

// Handle selecting image file
ipcMain.handle('select-image-file', async (event) => {
  const { dialog } = require('electron');
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// Handle selecting CSV file
ipcMain.handle('select-csv-file', async (event) => {
  const { dialog } = require('electron');
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'CSV Files', extensions: ['csv'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// Handle opening folder
ipcMain.handle('open-folder', async (event, folderPath) => {
  const { shell } = require('electron');
  shell.openPath(folderPath);
  return { success: true };
});

// Handle selecting dataset directory or file
ipcMain.handle('select-dataset-directory', async (event, options = {}) => {
  try {
    const { modelPurpose, allowFiles = false } = options;
    
    // For tabular data, allow selecting CSV files or folders
    if (modelPurpose === 'tabular' || allowFiles) {
      // Use openFile to allow CSV file selection
      // Note: Electron doesn't support both openFile and openDirectory in same dialog
      // So we'll use openFile and handle folders separately if needed
      const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select CSV File (or navigate to folder and click Open)',
        properties: ['openFile'],
        filters: [
          { name: 'CSV Files', extensions: ['csv'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      });
      
      // If cancelled, offer folder selection
      if (result.canceled) {
        const folderResult = await dialog.showOpenDialog(mainWindow, {
          title: 'Select Folder Containing CSV Files',
          properties: ['openDirectory'],
          buttonLabel: 'Select Folder'
        });
        return folderResult;
      }
      
      return result;
    } else {
      // For other model types, only allow directory selection
      const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Dataset Directory',
        properties: ['openDirectory']
      });
      return result;
    }
  } catch (error) {
    return { canceled: true, error: error.message };
  }
});

// Handle checking dataset format
ipcMain.handle('check-dataset-format', async (event, folderPath) => {
  try {
    const files = fs.readdirSync(folderPath);
    const jsonFiles = files.filter(f => f.toLowerCase().endsWith('.json'));
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
    const imageFiles = files.filter(f => {
      const ext = path.extname(f).toLowerCase();
      return imageExtensions.includes(ext);
    });
    
    // LabelMe format: has JSON files and image files with matching names
    const isLabelMe = jsonFiles.length > 0 && imageFiles.length > 0;
    
    return {
      isLabelMe: isLabelMe,
      jsonCount: jsonFiles.length,
      imageCount: imageFiles.length
    };
  } catch (error) {
    return { isLabelMe: false, error: error.message };
  }
});

// Handle getting prepared dataset path
ipcMain.handle('get-prepared-dataset-path', async (event, folderName) => {
  const preparedDir = path.join(os.homedir(), 'Documents', 'UniTrainer', 'PreparedDatasets');
  if (!fs.existsSync(preparedDir)) {
    fs.mkdirSync(preparedDir, { recursive: true });
  }
  return path.join(preparedDir, folderName + '_yolo');
});

// Handle preparing dataset (converting LabelMe JSON to YOLO format)
ipcMain.handle('prepare-dataset', async (event, inputFolder, outputFolder) => {
  return new Promise((resolve) => {
    try {
      // Try multiple paths for the script (development and packaged)
      const execDir = path.dirname(process.execPath);
      const possiblePaths = [
        path.join(__dirname, 'python', 'prepare_dataset.py'), // Development
        path.join(execDir, 'resources', 'python', 'prepare_dataset.py'), // Packaged (electron-packager)
        path.join(execDir, 'python', 'prepare_dataset.py'), // Alternative packaged path
        path.resolve(__dirname, '..', 'python', 'prepare_dataset.py') // Fallback
      ];
      
      let pythonScript = null;
      for (const scriptPath of possiblePaths) {
        if (fs.existsSync(scriptPath)) {
          pythonScript = scriptPath;
          console.log('[Main] Found prepare_dataset.py at:', pythonScript);
          break;
        }
      }
      
      // Check if script exists
      if (!pythonScript) {
        console.error('prepare_dataset.py not found. Checked paths:');
        possiblePaths.forEach(p => console.error('  -', p));
        resolve({ success: false, error: 'Dataset preparation script not found. Please ensure prepare_dataset.py is in the python folder.' });
        return;
      }
      
      const pythonCmd = findPythonPath();
      if (!pythonCmd || !fs.existsSync(pythonCmd)) {
        resolve({ success: false, error: 'Python not found. Please install Python 3.8+' });
        return;
      }
      
      console.log('[Main] Preparing dataset...');
      console.log('  Input folder:', inputFolder);
      console.log('  Output folder:', outputFolder);
      console.log('  Python:', pythonCmd);
      console.log('  Script:', pythonScript);
      
      const prepareProcess = spawn(pythonCmd, [
        pythonScript,
        inputFolder,
        outputFolder,
        '--class-name', 'person'
      ], {
        cwd: path.dirname(pythonScript),
        stdio: ['pipe', 'pipe', 'pipe']
      });
      
      let stdout = '';
      let stderr = '';
      
      prepareProcess.stdout.on('data', (data) => {
        const output = data.toString();
        stdout += output;
        console.log('[Dataset Prep]', output.trim());
        // Send progress updates to renderer
        mainWindow.webContents.send('dataset-prep-log', output);
      });
      
      prepareProcess.stderr.on('data', (data) => {
        const output = data.toString();
        stderr += output;
        console.error('[Dataset Prep Error]', output.trim());
        mainWindow.webContents.send('dataset-prep-log', output);
      });
      
      prepareProcess.on('close', (code) => {
        console.log('[Main] Dataset preparation process exited with code:', code);
        if (code === 0) {
          // Check if data.yaml was created
          const dataYamlPath = path.join(outputFolder, 'data.yaml');
          if (fs.existsSync(dataYamlPath)) {
            resolve({ 
              success: true, 
              outputFolder: outputFolder,
              dataYamlPath: dataYamlPath,
              message: 'Dataset prepared successfully'
            });
          } else {
            resolve({ 
              success: false, 
              error: 'Dataset preparation completed but data.yaml not found',
              stdout: stdout,
              stderr: stderr
            });
          }
        } else {
          resolve({ 
            success: false, 
            error: `Dataset preparation failed with exit code ${code}`,
            stdout: stdout,
            stderr: stderr
          });
        }
      });
      
      prepareProcess.on('error', (error) => {
        console.error('[Main] Error starting dataset preparation:', error);
        resolve({ 
          success: false, 
          error: `Failed to start dataset preparation: ${error.message}` 
        });
      });
      
    } catch (error) {
      console.error('[Main] Exception in prepare-dataset handler:', error);
      resolve({ success: false, error: error.message });
    }
  });
});

// Store CanopyWave API client instances (keyed by API key)
const canopywaveClients = new Map();

// Validate CanopyWave API key
ipcMain.handle('validate-canopywave-api-key', async (event, apiKey) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { valid: false, error: 'API key is required' };
    }

    // Create or reuse API client
    let client = canopywaveClients.get(apiKey);
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // Validate API key (now returns {valid, error} object)
    const validationResult = await client.validateAPIKey();
    
    if (validationResult.valid) {
      console.log('[Main] API key validated successfully');
      return { valid: true };
    } else {
      console.error('[Main] API key validation failed:', validationResult.error);
      return { 
        valid: false, 
        error: validationResult.error || 'Invalid API key. Please check your credentials.' 
      };
    }
  } catch (error) {
    console.error('[Main] Error validating CanopyWave API key:', error);
    return { valid: false, error: error.message || 'Failed to validate API key' };
  }
});

// Create cloud training job (launch instance)
ipcMain.handle('create-cloud-training-job', async (event, apiKey, trainingConfig) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!trainingConfig.project || !trainingConfig.region) {
      return { success: false, error: 'Project and region are required' };
    }
    if (!trainingConfig.flavor || !trainingConfig.image || !trainingConfig.password) {
      return { success: false, error: 'Flavor, image, and password are required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // Launch instance (VM) for training
    const instanceConfig = {
      project: trainingConfig.project,
      region: trainingConfig.region,
      name: trainingConfig.name || `training-${Date.now()}`,
      flavor: trainingConfig.flavor, // e.g., "H100-4"
      image: trainingConfig.image, // e.g., "GPU-Ubuntu.22.04"
      password: trainingConfig.password,
      keypair: trainingConfig.keypair, // Optional
      is_monitoring: trainingConfig.is_monitoring // Optional
    };

    const instance = await client.launchInstance(instanceConfig);
    
    return { success: true, instance: instance };
  } catch (error) {
    logAPIError('create-cloud-training-job', error);
    let errorMessage = error.message || 'Failed to create training instance';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
    }
    return { success: false, error: errorMessage };
  }
});

// Get cloud training job status (instance status)
ipcMain.handle('get-cloud-job-status', async (event, apiKey, instanceId, project, region) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!project || !region) {
      return { success: false, error: 'Project and region are required' };
    }

    // Get API client
    const client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      return { success: false, error: 'API client not found. Please login again.' };
    }

    // Get instance status
    const status = await client.getJobStatus(instanceId, project, region);
    
    return { success: true, status: status };
  } catch (error) {
    console.error('[Main] Error getting cloud job status:', error);
    return { success: false, error: error.message || 'Failed to get job status' };
  }
});

// List available GPUs/resources (instance types)
// Enhanced error logging wrapper for API handlers
function logAPIError(context, error) {
  console.error(`[Main] ${context} error:`, error);
  console.error(`[Main] ${context} error stack:`, error.stack);
  if (error.apiResponse) {
    console.error(`[Main] ${context} CanopyWave API Error Details:`, JSON.stringify(error.apiResponse, null, 2));
  }
  if (error.statusCode) {
    console.error(`[Main] ${context} HTTP Status:`, error.statusCode);
    console.error(`[Main] ${context} Endpoint:`, error.endpoint || 'unknown');
  }
  if (error.rawResponse) {
    console.error(`[Main] ${context} Raw API Response:`, error.rawResponse);
  }
  if (error.originalError) {
    console.error(`[Main] ${context} Original Error:`, error.originalError);
  }
}

ipcMain.handle('list-cloud-gpus', async (event, apiKey, project) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!project) {
      return { success: false, error: 'Project is required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // List available instance types (GPUs)
    const instanceTypes = await client.listAvailableGPUs(project);
    
    // Log the first GPU to see its structure
    if (instanceTypes && instanceTypes.length > 0) {
      console.log('[Main] Sample GPU data:', JSON.stringify(instanceTypes[0], null, 2));
      console.log('[Main] All GPU fields:', Object.keys(instanceTypes[0]));
    }
    
    return { success: true, gpus: instanceTypes };
  } catch (error) {
    logAPIError('list-cloud-gpus', error);
    let errorMessage = error.message || 'Failed to list available GPUs';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
    }
    return { success: false, error: errorMessage };
  }
});

// List regions
ipcMain.handle('list-canopywave-regions', async (event, apiKey, project) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!project) {
      return { success: false, error: 'Project is required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // List regions
    const regions = await client.listRegions(project);
    
    return { success: true, regions: regions };
  } catch (error) {
    logAPIError('list-canopywave-regions', error);
    let errorMessage = error.message || 'Failed to list regions';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
    }
    return { success: false, error: errorMessage };
  }
});

// List projects
ipcMain.handle('list-canopywave-projects', async (event, apiKey) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // List projects
    const projects = await client.listProjects();
    
    return { success: true, projects: projects };
  } catch (error) {
    logAPIError('list-canopywave-projects', error);
    let errorMessage = error.message || 'Failed to list projects';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
    }
    return { success: false, error: errorMessage };
  }
});

// Get account balance
ipcMain.handle('get-canopywave-balance', async (event, apiKey) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // Get balance
    const balance = await client.getBalance();
    
    return { success: true, balance: balance };
  } catch (error) {
    logAPIError('get-canopywave-balance', error);
    let errorMessage = error.message || 'Failed to get balance';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
    }
    return { success: false, error: errorMessage };
  }
});

// Check flavor/GPU availability
ipcMain.handle('check-canopywave-flavor-availability', async (event, apiKey, project, flavor, region) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!project || !flavor || !region) {
      return { success: false, error: 'Project, flavor, and region are required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // Check availability
    const availability = await client.checkFlavorAvailability(project, flavor, region);
    
    return { success: true, availability: availability };
  } catch (error) {
    logAPIError('check-flavor-availability', error);
    let errorMessage = error.message || 'Failed to check availability';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
    }
    return { success: false, error: errorMessage };
  }
});

// Check GPU availability (simplified for UI)
ipcMain.handle('check-gpu-availability', async (event, apiKey, project, flavor, region) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!project || !flavor || !region) {
      return { success: false, error: 'Project, flavor, and region are required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // Check availability
    console.log('[GPU Availability] Checking with params:', { project, flavor, region });
    const availability = await client.checkFlavorAvailability(project, flavor, region);
    
    // Parse availability with proper precedence (no false positives)
    function parseAvailability(a) {
      if (!a || typeof a !== 'object') {
        console.log('[GPU Availability] Response is not an object:', typeof a, a);
        return null; // unknown
      }

      // Log the raw response for debugging
      console.log('[GPU Availability] Raw API response:', JSON.stringify(a, null, 2));

      // 1) If API provides explicit boolean, trust it (highest priority)
      if (typeof a.available === 'boolean') return a.available;

      // 2) Normalize status strings
      if (typeof a.status === 'string') {
        const s = a.status.trim().toLowerCase();
        if (['available', 'in_stock', 'in-stock', 'ok', 'active'].includes(s)) return true;
        if (['unavailable', 'out_of_stock', 'out-of-stock', 'none', 'capacity_exhausted', 'exhausted'].includes(s)) return false;
      }

      // 3) Check for CanopyWave's dynamic field format: {region}_{flavor}_available_vms
      const keys = Object.keys(a);
      for (const key of keys) {
        if (key.endsWith('_available_vms') && typeof a[key] === 'number') {
          console.log('[GPU Availability] Found available_vms field:', key, '=', a[key]);
          return a[key] > 0;
        }
      }

      // 4) Only use counts if field clearly indicates "available/free"
      const availableCountFields = ['available_count', 'free', 'free_count', 'remaining', 'remaining_count', 'capacity_available'];
      for (const k of availableCountFields) {
        if (typeof a[k] === 'number') return a[k] > 0;
      }

      // 5) Ambiguous "count" field - treat as unknown, not available
      if (typeof a.count === 'number') return null;

      // Log all fields to help debug
      console.log('[GPU Availability] Could not parse. Available fields:', Object.keys(a));

      return null; // unknown
    }
    
    const available = parseAvailability(availability);
    
    console.log('[GPU Availability] Parsed result:', { available, flavor, region });
    
    return { success: true, available, details: availability };
  } catch (error) {
    logAPIError('check-gpu-availability', error);
    let errorMessage = error.message || 'Failed to check GPU availability';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
    }
    return { success: false, error: errorMessage };
  }
});

// List instances
ipcMain.handle('list-canopywave-instances', async (event, apiKey, project, region = null) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!project) {
      return { success: false, error: 'Project is required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // List instances
    const instances = await client.listInstances(project, region);
    
    return { success: true, instances: instances };
  } catch (error) {
    console.error('[Main] Error listing instances:', error);
    return { success: false, error: error.message || 'Failed to list instances' };
  }
});

// Store active cloud training handler
let activeCloudTrainingHandler = null;

// Start cloud training with full workflow
ipcMain.handle('start-cloud-training', async (event, config) => {
  try {
    console.log('[Main] Starting cloud training with config:', config);

    // Extract API key from config
    const apiKey = config.apiKey;
    
    if (!apiKey || typeof apiKey !== 'string' || !apiKey.trim()) {
      return { success: false, error: 'API key is required and must be a string' };
    }

    // Get API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // Create cloud training handler
    activeCloudTrainingHandler = new CloudTrainingHandler(client, mainWindow);

    // Start training (this will handle everything: launch, setup, train, download, cleanup)
    const result = await activeCloudTrainingHandler.startCloudTraining(config);

    activeCloudTrainingHandler = null;
    return { success: true, result: result };

  } catch (error) {
    console.error('[Main] Cloud training error:', error);
    console.error('[Main] Error stack:', error.stack);
    
    // Log full CanopyWave API error details if available
    if (error.apiResponse) {
      console.error('[Main] CanopyWave API Error Details:', JSON.stringify(error.apiResponse, null, 2));
    }
    if (error.statusCode) {
      console.error('[Main] HTTP Status:', error.statusCode);
      console.error('[Main] Endpoint:', error.endpoint || 'unknown');
    }
    if (error.rawResponse) {
      console.error('[Main] Raw API Response:', error.rawResponse);
    }
    if (error.originalError) {
      console.error('[Main] Original Error:', error.originalError);
    }
    
    activeCloudTrainingHandler = null;
    
    // Build detailed error message for renderer
    let errorMessage = error.message || 'Cloud training failed';
    if (error.apiResponse) {
      errorMessage += `\n\nCanopyWave API Error Details:\n${JSON.stringify(error.apiResponse, null, 2)}`;
    }
    if (error.statusCode) {
      errorMessage += `\n\nHTTP Status: ${error.statusCode}`;
      if (error.endpoint) {
        errorMessage += `\nEndpoint: ${error.endpoint}`;
      }
    }
    if (error.rawResponse) {
      errorMessage += `\n\nRaw Response:\n${error.rawResponse.substring(0, 1000)}`;
    }
    
    return { success: false, error: errorMessage };
  }
});

// Stop cloud training
ipcMain.handle('stop-cloud-training', async (event, apiKey, instanceId, project, region) => {
  try {
    console.log('[Main] Stopping cloud training...');
    console.log('[Main] Parameters:', { apiKey: apiKey ? '***' : 'missing', instanceId, project, region });
    
    // First, try to stop via the active training handler
    if (activeCloudTrainingHandler) {
      console.log('[Main] Stopping active training handler...');
      try {
        await activeCloudTrainingHandler.stopTraining();
      } catch (handlerError) {
        console.error('[Main] Error stopping handler:', handlerError);
        // Continue to direct termination even if handler fails
      }
      activeCloudTrainingHandler = null;
    }
    
    // Always terminate the instance directly if parameters provided (fallback)
    if (apiKey && instanceId && project && region) {
      console.log('[Main] Terminating instance directly:', instanceId);
      
      // Get or create API client
      let client = canopywaveClients.get(apiKey.trim());
      if (!client) {
        console.log('[Main] Client not found in cache, creating new one...');
        const CanopyWaveAPI = require('./canopywave-api');
        client = new CanopyWaveAPI(apiKey.trim());
        canopywaveClients.set(apiKey.trim(), client);
      }
      
      try {
        const result = await client.terminateInstance(instanceId, project, region);
        console.log('[Main] Instance termination result:', result);
        console.log('[Main] Instance terminated successfully');
        return { success: true, message: 'Instance terminated successfully' };
      } catch (terminateError) {
        console.error('[Main] Error terminating instance:', terminateError);
        console.error('[Main] Termination error stack:', terminateError.stack);
        
        // Log full API error details
        if (terminateError.apiResponse) {
          console.error('[Main] CanopyWave API Error Details:', JSON.stringify(terminateError.apiResponse, null, 2));
        }
        if (terminateError.statusCode) {
          console.error('[Main] HTTP Status:', terminateError.statusCode);
        }
        if (terminateError.rawResponse) {
          console.error('[Main] Raw API Response:', terminateError.rawResponse);
        }
        
        let errorMessage = terminateError.message || 'Failed to terminate instance';
        if (terminateError.apiResponse) {
          errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(terminateError.apiResponse, null, 2)}`;
        }
        if (terminateError.statusCode) {
          errorMessage += `\n\nHTTP Status: ${terminateError.statusCode}`;
        }
        
        return { success: false, error: errorMessage };
      }
    } else {
      console.warn('[Main] Missing parameters for instance termination:', { apiKey: !!apiKey, instanceId: !!instanceId, project: !!project, region: !!region });
      return { success: false, error: 'Missing required parameters (apiKey, instanceId, project, region)' };
    }
  } catch (error) {
    console.error('[Main] Error stopping cloud training:', error);
    return { success: false, error: error.message || 'Failed to stop cloud training' };
  }
});

// Direct instance termination handler (for terminate button)
ipcMain.handle('terminate-cloud-instance', async (event, apiKey, instanceId, project, region) => {
  try {
    console.log('[Main] Direct instance termination requested...');
    console.log('[Main] Parameters:', { apiKey: apiKey ? '***' : 'missing', instanceId, project, region });
    
    if (!apiKey || !instanceId) {
      return { success: false, error: 'Missing required parameters (apiKey, instanceId)' };
    }
    
    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      console.log('[Main] Client not found in cache, creating new one...');
      const CanopyWaveAPI = require('./canopywave-api');
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }
    
    // Terminate instance (project and region are optional but recommended)
    try {
      const result = await client.terminateInstance(instanceId, project || '', region || '');
      console.log('[Main] Instance terminated successfully');
      return { success: true, message: 'Instance terminated successfully' };
    } catch (terminateError) {
      console.error('[Main] Error terminating instance:', terminateError);
      
      // Log full API error details
      logAPIError('terminate-cloud-instance', terminateError);
      
      let errorMessage = terminateError.message || 'Failed to terminate instance';
      if (terminateError.apiResponse) {
        errorMessage += `\n\nCanopyWave API Error:\n${JSON.stringify(terminateError.apiResponse, null, 2)}`;
      }
      if (terminateError.statusCode) {
        errorMessage += `\n\nHTTP Status: ${terminateError.statusCode}`;
      }
      
      return { success: false, error: errorMessage };
    }
  } catch (error) {
    console.error('[Main] Error in terminate-cloud-instance handler:', error);
    return { success: false, error: error.message };
  }
});

// Handle saving uploaded files to temporary directory
ipcMain.handle('save-uploaded-files', async (event, files) => {
  try {
    const tempDir = path.join(os.tmpdir(), 'unitrainer-uploaded-' + Date.now());
    fs.mkdirSync(tempDir, { recursive: true });
    
    // Save each file
    for (const fileData of files) {
      const filePath = path.join(tempDir, fileData.name);
      // fileData.data is base64 encoded string
      const buffer = Buffer.from(fileData.data, 'base64');
      fs.writeFileSync(filePath, buffer);
    }
    
    return { success: true, directory: tempDir };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Handle starting real training
ipcMain.on('start-real-training', async (event, trainingConfig) => {
  // Store expected total epochs for progress validation
  expectedTotalEpochs = trainingConfig.epochs || null;
  lastParsedEpoch = 0;
  lastParsedTotalEpochs = expectedTotalEpochs || 10;
  console.log('[Main] Expected total epochs:', expectedTotalEpochs);
  console.log('\n' + '='.repeat(80));
  console.log('IPC: start-real-training handler called');
  console.log('Timestamp:', new Date().toISOString());
  console.log('='.repeat(80));
  
  try {
    // Stop existing training if running
    if (trainingProcess) {
      trainingProcess.kill();
      trainingProcess = null;
    }
    
    const trainerScript = getRunnableTrainerPyPath();
    
    // Check if Python script exists
    if (!trainerScript) {
      console.error('!!! ERROR: trainer.py not found !!!');
      mainWindow.webContents.send('training-error', { error: 'trainer.py not found' });
      return;
    }
    
    // Use shared findPythonPath function (defined at module level)
    console.log('Calling ensurePythonAvailable()...');
    let pythonCmd = await ensurePythonAvailable();
    console.log('ensurePythonAvailable() returned:', pythonCmd);
    
    const hasFullPath = pythonCmd && pythonCmd.includes('\\');
    console.log('hasFullPath:', hasFullPath);
    
    // Verify the file actually exists before using it
    if (!pythonCmd) {
      console.error('!!! ERROR: findPythonPath() returned null/undefined !!!');
      mainWindow.webContents.send('training-error', { 
        error: `Python not found. Please install Python 3.8+ from python.org` 
      });
      return;
    }
    
    if (!hasFullPath) {
      console.error('!!! ERROR: Python path does not contain backslash !!!');
      console.error('Python path:', pythonCmd);
      mainWindow.webContents.send('training-error', { 
        error: `Invalid Python path: ${pythonCmd}` 
      });
      return;
    }
    
    const pathExists = isUsablePythonPath(pythonCmd);
    console.log('Python path exists check:', pathExists);
    console.log('Python path being checked:', pythonCmd);
    console.log('Path type:', typeof pythonCmd);
    console.log('Path length:', pythonCmd ? pythonCmd.length : 'N/A');
    
    if (!pathExists) {
      console.error('!!! ERROR: Python path resolved but file does not exist !!!');
      console.error('process.resourcesPath:', process.resourcesPath);
      console.error('process.execPath:', process.execPath);
      console.error('__dirname:', __dirname);
      
      // Try to see what's in the parent directory
      const parentDir = path.dirname(pythonCmd);
      console.error('Parent directory:', parentDir);
      console.error('Parent exists?', fs.existsSync(parentDir));
      if (fs.existsSync(parentDir)) {
        try {
          const files = fs.readdirSync(parentDir);
          console.error('Parent directory contents:', files.slice(0, 10));
        } catch (e) {
          console.error('Cannot read parent directory:', e.message);
        }
      }
      
      mainWindow.webContents.send('training-error', { 
        error: `Python not found at: ${pythonCmd}. Please install Python 3.8+ from python.org` 
      });
      return;
    }
    
    console.log('✓ Python path exists! Proceeding to spawn...');
    
    // Use spawn with shell: false when we have a full path (avoids cmd.exe dependency)
    try {
      // Use the normalized path to ensure proper path handling
      const normalizedPath = path.resolve(pythonCmd);
      const pythonDir = path.dirname(normalizedPath);
      const isWindowsAppsPython = normalizedPath.toLowerCase().includes('\\windowsapps\\');
      
      // For packaged apps, scripts are in app.asar - MUST extract to temp location
      // Python cannot execute scripts directly from asar archives
      const scriptPath = trainerScript;
      
      console.log('Spawning Python process:');
      console.log('  Executable (raw):', pythonCmd);
      console.log('  Executable (normalized):', normalizedPath);
      console.log('  Script:', scriptPath);
      console.log('  Python dir:', pythonDir);
      console.log('  CWD:', path.dirname(scriptPath));
      console.log('  Script exists?', fs.existsSync(scriptPath));
      
      // Use exec instead of spawn/execFile - most reliable on Windows for executables with DLLs
      // exec uses cmd.exe but handles DLL loading better
      const { exec } = require('child_process');
      
      // Set up environment with Python directory in PATH for DLLs
      // Also set PYTHONPATH to ensure bundled site-packages are found
      const pythonLibPath = path.join(pythonDir, 'Lib');
      const sitePackagesPath = path.join(pythonLibPath, 'site-packages');
      
      // Verify site-packages exists
      if (!isWindowsAppsPython && !fs.existsSync(sitePackagesPath)) {
        console.error('ERROR: site-packages path does not exist:', sitePackagesPath);
        mainWindow.webContents.send('training-error', { 
          error: `Python site-packages not found at: ${sitePackagesPath}` 
        });
        return;
      } else if (isWindowsAppsPython && !fs.existsSync(sitePackagesPath)) {
        console.warn('site-packages not found for WindowsApps Python:', sitePackagesPath);
      }
      
      // Verify ultralytics exists
      const ultralyticsPath = path.join(sitePackagesPath, 'ultralytics');
      if (!isWindowsAppsPython && !fs.existsSync(ultralyticsPath)) {
        console.error('ERROR: ultralytics not found at:', ultralyticsPath);
        mainWindow.webContents.send('training-error', { 
          error: `ultralytics package not found in bundled Python. Please rebuild the app.` 
        });
        return;
      } else if (isWindowsAppsPython && !fs.existsSync(ultralyticsPath)) {
        console.warn('ultralytics not found for WindowsApps Python:', ultralyticsPath);
      }
      
      // Set up environment for bundled Python (best practice)
      // Use PYTHONHOME and PYTHONPATH to ensure packages resolve correctly
      const env = {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONHOME: isWindowsAppsPython ? undefined : pythonDir,  // Avoid setting for WindowsApps stubs
        PYTHONPATH: isWindowsAppsPython ? undefined : pythonLibPath,  // Avoid setting for WindowsApps stubs
        PATH: `${pythonDir};${process.env.PATH}`  // Add Python to PATH for DLLs
      };
      
      // If site-packages exists, add it to PYTHONPATH
      if (!isWindowsAppsPython && fs.existsSync(sitePackagesPath)) {
        env.PYTHONPATH = `${sitePackagesPath};${pythonLibPath}`;
      }
      console.log('Environment setup:');
      console.log('  Python dir:', pythonDir);
      console.log('  Site-packages:', sitePackagesPath);
      console.log('  PYTHONPATH:', env.PYTHONPATH);
      console.log('  PYTHONHOME:', env.PYTHONHOME);
      console.log('  ultralytics exists:', fs.existsSync(ultralyticsPath));
      
      // Quote paths to handle spaces
      const execPython = isWindowsAppsPython ? resolvePyLauncherExecutable() : normalizedPath;
      const quotedPython = `"${execPython}"`;
      const quotedScript = `"${scriptPath}"`;
      const command = isWindowsAppsPython
        ? `${quotedPython} -3 ${quotedScript}`
        : `${quotedPython} ${quotedScript}`;
      
      console.log('  Using exec with quoted paths (most reliable on Windows)');
      console.log('  Command:', command);
      
      trainingProcess = exec(command, {
        cwd: path.dirname(scriptPath),
        env: env,
        maxBuffer: 1024 * 1024 * 10 // 10MB buffer
      });
      
      console.log('✓ Spawn call completed successfully');
      console.log('Process PID:', trainingProcess.pid);
      
    } catch (spawnError) {
      console.error('\n!!! SPAWN EXCEPTION CAUGHT !!!');
      console.error('Error type:', spawnError.constructor.name);
      console.error('Error message:', spawnError.message);
      console.error('Error code:', spawnError.code);
      console.error('Error stack:', spawnError.stack);
      console.error('Full error object:', JSON.stringify(spawnError, Object.getOwnPropertyNames(spawnError), 2));
      mainWindow.webContents.send('training-error', { 
        error: `Failed to start Python process: ${spawnError.message}.\n\nPlease install Python 3.8+ from python.org and ensure it's added to your PATH, or install Python to one of the common locations (C:\\Python39, C:\\Program Files\\Python39, etc.)`
      });
      return;
    }
    
    // Send configuration to Python script via stdin
    console.log('Sending training configuration to Python script...');
    console.log('Config:', JSON.stringify(trainingConfig, null, 2));
    const configJson = JSON.stringify(trainingConfig) + '\n';
    trainingProcess.stdin.write(configJson);
    trainingProcess.stdin.end();
    
    // Send initial progress to show training has started
    setTimeout(() => {
      console.log('[Main] Sending initial progress update');
      mainWindow.webContents.send('training-progress', {
        epoch: 0,
        total_epochs: trainingConfig.epochs || 10,
        loss: 1.0,
        accuracy: 0.0,
        progress: 0.0,
        status: "starting"
      });
    }, 1000);
    
    // Handle stdout (training progress and results)
    let buffer = '';
    trainingProcess.stdout.on('data', (data) => {
      buffer += data.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.trim()) {
          try {
            const result = JSON.parse(line);
            if (result.type === 'progress') {
              mainWindow.webContents.send('training-progress', result.data);
            } else if (result.type === 'log') {
              mainWindow.webContents.send('training-log', result.data);
            } else if (result.type === 'result') {
              mainWindow.webContents.send('training-result', result.data);
            } else if (result.type === 'error') {
              mainWindow.webContents.send('training-error', result.data);
            }
          } catch (e) {
            // Not JSON - try to parse YOLO progress output
            const yoloProgress = parseYOLOProgress(line);
            if (yoloProgress) {
              console.log('[YOLO] Sending progress update:', yoloProgress);
              mainWindow.webContents.send('training-progress', yoloProgress);
            } else {
              // Treat as log message (don't try flexible parsing - too risky for false positives)
              if (line.trim()) {
                mainWindow.webContents.send('training-log', { message: line.trim(), level: 'log' });
              }
            }
          }
        }
      }
    });
    
    // Handle stderr - YOLO outputs progress to stderr
    // Use simpler parsing first, then fall back to detailed parsing
    trainingProcess.stderr.on('data', (data) => {
      const errorMsg = data.toString();
      const lines = errorMsg.split('\n');
      
      for (const line of lines) {
        if (line.trim()) {
          // Try simple progress parsing first (more flexible)
          let progress = parseSimpleYOLOProgress(line);
          
          // If simple parsing didn't work, try detailed parsing
          if (!progress) {
            progress = parseYOLOProgress(line);
          }
          
          if (progress) {
            lastProgressTime = Date.now();
            console.log('[YOLO stderr] Parsed progress, sending:', progress);
            mainWindow.webContents.send('training-progress', progress);
          } else {
            // Send as log message for status detection
            mainWindow.webContents.send('training-log', { message: line.trim(), level: 'log' });
          }
        }
      }
    });
    
    // Heartbeat check - detect if training stalls
    const progressCheckInterval = setInterval(() => {
      if (trainingProcess && Date.now() - lastProgressTime > 60000) { // 60 seconds
        console.warn('[PROGRESS] No progress updates for 60 seconds - training may have stalled');
        // Don't send error, just log - training might be doing heavy computation
      }
    }, 30000); // Check every 30 seconds
    
    // Clear interval when process exits
    trainingProcess.on('exit', () => {
      clearInterval(progressCheckInterval);
    });
    
    // Handle process errors (e.g., Python not found)
    trainingProcess.on('error', (error) => {
      console.error('\n!!! TRAINING PROCESS ERROR EVENT !!!');
      console.error('Error type:', error.constructor.name);
      console.error('Error message:', error.message);
      console.error('Error code:', error.code);
      console.error('Error syscall:', error.syscall);
      console.error('Error path:', error.path);
      console.error('Error stack:', error.stack);
      console.error('Full error object:', JSON.stringify(error, Object.getOwnPropertyNames(error), 2));
      mainWindow.webContents.send('training-error', { 
        error: `Failed to start training: ${error.message}. Make sure Python is installed and accessible.` 
      });
      trainingProcess = null;
    });
    
    // Handle process exit
    trainingProcess.on('exit', (code) => {
      const wasStopRequested = stopRequestedByUser;
      trainingProcess = null;
      // Reset flags immediately
      trainingManuallyStopped = false;
      stopRequestedByUser = false;
      
      // Differentiate between finished and stopped
      if (wasStopRequested) {
        // User clicked Stop button - message already sent in stop handler, don't send again
        // The stop handler already sent 'training-stopped', so we don't need to send it here
        return;
      } else if (code === 0) {
        // Training finished successfully (not user-stopped)
        mainWindow.webContents.send('training-finished', { exitCode: code });
      } else if (code !== null) {
        // Training failed with error (non-zero exit code)
        mainWindow.webContents.send('training-stopped', { manuallyStopped: false, exitCode: code });
        mainWindow.webContents.send('training-error', { error: `Training process exited with code ${code}` });
      }
    });
    
    mainWindow.webContents.send('training-started');
  } catch (error) {
    mainWindow.webContents.send('training-error', { error: error.message });
  }
});

// Handle stopping training
ipcMain.on('stop-real-training', (event) => {
  try {
    if (trainingProcess) {
      stopRequestedByUser = true; // Mark that user requested stop
      trainingManuallyStopped = true; // Also set legacy flag for compatibility
      trainingProcess.kill('SIGTERM');
      // Don't set trainingProcess to null here - let the exit handler do it
      // This ensures we only send the stopped message once
      mainWindow.webContents.send('training-stopped', { manuallyStopped: true });
    }
  } catch (error) {
    stopRequestedByUser = false; // Reset on error
    trainingManuallyStopped = false; // Reset on error
    mainWindow.webContents.send('training-error', { error: error.message });
  }
});

// Handle app quit request from renderer
ipcMain.handle('app-quit', async () => {
  console.log('[Main] App quit requested');
  // Cleanup any training processes
  if (trainingProcess) {
    trainingProcess.kill();
    trainingProcess = null;
  }
  // Quit the app
  app.quit();
  return { success: true };
});

// Cleanup on app quit
app.on('before-quit', () => {
  if (trainingProcess) {
    trainingProcess.kill();
    trainingProcess = null;
  }
});

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
