const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');

const PYTHON_VERSION = '3.11.9'; // Use a stable version
const PYTHON_URL = `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip`;
const projectRoot = path.join(__dirname, '..');
const PYTHON_DIR = path.join(projectRoot, 'python');
const ZIP_FILE = path.join(projectRoot, 'python-embed.zip');

console.log('=== Bundled Python Setup for Uni Trainer ===\n');

// Step 1: Create python directory
console.log('Step 1: Creating python directory...');
if (!fs.existsSync(PYTHON_DIR)) {
  fs.mkdirSync(PYTHON_DIR, { recursive: true });
  console.log(`✓ Created directory: ${PYTHON_DIR}\n`);
} else {
  console.log(`✓ Directory already exists: ${PYTHON_DIR}\n`);
}

// Step 2: Check if Python is already bundled
const pythonExe = path.join(PYTHON_DIR, 'python.exe');
if (fs.existsSync(pythonExe)) {
  console.log('✓ Bundled Python already exists!\n');
  console.log('Python location:', pythonExe);
  
  // Check if packages are installed
  try {
    const result = execSync(`"${pythonExe}" -m pip list`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
    console.log('\nInstalled packages:');
    console.log(result);
  } catch (e) {
    console.log('\nPython found but pip may not be configured. Run setup-pip.bat first.');
  }
  
  process.exit(0);
}

// Step 3: Download Python embeddable package
console.log('Step 2: Python embeddable package needs to be downloaded manually.');
console.log('\nPlease follow these steps:\n');
console.log(`1. Download Python ${PYTHON_VERSION} embeddable package (64-bit):`);
console.log(`   ${PYTHON_URL}`);
console.log('\n2. Save the zip file as: python-embed.zip in this directory');
console.log(`   Or download to: ${ZIP_FILE}\n`);
console.log('3. Run this script again after downloading, or run: node extract-python.js\n');

// Alternative: Try to download if user wants
if (process.argv.includes('--download')) {
  console.log('\nAttempting to download Python embeddable package...\n');
  
  const file = fs.createWriteStream(ZIP_FILE);
  
  https.get(PYTHON_URL, (response) => {
    if (response.statusCode === 302 || response.statusCode === 301) {
      // Follow redirect
      https.get(response.headers.location, (response2) => {
        response2.pipe(file);
        file.on('finish', () => {
          file.close();
          console.log(`✓ Downloaded: ${ZIP_FILE}`);
          console.log('\nNow extracting...\n');
          extractPython();
        });
      });
      return;
    }
    
    response.pipe(file);
    file.on('finish', () => {
      file.close();
      console.log(`✓ Downloaded: ${ZIP_FILE}`);
      console.log('\nNow extracting...\n');
      extractPython();
    });
  }).on('error', (err) => {
    fs.unlinkSync(ZIP_FILE);
    console.error('✗ Download failed:', err.message);
    console.log('\nPlease download manually from:', PYTHON_URL);
  });
} else {
  // Check if zip file already exists
  if (fs.existsSync(ZIP_FILE)) {
    console.log(`✓ Found zip file: ${ZIP_FILE}`);
    console.log('Extracting...\n');
    extractPython();
  }
}

function extractPython() {
  try {
    // Check if we have a zip extraction library
    let AdmZip;
    try {
      AdmZip = require('adm-zip');
    } catch (e) {
      console.log('adm-zip not found. Installing...\n');
      execSync('npm install adm-zip --save-dev', { stdio: 'inherit', cwd: __dirname });
      AdmZip = require('adm-zip');
    }
    
    console.log('Extracting Python embeddable package...');
    const zip = new AdmZip(ZIP_FILE);
    zip.extractAllTo(PYTHON_DIR, true);
    console.log(`✓ Extracted to: ${PYTHON_DIR}\n`);
    
    // Delete zip file
    fs.unlinkSync(ZIP_FILE);
    console.log('✓ Cleaned up zip file\n');
    
    // Setup pip
    console.log('Step 3: Setting up pip...\n');
    setupPip();
    
  } catch (error) {
    console.error('✗ Extraction failed:', error.message);
    console.log('\nAlternative: Extract python-embed.zip manually to the python/ folder');
    console.log('Then run: node setup-pip.js');
  }
}

function setupPip() {
  const pythonDir = PYTHON_DIR;
  const pythonExe = path.join(pythonDir, 'python.exe');
  const pythonPth = path.join(pythonDir, 'python311._pth'); // Adjust version number
  
  if (!fs.existsSync(pythonExe)) {
    console.error('✗ python.exe not found. Extraction may have failed.');
    return;
  }
  
  // Enable site-packages in _pth file
  const pthFiles = fs.readdirSync(pythonDir).filter(f => f.endsWith('._pth'));
  if (pthFiles.length > 0) {
    const pthFile = path.join(pythonDir, pthFiles[0]);
    let pthContent = fs.readFileSync(pthFile, 'utf8');
    
    // Uncomment import site line if it exists
    pthContent = pthContent.replace(/#\s*import site/g, 'import site');
    
    fs.writeFileSync(pthFile, pthContent);
    console.log('✓ Configured python._pth file\n');
  }
  
  // Download get-pip.py
  console.log('Downloading get-pip.py...');
  const getPipUrl = 'https://bootstrap.pypa.io/get-pip.py';
  const getPipFile = path.join(pythonDir, 'get-pip.py');
  
  https.get(getPipUrl, (response) => {
    const file = fs.createWriteStream(getPipFile);
    response.pipe(file);
    file.on('finish', () => {
      file.close();
      console.log('✓ Downloaded get-pip.py\n');
      
      // Install pip
      console.log('Installing pip...');
      try {
        execSync(`"${pythonExe}" "${getPipFile}"`, { 
          stdio: 'inherit',
          cwd: pythonDir,
          env: { ...process.env, PYTHONHOME: pythonDir }
        });
        console.log('\n✓ Pip installed successfully!\n');
        
        // Install required packages
        installPackages();
      } catch (error) {
        console.error('\n✗ Pip installation failed:', error.message);
        console.log('\nTry running manually:');
        console.log(`"${pythonExe}" "${getPipFile}"`);
      }
    });
  }).on('error', (err) => {
    console.error('✗ Failed to download get-pip.py:', err.message);
    console.log('\nDownload manually from: https://bootstrap.pypa.io/get-pip.py');
    console.log(`Save to: ${getPipFile}`);
  });
}

function installPackages() {
  const pythonExe = path.join(PYTHON_DIR, 'python.exe');
  const requirementsFile = path.join(__dirname, 'requirements.txt');
  
  if (!fs.existsSync(requirementsFile)) {
    console.log('⚠ requirements.txt not found. Creating basic requirements...');
    const basicRequirements = `ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
mss>=9.0.0
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.0.0
xgboost>=1.0.0
lightgbm>=3.0.0
pandas>=1.0.0`;
    fs.writeFileSync(requirementsFile, basicRequirements);
    console.log('✓ Created requirements.txt\n');
  }
  
  console.log('Installing required packages (this may take several minutes)...\n');
  console.log('Packages to install: torch, numpy, ultralytics, scikit-learn, xgboost, etc.\n');
  
  try {
    execSync(`"${pythonExe}" -m pip install -r "${requirementsFile}"`, {
      stdio: 'inherit',
      env: { ...process.env, PYTHONHOME: PYTHON_DIR }
    });
    console.log('\n✓ All packages installed successfully!\n');
    console.log('Python is now bundled and ready for distribution!\n');
  } catch (error) {
    console.error('\n✗ Package installation failed:', error.message);
    console.log('\nTry installing packages manually:');
    console.log(`"${pythonExe}" -m pip install torch numpy ultralytics scikit-learn xgboost`);
  }
}

// If zip exists, extract it
if (fs.existsSync(ZIP_FILE) && !process.argv.includes('--download')) {
  extractPython();
}