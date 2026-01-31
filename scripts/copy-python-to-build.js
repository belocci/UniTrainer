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

const fs = require('fs');
const path = require('path');

const projectRoot = path.join(__dirname, '..');
// Copy python folder to build directory
const sourcePython = path.join(projectRoot, 'python');
// Build is in ./build-output/Uni Trainer-win32-x64
const buildDir = path.join(projectRoot, 'build-output', 'Uni Trainer-win32-x64');

if (!fs.existsSync(buildDir)) {
  console.log('Build directory not found:', buildDir);
  process.exit(1);
}

// Copy python to resources folder (same level as app.asar)
const destPython = path.join(buildDir, 'resources', 'python');

console.log('Copying python folder to build...');
console.log('From:', sourcePython);
console.log('To:', destPython);

// Use a simple recursive copy function
function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      copyRecursiveSync(
        path.join(src, childItemName),
        path.join(dest, childItemName)
      );
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}

try {
  if (!fs.existsSync(sourcePython)) {
    console.log('⚠ Python folder not found in source, skipping copy (app will use system Python)');
    process.exit(0);
  }
  
  if (fs.existsSync(destPython)) {
    console.log('Removing existing python folder...');
    fs.rmSync(destPython, { recursive: true, force: true });
  }
  
  copyRecursiveSync(sourcePython, destPython);
  console.log('✓ Python folder copied successfully!');
} catch (error) {
  console.error('✗ Error copying python folder:', error.message);
  console.log('⚠ Continuing build without bundled Python (app will use system Python)');
  process.exit(0); // Don't fail the build
}
