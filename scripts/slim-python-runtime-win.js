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

const runtimeRoot = process.argv[2];
if (!runtimeRoot) {
  console.error('[slim] Missing runtime path argument.');
  process.exit(1);
}

if (!fs.existsSync(runtimeRoot)) {
  console.error('[slim] Runtime path does not exist:', runtimeRoot);
  process.exit(1);
}

const projectRoot = process.env.PROJECT_ROOT || path.resolve(__dirname, '..');
const optionalWeightsDir = path.join(projectRoot, 'optional-downloads', 'weights');
const weightExtensions = new Set(['.pt', '.onnx', '.bin']);
const weightMinBytes = 50 * 1024 * 1024; // Only treat large files as weights

function getSizeBytes(targetPath) {
  if (!fs.existsSync(targetPath)) {
    return 0;
  }
  const stats = fs.statSync(targetPath);
  if (!stats.isDirectory()) {
    return stats.size;
  }
  let total = 0;
  const entries = fs.readdirSync(targetPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(targetPath, entry.name);
    if (entry.isDirectory()) {
      total += getSizeBytes(fullPath);
    } else {
      total += fs.statSync(fullPath).size;
    }
  }
  return total;
}

function removeDir(targetPath, removed) {
  if (fs.existsSync(targetPath)) {
    fs.rmSync(targetPath, { recursive: true, force: true });
    removed.push(targetPath);
  }
}

function removeFile(targetPath, removed) {
  if (fs.existsSync(targetPath)) {
    fs.rmSync(targetPath, { force: true });
    removed.push(targetPath);
  }
}

function ensureDir(targetPath) {
  if (!fs.existsSync(targetPath)) {
    fs.mkdirSync(targetPath, { recursive: true });
  }
}

const removedItems = [];
const warnings = [];
const movedWeights = [];

const sizeBefore = getSizeBytes(runtimeRoot);
console.log('[slim] Runtime path:', runtimeRoot);
console.log('[slim] Size before:', (sizeBefore / (1024 * 1024)).toFixed(2), 'MB');

// Remove known standard library tests
removeDir(path.join(runtimeRoot, 'Lib', 'test'), removedItems);
removeDir(path.join(runtimeRoot, 'Lib', 'unittest', 'test'), removedItems);

// Remove tkinter/idlelib if present (not used by Uni Trainer)
removeDir(path.join(runtimeRoot, 'Lib', 'tkinter'), removedItems);
removeDir(path.join(runtimeRoot, 'Lib', 'idlelib'), removedItems);

// Remove docs/examples if present
['Doc', 'Docs', 'docs', 'Examples', 'examples'].forEach(name => {
  removeDir(path.join(runtimeRoot, name), removedItems);
});

function walkAndClean(currentPath) {
  const entries = fs.readdirSync(currentPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(currentPath, entry.name);
    if (entry.isDirectory()) {
      const dirName = entry.name.toLowerCase();
      if (dirName === '__pycache__') {
        removeDir(fullPath, removedItems);
        continue;
      }
      if (dirName === '.cache') {
        removeDir(fullPath, removedItems);
        continue;
      }
      if (dirName === 'pip-cache' || dirName === 'pip_cache') {
        removeDir(fullPath, removedItems);
        continue;
      }
      if (dirName === 'cache' && fullPath.toLowerCase().includes(`${path.sep}pip${path.sep}`)) {
        removeDir(fullPath, removedItems);
        continue;
      }
      if (dirName === 'wheels' && fullPath.toLowerCase().includes(`${path.sep}pip${path.sep}`)) {
        removeDir(fullPath, removedItems);
        continue;
      }
      walkAndClean(fullPath);
    } else {
      const ext = path.extname(entry.name).toLowerCase();
      if (ext === '.pyc') {
        removeFile(fullPath, removedItems);
        continue;
      }
      if (weightExtensions.has(ext)) {
        const size = fs.statSync(fullPath).size;
        if (size >= weightMinBytes) {
          ensureDir(optionalWeightsDir);
          const targetPath = path.join(optionalWeightsDir, entry.name);
          fs.renameSync(fullPath, targetPath);
          movedWeights.push({ from: fullPath, to: targetPath, size });
          warnings.push(`[slim] Warning: moved large model file out of runtime: ${entry.name}`);
          continue;
        }
      }
    }
  }
}

walkAndClean(runtimeRoot);

const sizeAfter = getSizeBytes(runtimeRoot);
console.log('[slim] Removed items:', removedItems.length);
removedItems.forEach(item => console.log('[slim] - removed', item));

if (movedWeights.length > 0) {
  console.log('[slim] Moved large model files:', movedWeights.length);
  movedWeights.forEach(item => {
    console.log('[slim] -', item.from, '->', item.to, `(${(item.size / (1024 * 1024)).toFixed(2)} MB)`);
  });
}

warnings.forEach(message => console.log(message));

console.log('[slim] Size after:', (sizeAfter / (1024 * 1024)).toFixed(2), 'MB');
console.log('[slim] Saved:', ((sizeBefore - sizeAfter) / (1024 * 1024)).toFixed(2), 'MB');
