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

const projectRoot = path.resolve(__dirname, '..');
const releasesRoot = path.join(projectRoot, 'releases');

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

function findReleaseFolder() {
  if (!fs.existsSync(releasesRoot)) {
    return null;
  }
  const candidates = fs.readdirSync(releasesRoot, { withFileTypes: true })
    .filter(entry => entry.isDirectory())
    .map(entry => path.join(releasesRoot, entry.name));

  const valid = candidates.filter(folder => {
    const resourcesPython = path.join(folder, 'resources', 'python');
    return fs.existsSync(resourcesPython);
  });

  if (valid.length === 0) {
    return null;
  }

  valid.sort((a, b) => {
    const aStat = fs.statSync(a);
    const bStat = fs.statSync(b);
    return bStat.mtimeMs - aStat.mtimeMs;
  });

  return valid[0];
}

function listLargestFiles(rootPath, limit = 15) {
  const files = [];
  function walk(currentPath) {
    const entries = fs.readdirSync(currentPath, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentPath, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else {
        const size = fs.statSync(fullPath).size;
        files.push({ path: fullPath, size });
      }
    }
  }
  walk(rootPath);
  files.sort((a, b) => b.size - a.size);
  return files.slice(0, limit);
}

const releaseFolder = findReleaseFolder();
if (!releaseFolder) {
  console.error('[size] No release folder with resources/python found under releases/.');
  process.exit(1);
}

const pythonPath = path.join(releaseFolder, 'resources', 'python');
const releaseSize = getSizeBytes(releaseFolder);
const pythonSize = getSizeBytes(pythonPath);

console.log('[size] Release folder:', releaseFolder);
console.log('[size] Packaged folder size:', (releaseSize / (1024 * 1024)).toFixed(2), 'MB');
console.log('[size] resources/python size:', (pythonSize / (1024 * 1024)).toFixed(2), 'MB');

const largest = listLargestFiles(pythonPath, 15);
console.log('[size] Top 15 largest files in resources/python:');
largest.forEach(item => {
  console.log('[size] -', (item.size / (1024 * 1024)).toFixed(2), 'MB', item.path);
});
