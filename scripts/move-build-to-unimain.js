// Move build from temporary folder to UNIMAIN directory
const fs = require('fs');
const path = require('path');

const projectRoot = path.join(__dirname, '..');
const tempBuildDir = path.join(projectRoot, 'build-output', 'Uni Trainer-win32-x64');
const targetDir = projectRoot; // UNIMAIN directory

if (!fs.existsSync(tempBuildDir)) {
  console.log('Temporary build directory not found:', tempBuildDir);
  process.exit(0); // Not an error, just nothing to move
}

console.log('Moving build files to UNIMAIN directory...');
console.log('From:', tempBuildDir);
console.log('To:', targetDir);

// Copy all files from temp build to target
function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      // Skip source directories
      if (['build-output', 'node_modules', 'extracted_working_app', '.git'].includes(childItemName)) {
        return;
      }
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
  // Copy all files except source directories
  const items = fs.readdirSync(tempBuildDir);
  let copied = 0;
  let skipped = 0;
  
  items.forEach(item => {
    // Skip source directories
    if (['build-output', 'node_modules', 'extracted_working_app', '.git'].includes(item)) {
      return;
    }
    
    const src = path.join(tempBuildDir, item);
    const dest = path.join(targetDir, item);
    
    try {
      const stats = fs.statSync(src);
      
      if (stats.isDirectory()) {
        if (fs.existsSync(dest)) {
          fs.rmSync(dest, { recursive: true, force: true });
        }
        copyRecursiveSync(src, dest);
        copied++;
      } else {
        // Try to copy file, skip if locked
        try {
          fs.copyFileSync(src, dest);
          copied++;
        } catch (fileError) {
          if (fileError.code === 'EBUSY' || fileError.code === 'EPERM') {
            console.log(`⚠ Skipping locked file: ${item}`);
            skipped++;
          } else {
            throw fileError;
          }
        }
      }
    } catch (itemError) {
      if (itemError.code === 'EBUSY' || itemError.code === 'EPERM') {
        console.log(`⚠ Skipping locked item: ${item}`);
        skipped++;
      } else {
        console.warn(`⚠ Warning copying ${item}:`, itemError.message);
        skipped++;
      }
    }
  });
  
  // Try to remove temporary build directory (may fail if files are locked)
  try {
    console.log('Removing temporary build directory...');
    fs.rmSync(path.join(projectRoot, 'build-output'), { recursive: true, force: true });
  } catch (rmError) {
    console.log('⚠ Could not remove temporary directory (some files may be in use)');
  }
  
  console.log(`✓ Build moved to UNIMAIN directory! (${copied} items copied, ${skipped} skipped)`);
} catch (error) {
  console.error('✗ Error moving build:', error.message);
  process.exit(1);
}
