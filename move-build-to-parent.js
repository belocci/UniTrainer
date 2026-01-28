// Move build from temporary folder to parent directory
const fs = require('fs');
const path = require('path');

const tempBuildDir = path.join(__dirname, '..', 'Uni Trainer-win32-x64');
const parentDir = path.join(__dirname, '..');

if (!fs.existsSync(tempBuildDir)) {
  console.log('Temporary build directory not found:', tempBuildDir);
  process.exit(0); // Not an error, just nothing to move
}

console.log('Moving build files to parent directory...');
console.log('From:', tempBuildDir);
console.log('To:', parentDir);

// Copy all files from temp build to parent
function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      // Skip extracted_working_app directory
      if (childItemName === 'extracted_working_app') {
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
  // Copy all files except extracted_working_app
  const items = fs.readdirSync(tempBuildDir);
  let copied = 0;
  let skipped = 0;
  
  items.forEach(item => {
    if (item !== 'extracted_working_app') {
      const src = path.join(tempBuildDir, item);
      const dest = path.join(parentDir, item);
      
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
    }
  });
  
  // Try to remove temporary build directory (may fail if files are locked)
  try {
    console.log('Removing temporary build directory...');
    fs.rmSync(tempBuildDir, { recursive: true, force: true });
  } catch (rmError) {
    console.log('⚠ Could not remove temporary directory (some files may be in use)');
  }
  
  console.log(`✓ Build moved to parent directory! (${copied} items copied, ${skipped} skipped)`);
} catch (error) {
  console.error('✗ Error moving build:', error.message);
  process.exit(1);
}
