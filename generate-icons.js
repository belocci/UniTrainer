// Generate icon files from logo description
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const toIco = require('to-ico');

const buildDir = path.join(__dirname, 'build');
if (!fs.existsSync(buildDir)) {
  fs.mkdirSync(buildDir, { recursive: true });
}

// Create SVG logo (white triangle/A shape on black background with star)
const svgLogo = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="512" height="512" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <!-- Black background -->
  <rect width="512" height="512" fill="#000000"/>
  
  <!-- White triangle/A shape - left and right bars meeting at top -->
  <path d="M 140 380 L 256 120 L 372 380 Z" fill="none" stroke="#FFFFFF" stroke-width="28" stroke-linecap="round" stroke-linejoin="round"/>
  
  <!-- Horizontal crossbar (with gaps on both sides) -->
  <rect x="180" y="305" width="70" height="22" fill="#FFFFFF" rx="4"/>
  <rect x="262" y="305" width="70" height="22" fill="#FFFFFF" rx="4"/>
  
  <!-- Small star icon in bottom right corner -->
  <g transform="translate(430, 440) scale(2)">
    <path d="M 0 -8 L 2.5 -2 L 8 -2 L 3.5 1.5 L 5 7 L 0 4 L -5 7 L -3.5 1.5 L -8 -2 L -2.5 -2 Z" 
          fill="#CCCCCC" opacity="0.9"/>
  </g>
</svg>`;

async function generateIcons() {
  try {
    console.log('Generating icon files...');
    
    // Write SVG
    const svgPath = path.join(buildDir, 'logo.svg');
    fs.writeFileSync(svgPath, svgLogo);
    console.log('✓ SVG created');
    
    // Generate PNG from SVG (multiple sizes for ICO)
    const sizes = [16, 32, 48, 64, 128, 256];
    const pngBuffers = [];
    
    for (const size of sizes) {
      const pngBuffer = await sharp(Buffer.from(svgLogo))
        .resize(size, size)
        .png()
        .toBuffer();
      pngBuffers.push(pngBuffer);
    }
    
    // Save main PNG (256x256)
    const mainPngPath = path.join(buildDir, 'logo.png');
    fs.writeFileSync(mainPngPath, pngBuffers[pngBuffers.length - 1]);
    console.log('✓ PNG created (256x256)');
    
    // Create ICO file (contains multiple sizes)
    const icoBuffer = await toIco(pngBuffers);
    const icoPath = path.join(buildDir, 'icon.ico');
    fs.writeFileSync(icoPath, icoBuffer);
    console.log('✓ ICO created (multi-size)');
    
    // For Mac ICNS, we'll create a placeholder note
    // ICNS requires special format, so we'll provide instructions
    const icnsNote = `ICNS File Creation:

For macOS builds, you'll need to convert logo.png to icon.icns.
You can use one of these methods:

1. Online: https://cloudconvert.com/png-to-icns
   - Upload build/logo.png
   - Download and save as build/icon.icns

2. On Mac: Use iconutil command:
   iconutil -c icns build/icon.iconset

3. Or use an app like Icon Composer

For now, Windows builds will work with icon.ico`;
    
    fs.writeFileSync(path.join(buildDir, 'ICNS_INSTRUCTIONS.txt'), icnsNote);
    console.log('✓ Icon generation complete!');
    console.log('\nWindows icon ready: build/icon.ico');
    console.log('See build/ICNS_INSTRUCTIONS.txt for Mac icon setup');
    
  } catch (error) {
    console.error('Error generating icons:', error);
    process.exit(1);
  }
}

generateIcons();
