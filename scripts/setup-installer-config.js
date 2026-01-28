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
 * Setup script to configure installer with component URLs
 * Run this after uploading components to your CDN/server
 * 
 * Usage: node scripts/setup-installer-config.js <base-url>
 * Example: node scripts/setup-installer-config.js https://cdn.yourdomain.com/uni-trainer
 */

const fs = require('fs');
const path = require('path');

const baseUrl = process.argv[2];
const installerJsPath = path.join(__dirname, '..', 'installer', 'installer.js');

if (!baseUrl) {
    console.error('Usage: node setup-installer-config.js <base-url>');
    console.error('Example: node setup-installer-config.js https://cdn.yourdomain.com/uni-trainer');
    process.exit(1);
}

// Read installer.js
let installerJs = fs.readFileSync(installerJsPath, 'utf8');

// Update URLs
installerJs = installerJs.replace(
    /url: 'https:\/\/your-cdn\.com\/uni-trainer-app\.zip'/g,
    `url: '${baseUrl}/uni-trainer-app.zip'`
);
installerJs = installerJs.replace(
    /url: 'https:\/\/your-cdn\.com\/uni-trainer-python\.zip'/g,
    `url: '${baseUrl}/uni-trainer-python.zip'`
);

// Write back
fs.writeFileSync(installerJsPath, installerJs, 'utf8');

console.log(`✓ Installer configured with base URL: ${baseUrl}`);
console.log('\nComponent URLs:');
console.log(`  App: ${baseUrl}/uni-trainer-app.zip`);
console.log(`  Python: ${baseUrl}/uni-trainer-python.zip`);
