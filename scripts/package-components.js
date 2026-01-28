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
const AdmZip = require('adm-zip');
const archiver = require('archiver');
const crypto = require('crypto');

const componentsDir = path.join(__dirname, '..', 'dist-components');
const buildDir = path.join(__dirname, '..', 'dist', 'Uni Trainer-win32-x64');

function calculateChecksum(filePath) {
    // Use streaming for large files to avoid loading entire file into memory
    const hash = crypto.createHash('sha256');
    const fileStream = fs.createReadStream(filePath);
    
    return new Promise((resolve, reject) => {
        fileStream.on('data', (chunk) => {
            hash.update(chunk);
        });
        fileStream.on('end', () => {
            resolve(hash.digest('hex'));
        });
        fileStream.on('error', reject);
    });
}

async function packageComponent(sourceDir, outputFile, componentName, excludeDirs = []) {
    console.log(`\nPackaging ${componentName}...`);
    
    if (!fs.existsSync(sourceDir)) {
        throw new Error(`Source directory not found: ${sourceDir}`);
    }

    // Create output directory
    if (!fs.existsSync(componentsDir)) {
        fs.mkdirSync(componentsDir, { recursive: true });
    }

    const outputPath = path.join(componentsDir, outputFile);
    
    // Use archiver for large files (>2GB), AdmZip for smaller files
    // Check if we should use archiver (estimate: if directory is >1GB, use archiver)
    const useArchiver = componentName.includes('Python') || componentName.includes('python');
    
    if (useArchiver) {
        return await packageWithArchiver(sourceDir, outputPath, outputFile, componentName, excludeDirs);
    } else {
        return await packageWithAdmZip(sourceDir, outputPath, outputFile, componentName, excludeDirs);
    }
}

async function packageWithArchiver(sourceDir, outputPath, outputFile, componentName) {
    return new Promise((resolve, reject) => {
        const output = fs.createWriteStream(outputPath);
        const archive = archiver('zip', {
            zlib: { level: 1 } // Fast compression for large files
        });

        output.on('close', async () => {
            const fileSize = (archive.pointer() / (1024 * 1024)).toFixed(2);
            console.log(`✓ ${componentName} packaged: ${outputFile} (${fileSize} MB)`);
            
            // Calculate checksum (this might take a while for large files)
            console.log(`  Calculating checksum...`);
            try {
                const checksum = await calculateChecksum(outputPath);
                console.log(`  SHA256: ${checksum}`);
                resolve({ file: outputFile, size: fileSize, checksum });
            } catch (err) {
                console.log(`  Warning: Could not calculate checksum: ${err.message}`);
                resolve({ file: outputFile, size: fileSize, checksum: null });
            }
        });

        archive.on('error', (err) => {
            reject(err);
        });

        archive.pipe(output);
        archive.directory(sourceDir, false);
        archive.finalize();
    });
}

async function packageWithAdmZip(sourceDir, outputPath, outputFile, componentName) {
    const zip = new AdmZip();
    
    // Add all files from source directory
    const files = fs.readdirSync(sourceDir, { withFileTypes: true });
    
    for (const file of files) {
        const sourcePath = path.join(sourceDir, file.name);
        const relativePath = file.name;
        
        if (file.isDirectory()) {
            zip.addLocalFolder(sourcePath, relativePath);
        } else {
            zip.addLocalFile(sourcePath, relativePath);
        }
    }

    zip.writeZip(outputPath);
    
    const fileSize = (fs.statSync(outputPath).size / (1024 * 1024)).toFixed(2);
    
    // Calculate checksum (use sync version for small files)
    let checksum;
    try {
        const fileBuffer = fs.readFileSync(outputPath);
        const hash = crypto.createHash('sha256');
        hash.update(fileBuffer);
        checksum = hash.digest('hex');
        console.log(`✓ ${componentName} packaged: ${outputFile} (${fileSize} MB)`);
        console.log(`  SHA256: ${checksum}`);
    } catch (err) {
        console.log(`✓ ${componentName} packaged: ${outputFile} (${fileSize} MB)`);
        console.log(`  Warning: Could not calculate checksum: ${err.message}`);
        checksum = null;
    }
    
    return { file: outputFile, size: fileSize, checksum };
}

async function main() {
    console.log('Packaging Uni Trainer components for distribution...\n');
    
    // Package application files (excluding python)
    const appOutputPath = path.join(buildDir, 'resources');
    const appFiles = [
        'Uni Trainer.exe',
        'resources/app.asar',
        'locales',
        'LICENSES.chromium.html',
        'version'
    ].filter(file => {
        const fullPath = path.join(buildDir, file);
        return fs.existsSync(fullPath);
    });

    // Create temp directory for app files
    const tempAppDir = path.join(componentsDir, 'temp-app');
    if (fs.existsSync(tempAppDir)) {
        fs.rmSync(tempAppDir, { recursive: true, force: true });
    }
    fs.mkdirSync(tempAppDir, { recursive: true });

    // Copy app files
    for (const file of appFiles) {
        const source = path.join(buildDir, file);
        const dest = path.join(tempAppDir, file);
        const destDir = path.dirname(dest);
        
        if (!fs.existsSync(destDir)) {
            fs.mkdirSync(destDir, { recursive: true });
        }
        
        if (fs.statSync(source).isDirectory()) {
            copyRecursiveSync(source, dest);
        } else {
            fs.copyFileSync(source, dest);
        }
    }

    const appResult = await packageComponent(tempAppDir, 'uni-trainer-app.zip', 'Application Files');
    
    // Package Python environment
    const pythonDir = path.join(buildDir, 'resources', 'python');
    const pythonResult = await packageComponent(pythonDir, 'uni-trainer-python.zip', 'Python Environment');

    // Create components manifest
    const manifest = {
        version: '1.0.0',
        components: [
            {
                name: appResult.file,
                size: appResult.size + ' MB',
                checksum: appResult.checksum,
                extractTo: 'Uni Trainer'
            },
            {
                name: pythonResult.file,
                size: pythonResult.size + ' MB',
                checksum: pythonResult.checksum,
                extractTo: 'Uni Trainer/resources'
            }
        ],
        timestamp: new Date().toISOString()
    };

    fs.writeFileSync(
        path.join(componentsDir, 'components.json'),
        JSON.stringify(manifest, null, 2)
    );

    console.log('\n✓ Components packaged successfully!');
    console.log(`\nComponents directory: ${componentsDir}`);
    console.log('\nNext steps:');
    console.log('1. Upload components to your CDN/server');
    console.log('2. Update URLs in installer/installer.js');
    console.log('3. Update checksums in installer/installer.js (optional but recommended)');

    // Cleanup temp directory
    if (fs.existsSync(tempAppDir)) {
        fs.rmSync(tempAppDir, { recursive: true, force: true });
    }
}

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

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
