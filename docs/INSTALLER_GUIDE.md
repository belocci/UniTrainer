# Uni Trainer Installer - Production Guide

## Overview

The installer system allows users to download a small installer (~50-100 MB) that then downloads and installs the full application components (~5.6 GB total). This improves user experience by making the initial download fast.

## Architecture

```
┌─────────────────────┐
│  Installer App      │  ~50-100 MB
│  (Lightweight)      │
└──────────┬──────────┘
           │
           │ Downloads
           ▼
┌─────────────────────┐
│  Component Server   │
│  - app.zip          │  ~100 MB
│  - python.zip       │  ~5.5 GB
└──────────┬──────────┘
           │
           │ Extracts
           ▼
┌─────────────────────┐
│  Local Installation │
│  - Uni Trainer.exe  │
│  - resources/       │
│  - python/          │
└─────────────────────┘
```

## Setup Process

### Step 1: Build Full Application

Build the complete application first:

```bash
npm run build:win
```

This creates `dist/Uni Trainer-win32-x64/` with all files.

### Step 2: Package Components

Package the application into downloadable components:

```bash
npm run build:components
```

This creates:
- `dist-components/uni-trainer-app.zip` (~100 MB)
- `dist-components/uni-trainer-python.zip` (~5.5 GB)
- `dist-components/components.json` (metadata with checksums)

### Step 3: Upload to CDN/Server

Upload the ZIP files to your hosting/CDN:
- Recommended: AWS S3, Google Cloud Storage, Azure Blob Storage, or any CDN
- Make files publicly accessible (or use signed URLs)
- Note the base URL (e.g., `https://cdn.yourdomain.com/uni-trainer`)

### Step 4: Configure Installer URLs

Update the installer with your component URLs:

```bash
node scripts/setup-installer-config.js https://cdn.yourdomain.com/uni-trainer
```

Or manually edit `installer/installer.js`:

```javascript
const components = [
    {
        name: 'Application Files',
        url: 'https://cdn.yourdomain.com/uni-trainer/uni-trainer-app.zip',
        filename: 'app.zip',
        extractTo: 'Uni Trainer',
        size: '100 MB',
        checksum: 'abc123...' // Optional: from components.json
    },
    {
        name: 'Python Environment',
        url: 'https://cdn.yourdomain.com/uni-trainer/uni-trainer-python.zip',
        filename: 'python.zip',
        extractTo: 'Uni Trainer/resources',
        size: '5.5 GB',
        checksum: 'def456...' // Optional: from components.json
    }
];
```

### Step 5: Build Installer

Build the lightweight installer:

```bash
npm run build:installer
```

This creates `dist-installer/Uni Trainer Installer-win32-x64/` with the installer executable.

### Step 6: Distribute Installer

Distribute the installer executable to users. They will:
1. Download and run the installer (~50-100 MB)
2. Select installation directory
3. Installer downloads components
4. Components are extracted automatically
5. Application is ready to launch

## User Experience Flow

1. **Welcome Screen**: User sees overview of what will be installed
2. **Path Selection**: User selects where to install
3. **Download Progress**: Real-time progress for each component
4. **Installation Complete**: User can launch the application

## Features

- ✅ Real-time download progress
- ✅ Component verification (optional checksums)
- ✅ Automatic ZIP extraction
- ✅ Error handling and recovery
- ✅ Clean, modern UI matching Uni Trainer aesthetic
- ✅ Launch application after installation

## Optional: Add Checksums

For security, add SHA256 checksums to verify downloaded files haven't been corrupted:

1. Run `npm run build:components` to generate checksums
2. Check `dist-components/components.json` for checksum values
3. Add checksums to `installer/installer.js` components array

The installer will verify files match the checksums before extraction.

## Troubleshooting

### Download Fails
- Verify URLs are correct and accessible
- Check network connectivity
- Ensure CDN/server is online
- Check file permissions

### Extraction Fails
- Verify ZIP files are not corrupted
- Check available disk space
- Ensure write permissions to installation directory

### Checksum Mismatch
- Re-download the component
- Verify file wasn't corrupted during upload
- Disable checksum verification for testing (remove checksum field)

### Large Installer Size
- The installer should be ~50-100 MB
- If larger, check dependencies
- Consider using Electron Builder's NSIS installer instead

## Production Checklist

- [ ] Components built and tested
- [ ] Components uploaded to reliable CDN/server
- [ ] URLs configured in installer
- [ ] Checksums added (recommended for security)
- [ ] Installer built and tested end-to-end
- [ ] Error handling verified
- [ ] Progress indicators working correctly
- [ ] Installation path validation working
- [ ] Application launches correctly after installation
- [ ] All dependencies included
- [ ] Documentation updated

## Alternative: NSIS Installer

For a more traditional Windows installer experience, you can use Electron Builder's NSIS installer which also supports component downloads. This requires additional configuration in `package.json` build section.
