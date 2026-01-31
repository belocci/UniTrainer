# Installer Setup Guide

## Overview

The Uni Trainer installer is a lightweight downloader that fetches and installs the full application components. This makes the initial download small (~50-100 MB) while the full app is ~5.6 GB.

## Quick Start

### 1. Build Components

First, build the full application and package components:

```bash
npm run build:components
```

This will:
- Build the full application
- Create ZIP files for app and Python components
- Generate checksums for verification
- Output files to `dist-components/`

### 2. Upload Components to CDN/Server

Upload the following files from `dist-components/` to your hosting/CDN:
- `uni-trainer-app.zip` (~100 MB)
- `uni-trainer-python.zip` (~5.5 GB)

### 3. Configure Installer URLs

Update the installer with your component URLs:

```bash
node scripts/setup-installer-config.js https://your-cdn.com/uni-trainer
```

Or manually edit `installer/installer.js` and update the URLs in the `components` array.

### 4. Build Installer

```bash
npm run build:installer
```

The installer will be built to `dist-installer/Uni Trainer Installer-win32-x64/`

### 5. Distribute

Distribute the installer executable. Users will:
1. Download and run the installer (~50-100 MB)
2. Select installation directory
3. Installer downloads components
4. Components are extracted
5. Application is ready to use

## Component Structure

### Application Component (`uni-trainer-app.zip`)
Contains:
- `Uni Trainer.exe`
- `resources/app.asar`
- `locales/`
- Other Electron runtime files

### Python Component (`uni-trainer-python.zip`)
Contains:
- `resources/python/` - Full Python environment with ML libraries

## Optional: Add Checksums

For security, add SHA256 checksums to `installer/installer.js`:

```javascript
{
    name: 'Application Files',
    url: 'https://your-cdn.com/uni-trainer-app.zip',
    filename: 'app.zip',
    extractTo: 'Uni Trainer',
    size: '100 MB',
    checksum: 'abc123...' // SHA256 checksum
}
```

Checksums are printed when running `npm run build:components`.

## Troubleshooting

- **Download fails**: Check URLs are accessible and correct
- **Extraction fails**: Verify ZIP files are not corrupted
- **Checksum mismatch**: Re-download components or disable checksum verification
- **Large download size**: Consider using a CDN with compression

## Production Checklist

- [ ] Components uploaded to reliable CDN/server
- [ ] URLs configured in installer
- [ ] Checksums added (recommended)
- [ ] Installer tested end-to-end
- [ ] Error handling verified
- [ ] Progress indicators working
- [ ] Installation path validated
