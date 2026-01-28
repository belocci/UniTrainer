# Uni Trainer Installer

## Overview

This is a lightweight installer/downloader for Uni Trainer. The installer itself is small (~50-100 MB) and downloads the full application components on first run.

## Structure

```
installer/
├── installer-main.js    # Electron main process for installer
├── installer.html       # Installer UI
├── installer.css        # Installer styles
├── installer.js         # Installer logic
└── package.json         # Installer dependencies
```

## How It Works

1. **Small Initial Download**: User downloads only the installer (~50-100 MB)
2. **Component Download**: Installer downloads:
   - Application files (~100 MB)
   - Python environment (~5.5 GB)
3. **Installation**: Components are extracted and installed locally
4. **Launch**: User can launch the full application

## Implementation Status

⚠️ **This is a basic structure**. To complete implementation:

1. **Host Components**: 
   - Upload application ZIP and Python environment ZIP to a CDN/server
   - Update URLs in `installer.js`

2. **Add Extraction**:
   - Install `adm-zip` package: `npm install adm-zip`
   - Implement zip extraction in `installer-main.js`

3. **Component Verification**:
   - Add checksum verification for downloaded files
   - Verify component integrity before installation

4. **Build Scripts**:
   - Create separate build process for installer
   - Create build process for downloadable components

5. **Update Mechanism**:
   - Check for component updates
   - Allow re-downloading if components are missing/corrupted

## Next Steps

1. Decide on hosting/CDN for components
2. Implement zip extraction
3. Add verification/checksums
4. Test download and installation flow
5. Create build automation
