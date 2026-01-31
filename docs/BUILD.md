# Building Uni Trainer for Distribution

## Prerequisites

1. **Node.js** (v16 or higher)
2. **npm** (comes with Node.js)

## Building for Windows

```bash
npm run build:win
```

This creates a folder: **`dist/Uni Trainer-win32-x64/`**

**To distribute:**
1. Zip the entire `Uni Trainer-win32-x64` folder
2. Share the ZIP file
3. Users extract and run `Uni Trainer.exe`

## Building for macOS

**Note**: macOS builds must be done on a Mac computer.

```bash
npm run build:mac
```

This creates folders:
- `dist/Uni Trainer-darwin-x64/` (Intel Mac)
- `dist/Uni Trainer-darwin-arm64/` (Apple Silicon/M1/M2)

**To distribute:**
1. Zip the appropriate folder(s)
2. Share the ZIP file(s)
3. Users extract and run `Uni Trainer.app`

**Mac Security:** Users may need to right-click → Open (first time), or allow in System Preferences.

## Distribution Process

### For Windows:
1. Run `npm run build:win`
2. Zip the `dist/Uni Trainer-win32-x64` folder
3. Share the ZIP file

### For Mac:
1. Run `npm run build:mac` (on a Mac)
2. Zip the appropriate `dist/Uni Trainer-darwin-*` folder(s)
3. Share the ZIP file(s)

## File Sizes
- Windows: ~150-200 MB
- Mac: ~150-200 MB

## User Requirements
- **Windows:** Windows 10 or higher
- **Mac:** macOS 10.13 or higher
- No additional software needed - everything is bundled!
