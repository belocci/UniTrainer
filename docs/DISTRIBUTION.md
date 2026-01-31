# Distribution Guide for Uni Trainer

## Building for Distribution

### Windows Build

```bash
npm run build:win
```

This creates a folder: `dist/Uni Trainer-win32-x64/`

**To distribute:**
1. Zip the entire `Uni Trainer-win32-x64` folder
2. Share the ZIP file with Windows users
3. Users can extract and run `Uni Trainer.exe` from the folder

### Mac Build

**Note:** Mac builds must be done on a Mac computer.

```bash
npm run build:mac
```

This creates folders:
- `dist/Uni Trainer-darwin-x64/` (Intel Mac)
- `dist/Uni Trainer-darwin-arm64/` (Apple Silicon/M1/M2)

**To distribute:**
1. Zip the appropriate folder
2. Share the ZIP file with Mac users
3. Users extract and run `Uni Trainer.app`

**Mac Security Note:**
- Users may need to right-click → Open → "Open" (first time only)
- Or: System Preferences → Security & Privacy → Allow

### Distribution Files

**For Windows:**
- Share: `Uni Trainer-win32-x64.zip` (zipped folder)
- Users extract and run `Uni Trainer.exe`

**For Mac (Intel):**
- Share: `Uni Trainer-darwin-x64.zip` (zipped folder)
- Users extract and run `Uni Trainer.app`

**For Mac (Apple Silicon):**
- Share: `Uni Trainer-darwin-arm64.zip` (zipped folder)
- Users extract and run `Uni Trainer.app`

## Creating ZIP Files

**Windows:**
- Right-click the `Uni Trainer-win32-x64` folder
- Send to → Compressed (zipped) folder

**Mac:**
- Right-click the `Uni Trainer-darwin-x64` or `Uni Trainer-darwin-arm64` folder
- Compress

## File Sizes

- Windows: ~150-200 MB
- Mac: ~150-200 MB

## Requirements

**Windows:**
- Windows 10 or higher
- No additional software needed

**Mac:**
- macOS 10.13 or higher
- No additional software needed
