# Setting Up Bundled Python for Distribution

This guide will help you bundle Python with your Uni Trainer application for distribution.

## Quick Setup (Automated)

Run the setup script:
```bash
node setup-bundled-python.js
```

Or with automatic download:
```bash
node setup-bundled-python.js --download
```

## Manual Setup Steps

### Step 1: Download Python Embeddable Package

1. Go to https://www.python.org/downloads/windows/
2. Scroll down to "Windows embeddable package (64-bit)"
3. Download Python 3.11.x (recommended version)
4. Save the zip file (e.g., `python-3.11.9-embed-amd64.zip`) to your project directory

### Step 2: Extract Python

1. Create a `python` folder in your project root:
   ```
   your-project/
   └── python/
   ```

2. Extract ALL contents of the zip file into the `python/` folder

3. You should have `python.exe` in the `python/` folder

### Step 3: Enable pip (Enable site-packages)

1. In the `python/` folder, find the file `python311._pth` (or similar, with your Python version)
2. Open it in a text editor
3. Uncomment (remove the `#`) from this line:
   ```
   import site
   ```
   Or add it if it's not there

4. Save the file

### Step 4: Install pip

1. Download `get-pip.py` from: https://bootstrap.pypa.io/get-pip.py
2. Save it to the `python/` folder
3. Run:
   ```bash
   python\python.exe python\get-pip.py
   ```

### Step 5: Install Required Packages

Run:
```bash
python\python.exe -m pip install -r requirements.txt
```

Or install individually:
```bash
python\python.exe -m pip install torch torchvision numpy ultralytics scikit-learn xgboost lightgbm pandas Pillow mss
```

This will take several minutes as it downloads and installs packages (especially PyTorch which is large ~2GB).

### Step 6: Verify Installation

Test that Python works:
```bash
python\python.exe --version
python\python.exe -m pip list
```

## Project Structure

After setup, your project should look like:
```
your-project/
├── python/
│   ├── python.exe
│   ├── python311.dll
│   ├── Lib/
│   │   └── site-packages/  (your installed packages)
│   ├── Scripts/
│   └── ...
├── trainer.py
├── detector.py
├── requirements.txt
├── package.json
└── ...
```

## Build Configuration

The `package.json` has been updated to include the `python/` folder in the build.

When you build with `npm run build:win`, the python folder will be included in the distribution.

## Testing

1. Run your app locally to verify Python is detected
2. Check the console logs - you should see: "Found bundled Python at: ..."
3. Try starting a training to verify packages are installed

## Distribution Size

- Python embeddable: ~25 MB
- PyTorch: ~1.5-2 GB
- Other packages: ~500 MB
- **Total: ~2-3 GB**

This is normal for ML applications with PyTorch.

## Troubleshooting

### Python not found
- Check that `python/python.exe` exists
- Verify the path in console logs
- Make sure you extracted the zip correctly

### Packages not found
- Verify `python/Lib/site-packages/` contains your packages
- Check that `python311._pth` has `import site` uncommented
- Reinstall packages if needed

### Build doesn't include Python
- Check `package.json` build configuration
- Verify `python/` folder exists before building
- Check build output directory

## Notes

- The bundled Python is for distribution only
- During development, you can use system Python
- The code checks bundled Python first, then falls back to system Python
- This ensures the app works both in development and distribution