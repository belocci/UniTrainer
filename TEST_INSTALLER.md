# Testing the Installer Locally

## Quick Test Guide

### Step 1: Build the Full Application

First, build the complete application:

```bash
npm run build:win
```

This creates `dist/Uni Trainer-win32-x64/` with all files.

### Step 2: Package Components

Package the application into downloadable ZIP files:

```bash
npm run build:components
```

This will:
- Create `dist-components/uni-trainer-app.zip`
- Create `dist-components/uni-trainer-python.zip`
- Generate checksums in `dist-components/components.json`

**Note**: This step requires the full build to exist first.

### Step 3: Set Up Local Server (for testing downloads)

Since the installer needs to download files via HTTP/HTTPS, you need a local server.

#### Option A: Simple Python Server (Recommended)

Navigate to the components directory and start a server:

```bash
cd dist-components
python -m http.server 8000
```

Or with Python 2:
```bash
python -m SimpleHTTPServer 8000
```

Keep this server running in a separate terminal.

#### Option B: Node.js HTTP Server

Install a simple server:
```bash
npm install -g http-server
```

Then run:
```bash
cd dist-components
http-server -p 8000
```

### Step 4: Configure Installer URLs

Update the installer to use your local server:

```bash
node scripts/setup-installer-config.js http://localhost:8000
```

Or manually edit `installer/installer.js` and change:
```javascript
url: 'http://localhost:8000/uni-trainer-app.zip'
url: 'http://localhost:8000/uni-trainer-python.zip'
```

### Step 5: Test the Installer

Open the installer in development mode:

```bash
cd installer
npm install
npm start
```

Or build and run:

```bash
npm run build:installer
cd dist-installer/"Uni Trainer Installer-win32-x64"
."Uni Trainer Installer.exe"
```

### Step 6: Test Installation Flow

1. **Welcome Screen**: Should show overview
2. **Path Selection**: Click Browse, select a test directory (e.g., `C:\TestInstall`)
3. **Download Progress**: Watch components download
4. **Extraction**: Files should extract automatically
5. **Completion**: Should show "Installation Complete"
6. **Launch**: Click "Launch Uni Trainer" to test

## Quick Test (Without Full Build)

If you just want to test the installer UI without the full 5.6GB download:

1. Create dummy ZIP files:
```bash
cd dist-components
echo "dummy" > dummy.txt
# Create small test ZIPs (you can use any ZIP tool)
```

2. Update installer URLs to point to these dummy files
3. Test the installer flow (it will fail at extraction, but you can verify UI/flow)

## Troubleshooting

- **"Component not found"**: Make sure the local server is running and files are in `dist-components/`
- **"Download failed"**: Check the server is accessible at `http://localhost:8000`
- **"Extraction failed"**: Verify ZIP files are valid (try extracting manually)
- **Port already in use**: Change port in server command and update URLs

## Testing Checklist

- [ ] Installer launches correctly
- [ ] Welcome screen displays
- [ ] Path selection works
- [ ] Downloads start and show progress
- [ ] Components download successfully
- [ ] Files extract correctly
- [ ] Application launches after installation
- [ ] Error messages display correctly (test by stopping server mid-download)
