# 🧪 Test Checklist - Uni Trainer Fixes

## ✅ Both Issues Fixed - Ready to Test

---

## Test 1: Stop Button Fix

**What was broken:**
```
Error: ReferenceError: currentCloudInstanceId is not defined
    at stopTraining (renderer.js:1490)
```

**What was fixed:**
- Switched to `window.currentCloudInstanceId` (bulletproof global)
- Added safety guards for missing `cloudConfig`

**How to test:**
1. ✅ Launch `Uni Trainer.exe` from `dist\Uni Trainer-win32-x64\`
2. ✅ Configure cloud training (enter API key, select project/region/GPU)
3. ✅ Start a cloud training job
4. ✅ Click **"Stop Training"** button
5. ✅ **Expected:** Instance terminates cleanly, no errors in console

**Success criteria:**
- No `currentCloudInstanceId is not defined` error
- Instance terminates successfully
- UI updates correctly (stop button becomes start button)

---

## Test 2: Automatic Public IP Association

**What was broken:**
```
[12:28:49 AM] Selected VM has no IP address
[Dashboard] "Associate IP" button required manual action
```

**What was fixed:**
- Implemented CanopyWave's 3-step IP process:
  1. Launch instance
  2. Allocate public IP (or find available one)
  3. Associate IP to instance

**How to test:**
1. ✅ Launch `Uni Trainer.exe`
2. ✅ Configure cloud training
3. ✅ Start cloud training
4. ✅ Watch the logs for:
   ```
   [Timestamp] Allocating public IP address...
   [Timestamp] Associating IP 213.181.xxx.xxx to instance...
   [Timestamp] ✓ Public IP 213.181.xxx.xxx associated successfully
   [Timestamp] Connecting to instance via SSH...
   ```
5. ✅ **Expected:** IP is automatically allocated and associated within 30-60 seconds

**Success criteria:**
- No "Selected VM has no IP address" error
- No manual "Associate IP" action required in CanopyWave dashboard
- SSH connection establishes automatically
- Training starts successfully

---

## Test 3: Full End-to-End Workflow

**Complete cloud training workflow:**

1. ✅ **Launch app**
   ```
   dist\Uni Trainer-win32-x64\Uni Trainer.exe
   ```

2. ✅ **Configure cloud training**
   - Click "Cloud Training" button
   - Enter CanopyWave API key
   - Select project (e.g., "your-project")
   - Select region (e.g., "KEF-2")
   - Select GPU (e.g., "2x H100")
   - Set max hours and budget
   - Click "Start Cloud Training"

3. ✅ **Monitor instance launch**
   - Watch for: "Launching cloud GPU instance..."
   - Watch for: "Instance ID: xxxxx"
   - Watch for: "Allocating public IP address..."
   - Watch for: "✓ Public IP xxx.xxx.xxx.xxx associated successfully"

4. ✅ **Monitor SSH connection**
   - Watch for: "Connecting to instance via SSH..."
   - Watch for: "Setting up training environment..."
   - Watch for: "Uploading dataset and training scripts..."

5. ✅ **Monitor training**
   - Watch for: "Starting training..."
   - Watch for training progress logs
   - GPU utilization should show in logs

6. ✅ **Test stop button**
   - Click "Stop Training" button
   - Watch for: "Stopping cloud training and terminating instance..."
   - Watch for: "Cloud instance terminated successfully"
   - No errors should appear

**Success criteria:**
- All steps complete without errors
- No manual intervention required
- Stop button works correctly
- UI remains responsive throughout

---

## 🐛 If You Encounter Issues

### Issue: Stop button still shows error
**Check:**
- Is the asar file 15.3 MB? (Should be, not 4.7 GB)
- Run: `dir "dist\Uni Trainer-win32-x64\resources\app.asar"`
- If wrong size, rebuild: `npm run build:win`

### Issue: IP association fails
**Check:**
- Do you have available public IPs in your CanopyWave project?
- Go to: https://cloud.canopywave.io → Public IPs
- If all IPs are in use, the app will allocate a new one (may cost extra)
- Check CanopyWave account balance and quotas

### Issue: "Project network not found"
**Check:**
- Does your CanopyWave project have a network configured?
- Go to: https://cloud.canopywave.io → Networks
- If no network exists, create one or contact CanopyWave support

### Issue: API key validation fails
**Check:**
- Is the API key correct? (No extra spaces/newlines)
- Does the API key have permission to launch instances?
- Try regenerating the API key in CanopyWave dashboard

---

## 📊 Expected Log Output (Success)

```
[01:30:00 AM] Starting CLOUD training...
[01:30:00 AM] Training will be performed on CanopyWave cloud GPU
[01:30:00 AM] Preparing cloud training...
[01:30:00 AM] GPU: 2x H100 (80 GB SXM5) - $2.25 / gpu / hour
[01:30:00 AM] Region: KEF-2
[01:30:00 AM] Max training time: 1 hours
[01:30:00 AM] Budget limit: $5
[01:30:00 AM] Starting cloud training job...
[01:30:05 AM] Launching cloud GPU instance...
[01:30:15 AM] Instance ID: 9bc56d2b-411b-47c3-b93a-691c3931009c
[01:30:15 AM] Allocating public IP address...
[01:30:18 AM] Found available public IP: 213.181.122.171
[01:30:18 AM] Associating IP 213.181.122.171 to instance...
[01:30:22 AM] ✓ Public IP 213.181.122.171 associated successfully
[01:30:22 AM] Waiting for instance to be ready...
[01:30:35 AM] Instance is ACTIVE with IP: 213.181.122.171
[01:30:35 AM] Connecting to instance via SSH...
[01:30:40 AM] SSH connection established
[01:30:40 AM] Setting up training environment...
[01:30:50 AM] Installing dependencies...
[01:31:20 AM] Uploading dataset...
[01:32:00 AM] Starting training...
[01:32:05 AM] Epoch 1/100: Loss: 0.5432
...
```

---

## ✅ Sign-Off Checklist

After testing, confirm:

- [ ] Stop button works without errors
- [ ] Public IP is automatically allocated
- [ ] Public IP is automatically associated
- [ ] SSH connection establishes automatically
- [ ] Training starts successfully
- [ ] Stop button terminates instance cleanly
- [ ] No manual CanopyWave dashboard actions required
- [ ] UI remains responsive throughout

---

**If all checkboxes are ticked, both issues are completely resolved!** 🎉

---

## 📁 Build Location

```
C:\Users\vaugh\transfer package - uni trainer\dist\Uni Trainer-win32-x64\Uni Trainer.exe
```

**Build info:**
- Date: 2026-01-18 01:30 AM
- Asar size: 15.3 MB
- Fixes: Stop button + Automatic IP association
- Status: Ready for production ✅
