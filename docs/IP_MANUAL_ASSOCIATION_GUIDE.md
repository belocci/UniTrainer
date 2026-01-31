# IP Association - Manual Workaround Guide

## Date
January 18, 2026

---

## 🎯 What Changed

Since automatic IP association isn't working reliably, the app now:
1. ✅ **Tries automatic IP association** (may or may not work)
2. ✅ **Waits up to 15 minutes** for IP to appear
3. ✅ **Shows clear instructions** after 30 seconds if no IP
4. ✅ **Continues automatically** once you manually associate the IP

---

## 📋 How to Use (Step-by-Step)

### Step 1: Start Training in Uni Trainer

1. Open Uni Trainer
2. Configure cloud training settings
3. Click "Start Training"
4. Watch the console logs

### Step 2: Wait for Instructions

After about 30 seconds, if no IP is auto-assigned, you'll see:

```
⚠️ Instance is active but no IP detected yet.
📋 Manual step needed:
1. Go to CanopyWave Dashboard (https://cloud.canopywave.io)
2. Find your instance (unitrainer-xxxxxxxx...)
3. Click "Associate a Public IP" or "Public IP" button
4. Select or create a floating IP and associate it
5. Training will continue automatically once IP is assigned
⏳ Continuing to wait for IP...
```

### Step 3: Associate IP in CanopyWave Dashboard

1. **Open CanopyWave Dashboard** in your browser:
   - Go to: https://cloud.canopywave.io
   - Log in to your account

2. **Find Your Instance:**
   - Navigate to "Virtual Machines" or "Instances"
   - Look for instance named `unitrainer-[timestamp]`
   - Status should be "active" or "running"

3. **Associate Public IP:**
   - Click the **"Associate a Public IP"** button (yellow warning icon)
   - OR click the **"Public IP"** button at the top
   - A dialog will appear

4. **Select or Create Floating IP:**
   - If you have available floating IPs: Select one from the list
   - If no floating IPs available: Click "Create New" or "Allocate Floating IP"
   - Click "Associate" or "Confirm"

5. **Wait a moment:**
   - The IP will be assigned (takes 5-10 seconds)
   - You'll see the IP address appear in the instance details

### Step 4: Training Continues Automatically

Once the IP is assigned:
- ✅ Uni Trainer will detect it automatically (checks every 5 seconds)
- ✅ You'll see: "Instance ready at [IP address]"
- ✅ Then: "Waiting for SSH to be ready..."
- ✅ Then: "Connecting to instance via SSH..."
- ✅ Training proceeds normally!

---

## ⏱️ Timeline

```
0:00 - Launch instance
0:05 - Instance becomes ACTIVE
0:30 - Manual instructions appear (if no IP yet)
     ↓
     [You manually associate IP in dashboard]
     ↓
0:35 - IP detected by Uni Trainer
0:40 - SSH connection established
0:45 - Environment setup begins
5:00 - Dataset upload starts
     ↓
     [Training proceeds normally]
```

---

## 💡 Tips

### Tip 1: Keep Both Windows Open
- Uni Trainer on one screen
- CanopyWave Dashboard on another
- Makes it easy to switch and associate IP quickly

### Tip 2: Pre-Create Floating IPs
- Go to CanopyWave Dashboard → Floating IPs
- Create 1-2 floating IPs in advance
- Next time, you can just select from the list (faster)

### Tip 3: Don't Close Uni Trainer
- Even if it's waiting, keep it running
- It will automatically continue once IP is assigned
- No need to restart training

### Tip 4: Check Console Logs
- Open Developer Tools (F12)
- Go to Console tab
- See detailed progress and any errors

---

## 🔍 Troubleshooting

### Issue: "Can't find my instance in dashboard"

**Solution:**
- Check the correct project is selected (top dropdown)
- Check the correct data center/region
- Look for instance name starting with `unitrainer-`
- Check "All Instances" or "Active Instances" filter

### Issue: "No 'Associate Public IP' button"

**Solution:**
- Instance might already have an IP (check instance details)
- Or look for "Public IP" button at the top of the page
- Or go to "Floating IPs" section and associate from there

### Issue: "No floating IPs available"

**Solution:**
- Click "Create Floating IP" or "Allocate IP"
- Select your project and region
- Click "Create"
- Then associate it with your instance

### Issue: "Training still not continuing after IP assigned"

**Solution:**
- Wait 10-20 seconds for detection
- Check Uni Trainer console logs
- Verify IP is actually assigned in dashboard
- Check if SSH port 22 is open in security groups

---

## 🎯 Why Manual Association?

The automatic IP association isn't working because:
1. **API endpoint differences** - CanopyWave's actual API might differ from documentation
2. **Permission requirements** - API key might need additional permissions
3. **Project configuration** - Some projects require manual IP management
4. **Network setup** - Project network might not support auto-assignment

**Manual association is reliable and only takes 30 seconds!**

---

## 📊 What the App Does Now

### Improved Behavior:

1. **Launches instance** ✅
2. **Tries auto-association** (best effort)
3. **Waits patiently** (up to 15 minutes)
4. **Shows clear instructions** (after 30 seconds)
5. **Detects manual association** (checks every 5 seconds)
6. **Continues automatically** (no restart needed)

### Better Logging:

```
[12:30:00] Launching cloud GPU instance...
[12:30:05] Instance ID: i-abc123def456
[12:30:05] Checking for public IP...
[12:30:06] Note: Could not auto-assign IP. You may need to manually associate...
[12:30:06] Waiting for instance to be ready and get IP address...
[12:30:06] ⏳ This may take 1-3 minutes. Please be patient...
[12:30:06] If instance is taking long, check CanopyWave dashboard...
[12:30:15] Instance starting... Status: BUILD (attempt 1)
[12:30:20] Instance starting... Status: BUILD (attempt 2)
[12:30:25] Instance active, waiting for IP address... (attempt 3)
[12:30:30] Instance active, waiting for IP address... (attempt 4)
[12:30:35] ⚠️ Instance is active but no IP detected yet.
[12:30:35] 📋 Manual step needed:
[12:30:35] 1. Go to CanopyWave Dashboard (https://cloud.canopywave.io)
[12:30:35] 2. Find your instance (unitrainer-abc123...)
[12:30:35] 3. Click "Associate a Public IP" button
[12:30:35] 4. Select or create a floating IP and associate it
[12:30:35] 5. Training will continue automatically once IP is assigned
[12:30:35] ⏳ Continuing to wait for IP...
[12:31:00] Instance active, waiting for IP address... (attempt 9)
[12:31:05] Instance ready at 203.0.113.45  ← IP detected!
[12:31:05] Waiting for SSH to be ready...
[12:31:15] Connecting to instance via SSH...
```

---

## 📦 Build Status

**Build Date**: January 18, 2026, 1:02 AM  
**Build Location**: `dist\Uni Trainer-win32-x64\`  
**Status**: ✅ Complete  
**Changes**: ✅ Better guidance and error handling  

---

## ✨ Result

The training workflow now:
- ✅ **Tries automatic IP** (best effort)
- ✅ **Provides clear instructions** (when manual needed)
- ✅ **Waits patiently** (up to 15 minutes)
- ✅ **Continues automatically** (after manual association)
- ✅ **No restart needed** (just associate and wait)

---

## 🚀 Quick Reference

**When you see the manual instructions:**

1. Open: https://cloud.canopywave.io
2. Find: `unitrainer-[timestamp]` instance
3. Click: "Associate a Public IP"
4. Select: Available floating IP (or create new)
5. Wait: 10-20 seconds for detection
6. Done: Training continues automatically!

**Total time: ~30 seconds of manual work**

---

**Status**: ✅ **Improved IP Handling Deployed**

The app now guides you through manual IP association when needed! 🎉
