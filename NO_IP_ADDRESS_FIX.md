# "Selected VM has no IP address" Error - Fix Guide

## Date
January 18, 2026

---

## 🐛 Error Message

```
Selected VM has no IP address
```

**What this means**: The instance launched successfully on CanopyWave and shows up in the dashboard, but it hasn't been assigned an IP address yet, or the IP isn't being retrieved properly.

---

## 🔍 Root Causes

1. **Instance still starting** - IP assignment takes time (30 seconds to 2 minutes)
2. **Network configuration issue** - Instance launched without proper network setup
3. **API response format** - CanopyWave returning IP in unexpected field
4. **Floating IP not assigned** - Some configurations require manual IP assignment

---

## ✅ Solutions

### Solution 1: Wait Longer (Most Common)

The updated app now:
- ✅ Waits up to 10 minutes for IP assignment
- ✅ Polls every 5 seconds
- ✅ Shows progress: "Instance active, waiting for IP address... (attempt X)"
- ✅ Checks multiple IP field formats

**What to do:**
1. Run the updated app (built just now)
2. Start cloud training
3. Watch the status messages
4. Wait for "Instance ready at [IP]" message

---

### Solution 2: Check CanopyWave Dashboard

If the app times out:

1. **Go to CanopyWave Dashboard**
   - Visit: https://cloud.canopywave.io
   - Navigate to your project
   - Find "Instances" or "VMs" section

2. **Check Instance Details**
   - Find your instance (name: `unitrainer-[timestamp]`)
   - Check Status: Should be "ACTIVE" or "Running"
   - Look for IP Address field

3. **If IP is shown in dashboard:**
   - Copy the IP address
   - The app should detect it on next poll
   - If not, there may be an API format issue

4. **If NO IP in dashboard:**
   - Instance may need network configuration
   - Try terminating and relaunching
   - Or contact CanopyWave support

---

### Solution 3: Manual Network Configuration

If instances consistently launch without IPs:

1. **Check Project Network Settings**
   - Go to CanopyWave Dashboard
   - Navigate to your project
   - Check "Networks" section
   - Ensure network has "External Gateway" enabled

2. **Create Floating IP (if needed)**
   - Some CanopyWave configurations require floating IPs
   - Go to "Floating IPs" section
   - Create new floating IP
   - Assign to your instance

3. **Verify Security Groups**
   - Check "Security Groups" section
   - Ensure SSH (port 22) is allowed
   - Add rule if missing: TCP port 22 from 0.0.0.0/0

---

## 🔧 What I Improved

### Enhanced IP Detection

**Before:**
```javascript
const ip = instanceDetails.ip 
    || instanceDetails.floating_ip 
    || instanceDetails.public_ip
    || instanceDetails.accessIPv4;
```
❌ Only checked 4 fields

**After:**
```javascript
const ip = instanceDetails.ip 
    || instanceDetails.floating_ip 
    || instanceDetails.public_ip
    || instanceDetails.accessIPv4
    || instanceDetails.access_ip
    || instanceDetails.ipv4
    || (instanceDetails.addresses && this.extractIPFromAddresses(instanceDetails.addresses))
    || (instanceDetails.networks && this.extractIPFromNetworks(instanceDetails.networks));
```
✅ Checks 8+ different fields and formats

### Better Status Updates

**Before:**
```
Waiting for instance to be ready...
[Long silence...]
```

**After:**
```
Waiting for instance to start and get IP address...
Instance starting... Status: BUILD (attempt 1)
Instance starting... Status: BUILD (attempt 2)
Instance active, waiting for IP address... (attempt 3)
Instance active, waiting for IP address... (attempt 4)
Instance ready at 203.0.113.45
Waiting for SSH to be ready...
```
✅ Clear progress updates every 5 seconds

### Detailed Logging

The app now logs full instance data to console for debugging:
```javascript
console.log('[CloudTraining] Full instance data:', JSON.stringify(instanceDetails, null, 2));
```

**To view logs:**
1. Open Developer Tools (F12 or Ctrl+Shift+I)
2. Go to Console tab
3. Look for `[CloudTraining]` messages
4. See exactly what data CanopyWave is returning

---

## 📝 Changes Made

### File: `cloud-training-handler.js`

**Lines 168-226** - Enhanced `waitForInstanceReady()`:

```javascript
async waitForInstanceReady(instanceId, project, region, maxWaitTime = 600000) {
    const startTime = Date.now();
    const pollInterval = 5000; // Check every 5 seconds
    let attemptCount = 0;

    this.sendStatus('Waiting for instance to start and get IP address...');

    while (Date.now() - startTime < maxWaitTime) {
        try {
            attemptCount++;
            const instanceDetails = await this.client.getInstance(instanceId, project, region);
            
            // Log full instance details for debugging
            console.log('[CloudTraining] Instance check #' + attemptCount);
            console.log('[CloudTraining] Instance status:', instanceDetails.status);
            console.log('[CloudTraining] Full instance data:', JSON.stringify(instanceDetails, null, 2));

            // Check if instance is active and has IP
            if (instanceDetails.status === 'ACTIVE' || instanceDetails.status === 'active') {
                // Try to get IP address from various possible fields
                const ip = instanceDetails.ip 
                    || instanceDetails.floating_ip 
                    || instanceDetails.public_ip
                    || instanceDetails.accessIPv4
                    || instanceDetails.access_ip
                    || instanceDetails.ipv4
                    || (instanceDetails.addresses && this.extractIPFromAddresses(instanceDetails.addresses))
                    || (instanceDetails.networks && this.extractIPFromNetworks(instanceDetails.networks));

                if (ip) {
                    console.log('[CloudTraining] Instance ready with IP:', ip);
                    this.sendStatus(`Instance ready at ${ip}`);
                    
                    // Wait for SSH to be ready
                    this.sendStatus('Waiting for SSH to be ready...');
                    const sshReady = await this.waitForSSHPort(ip, 22, 120000);
                    if (sshReady) {
                        return ip;
                    }
                } else {
                    // Instance is ACTIVE but no IP yet
                    this.sendStatus(`Instance active, waiting for IP address... (attempt ${attemptCount})`);
                }
            } else {
                // Instance not active yet
                this.sendStatus(`Instance starting... Status: ${instanceDetails.status} (attempt ${attemptCount})`);
            }

            await this.sleep(pollInterval);
        } catch (error) {
            console.error('[CloudTraining] Error checking instance status:', error);
            this.sendStatus(`Checking instance status... (attempt ${attemptCount})`);
            await this.sleep(pollInterval);
        }
    }

    throw new Error(`Timeout waiting for instance to be ready. Check CanopyWave dashboard for instance status.`);
}
```

**Lines 228-260** - Added `extractIPFromNetworks()` helper:

```javascript
/**
 * Extract IP from networks object (alternative format)
 */
extractIPFromNetworks(networks) {
    // networks might be an array or object
    if (Array.isArray(networks)) {
        for (const network of networks) {
            if (network.ip || network.addr || network.address) {
                return network.ip || network.addr || network.address;
            }
        }
    } else if (typeof networks === 'object') {
        // Try common field names
        for (const key in networks) {
            const network = networks[key];
            if (typeof network === 'string') {
                // Direct IP string
                return network;
            } else if (network && (network.ip || network.addr || network.address)) {
                return network.ip || network.addr || network.address;
            }
        }
    }
    return null;
}
```

---

## 🧪 Testing

### To Test the Fix:

1. **Run the updated application:**
   ```powershell
   cd "C:\Users\vaugh\transfer package - uni trainer\dist\Uni Trainer-win32-x64"
   .\Uni Trainer.exe
   ```

2. **Open Developer Tools** (F12) to see detailed logs

3. **Start cloud training:**
   - Configure cloud settings
   - Start training
   - Watch status messages in the app
   - Watch console logs for detailed info

4. **Expected behavior:**
   - Status updates every 5 seconds
   - Shows instance status (BUILD → ACTIVE)
   - Shows "waiting for IP address" if needed
   - Eventually shows "Instance ready at [IP]"

---

## 🔍 Debugging

If you still get "no IP address" error:

### Step 1: Check Console Logs
```
[CloudTraining] Instance check #1
[CloudTraining] Instance status: BUILD
[CloudTraining] Full instance data: { ... }
```

Look for the full instance data - see what fields are present.

### Step 2: Check CanopyWave Dashboard
- Does the instance have an IP in the dashboard?
- What is the instance status?
- Is there a network attached?

### Step 3: Share Debug Info
If still stuck, share:
- Console logs (especially the "Full instance data")
- Screenshot of CanopyWave dashboard showing the instance
- Project and region being used

---

## 💡 Common Scenarios

### Scenario 1: Instance Takes Long to Start
**Symptom**: Status shows "BUILD" for several minutes  
**Solution**: Normal - wait up to 5 minutes  
**Status**: ✅ App now waits and shows progress

### Scenario 2: Instance ACTIVE but No IP
**Symptom**: Status is "ACTIVE" but no IP assigned  
**Solution**: Network configuration issue  
**Fix**: Check project network settings, may need floating IP

### Scenario 3: IP in Different Field
**Symptom**: Dashboard shows IP but app doesn't detect it  
**Solution**: API returning IP in unexpected field  
**Fix**: Check console logs, share field name for update

---

## 📦 Build Status

**Build Date**: January 18, 2026, 12:33 AM  
**Build Location**: `dist\Uni Trainer-win32-x64\`  
**Status**: ✅ Complete  
**Changes**: ✅ Enhanced IP detection included  

---

## 🚀 Next Steps

1. **Try the updated app** - Should now detect IPs better
2. **Watch the status messages** - See what's happening
3. **Check console logs** - Detailed debugging info
4. **If still fails** - Share console logs for further diagnosis

---

## ✨ Result

The application now:
- ✅ **Checks 8+ IP field formats** - Much better detection
- ✅ **Shows clear progress** - Know what's happening
- ✅ **Logs detailed data** - Easy debugging
- ✅ **Waits longer** - Up to 10 minutes for IP
- ✅ **Better error messages** - Actionable guidance

---

**Status**: ✅ **Improved IP Detection Deployed**

The app should now successfully detect IP addresses in most cases! 🎉

---

## 📞 Still Having Issues?

If the updated app still can't get the IP:

1. **Open Developer Tools** (F12)
2. **Start training** and let it fail
3. **Copy the console logs** (especially "Full instance data")
4. **Check CanopyWave dashboard** - screenshot the instance details
5. **Share both** - we can identify the exact field name CanopyWave uses

This will help us add support for your specific CanopyWave configuration!
