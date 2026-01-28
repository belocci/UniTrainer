# GPU Availability Checking Code

## Overview
The app checks GPU availability using the CanopyWave API when a user selects a GPU.

---

## 📍 Code Locations

### 1. **Frontend (renderer.js)** - Lines 4361-4430

```javascript
// Check GPU availability
async function checkGPUAvailability() {
    // Get fresh reference to GPU select (it gets recreated dynamically)
    const gpuSelect = document.getElementById('cloudGPUSelect');
    
    const project = projectSelect ? projectSelect.value : '';
    const region = regionSelect ? regionSelect.value : '';
    const gpu = gpuSelect ? gpuSelect.value : '';
    
    if (!project || !region || !gpu || !canopywaveApiKey) {
        return;
    }
    
    try {
        // Show checking status
        const gpuHelpText = gpuSelect.parentElement.querySelector('.form-help .privacy-text-small');
        if (gpuHelpText) {
            gpuHelpText.textContent = '⏳ Checking GPU availability...';
            gpuHelpText.style.color = '#9A8A7A';
        }
        
        // Check availability via API
        const result = await ipcRenderer.invoke('check-gpu-availability', canopywaveApiKey, project, gpu, region);
        
        if (result.success && result.available !== undefined) {
            // Update global availability state
            cloudConfigGPUAvailable = result.available;
            
            if (gpuHelpText) {
                if (result.available) {
                    gpuHelpText.textContent = '✅ GPU is available';
                    gpuHelpText.style.color = '#27AE60';
                    gpuHelpText.style.fontWeight = '500';
                } else {
                    gpuHelpText.textContent = '❌ GPU is currently unavailable (in use or out of capacity)';
                    gpuHelpText.style.color = '#E74C3C';
                    gpuHelpText.style.fontWeight = '500';
                }
            }
            
            // Re-validate form to update Continue button state
            if (cloudConfigValidateForm) cloudConfigValidateForm();
        } else {
            // Reset to default if check fails (assume available to not block users)
            cloudConfigGPUAvailable = true;
            
            if (gpuHelpText) {
                gpuHelpText.textContent = '💡 Choose based on your budget and performance needs';
                gpuHelpText.style.color = '#9A8A7A';
                gpuHelpText.style.fontWeight = '400';
            }
            
            // Re-validate form
            if (cloudConfigValidateForm) cloudConfigValidateForm();
        }
    } catch (error) {
        console.error('Error checking GPU availability:', error);
        // Reset to default on error (assume available to not block users)
        cloudConfigGPUAvailable = true;
        
        const gpuHelpText = gpuSelect.parentElement.querySelector('.form-help .privacy-text-small');
        if (gpuHelpText) {
            gpuHelpText.textContent = '💡 Choose based on your budget and performance needs';
            gpuHelpText.style.color = '#9A8A7A';
            gpuHelpText.style.fontWeight = '400';
        }
        
        // Re-validate form
        if (cloudConfigValidateForm) cloudConfigValidateForm();
    }
}
```

**Triggered by**: When user selects a GPU from the dropdown

---

### 2. **Backend (main.js)** - Lines 1094-1126

```javascript
// Check GPU availability (simplified for UI)
ipcMain.handle('check-gpu-availability', async (event, apiKey, project, flavor, region) => {
  try {
    if (!apiKey || !apiKey.trim()) {
      return { success: false, error: 'API key is required' };
    }
    if (!project || !flavor || !region) {
      return { success: false, error: 'Project, flavor, and region are required' };
    }

    // Get or create API client
    let client = canopywaveClients.get(apiKey.trim());
    if (!client) {
      client = new CanopyWaveAPI(apiKey.trim());
      canopywaveClients.set(apiKey.trim(), client);
    }

    // Check availability
    const availability = await client.checkFlavorAvailability(project, flavor, region);
    
    // Parse the availability response
    // The API might return different formats, so we check multiple fields
    const isAvailable = availability && (
      availability.available === true ||
      availability.status === 'available' ||
      availability.in_stock === true ||
      (availability.count !== undefined && availability.count > 0)
    );
    
    return { success: true, available: isAvailable, details: availability };
  } catch (error) {
    console.error('[Main] Error checking GPU availability:', error);
    return { success: false, error: error.message || 'Failed to check availability' };
  }
});
```

**IPC Channel**: `check-gpu-availability`

---

### 3. **API Client (canopywave-api.js)** - Lines 313-315

```javascript
/**
 * Check flavor availability
 * GET /flavor-availability?project=<project>&flavor=<flavor>&region=<region>
 * @param {string} project - Project name
 * @param {string} flavor - Flavor name (e.g., "H100-8")
 * @param {string} region - Region name
 * @returns {Promise<object>} Availability information
 */
async checkFlavorAvailability(project, flavor, region) {
    return await this.request('/flavor-availability', 'GET', null, { project, flavor, region });
}
```

**API Endpoint**: `GET https://cloud-api.canopywave.io/api/v1/flavor-availability`

**Query Parameters**:
- `project` - Your CanopyWave project
- `flavor` - GPU type (e.g., "4x-H100-80-GB-SXM5")
- `region` - Region (e.g., "KEF-3")

---

## 🔄 Flow Diagram

```
User selects GPU
       ↓
checkGPUAvailability() (renderer.js)
       ↓
ipcRenderer.invoke('check-gpu-availability', ...)
       ↓
IPC Handler (main.js)
       ↓
client.checkFlavorAvailability(project, flavor, region)
       ↓
CanopyWave API: GET /flavor-availability
       ↓
Response: { available: true/false, ... }
       ↓
Parse response (check multiple fields)
       ↓
Return to renderer: { success: true, available: true/false }
       ↓
Update UI:
  - ✅ "GPU is available" (green) → Enable Continue
  - ❌ "GPU is currently unavailable" (red) → Disable Continue
```

---

## 📊 API Response Formats Handled

The code checks multiple possible response formats from CanopyWave:

```javascript
const isAvailable = availability && (
  availability.available === true ||        // Format 1
  availability.status === 'available' ||    // Format 2
  availability.in_stock === true ||         // Format 3
  (availability.count !== undefined && availability.count > 0)  // Format 4
);
```

**Examples**:

```json
// Format 1
{ "available": true }

// Format 2
{ "status": "available" }

// Format 3
{ "in_stock": true }

// Format 4
{ "count": 5 }
```

---

## 🎯 When It's Called

1. **When user selects a GPU** from the dropdown
2. **When region changes** (triggers re-check if GPU already selected)

**Event listener** (renderer.js):
```javascript
updatedGpuSelect.addEventListener('change', () => { 
    if (cloudConfigValidateForm) cloudConfigValidateForm(); 
    updateCloudCostEstimate(); 
    if (cloudConfigCheckGPUAvailability) cloudConfigCheckGPUAvailability(); 
});
```

---

## 🛡️ Error Handling

### If API call fails:
- Assumes GPU is **available** (to not block users)
- Shows default message
- Logs error to console
- Allows user to continue

### If API returns unexpected format:
- Checks multiple fields
- Returns `false` if none match
- User sees "unavailable" message

---

## 🔧 To Modify Availability Logic

**Location**: `main.js` lines 1114-1120

Change this section to adjust how availability is determined:

```javascript
const isAvailable = availability && (
  availability.available === true ||
  availability.status === 'available' ||
  availability.in_stock === true ||
  (availability.count !== undefined && availability.count > 0)
);
```

---

## 💡 Summary

**Files involved**:
1. `renderer.js` - Frontend logic (lines 4361-4430)
2. `main.js` - IPC handler (lines 1094-1126)
3. `canopywave-api.js` - API client (lines 313-315)

**API Endpoint**: 
```
GET https://cloud-api.canopywave.io/api/v1/flavor-availability
?project=X&flavor=Y&region=Z
```

**Result**: 
- ✅ Available → Enable Continue button
- ❌ Unavailable → Disable Continue button
