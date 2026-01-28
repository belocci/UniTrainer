# CanopyWave 500 Error - Troubleshooting Guide

## 🔴 Error: "Request failed with status code 500"

This error means **CanopyWave's API server had an internal error**. This is **not a bug in Uni Trainer** - it's a temporary issue on CanopyWave's side.

---

## 🔍 What Causes This

**HTTP 500 = Internal Server Error** on CanopyWave's servers. Common causes:

1. **Temporary server issue** - CanopyWave's API is experiencing problems
2. **Maintenance** - CanopyWave may be updating their systems
3. **Database issue** - Their backend database may be temporarily unavailable
4. **Rate limiting** - Too many requests in a short time (though usually returns 429)
5. **Account issue** - Your CanopyWave account may have a configuration problem

---

## ✅ Solutions (Try in Order)

### Solution 1: Wait and Retry (Most Common)
**This usually resolves itself within minutes.**

1. Wait **2-5 minutes**
2. Close and reopen Uni Trainer
3. Try entering your API key again
4. If projects load, the issue is resolved ✅

---

### Solution 2: Check CanopyWave Status

1. Go to: https://cloud.canopywave.io
2. Try logging into the dashboard manually
3. Check if you can see your projects in the web interface
4. If the website is slow or showing errors, CanopyWave is having issues

---

### Solution 3: Verify Your API Key

1. Go to: https://cloud.canopywave.io
2. Navigate to **API Keys** section
3. Check if your API key is still valid
4. Try **regenerating a new API key**
5. Use the new key in Uni Trainer

---

### Solution 4: Check Your Account

1. Log into CanopyWave dashboard
2. Check if your account has any alerts or warnings
3. Verify your payment method is active
4. Ensure your account is in good standing

---

### Solution 5: Contact CanopyWave Support

If the error persists for more than 30 minutes:

1. Go to: https://canopywave.com/contact or https://cloud.canopywave.io
2. Look for "Support" or "Contact" section
3. Report the issue:
   ```
   Subject: API Error 500 when calling /projects endpoint
   
   Description:
   I'm getting a 500 error when trying to list projects via the API.
   
   API Endpoint: GET https://cloud-api.canopywave.io/api/v1/projects
   Error: Request failed with status code 500
   Time: [Current date/time]
   API Key: [Last 4 characters of your key]
   
   Please investigate this issue.
   ```

---

## 🧪 Test If CanopyWave API Is Working

You can test the API manually using PowerShell:

```powershell
$apiKey = "YOUR_API_KEY_HERE"
$headers = @{
    "Authorization" = "Bearer $apiKey"
    "Content-Type" = "application/json"
}

try {
    $response = Invoke-RestMethod -Uri "https://cloud-api.canopywave.io/api/v1/projects" -Headers $headers -Method Get
    Write-Host "✅ API is working! Projects:" -ForegroundColor Green
    $response.data
} catch {
    Write-Host "❌ API Error:" -ForegroundColor Red
    $_.Exception.Message
}
```

**If this also returns a 500 error, it confirms the issue is on CanopyWave's side.**

---

## 📊 What Uni Trainer Now Does

**Improved error handling (latest build):**

1. ✅ Detects 500 errors specifically
2. ✅ Shows clear error message: "CanopyWave server error (500): This is a temporary issue on CanopyWave's side"
3. ✅ Suggests waiting and retrying
4. ✅ Updates dropdown to show "Error loading projects - see logs"
5. ✅ Logs full error details to console for debugging

---

## 🔄 When to Retry

**Retry after:**
- ✅ 2-5 minutes (for temporary glitches)
- ✅ 15-30 minutes (for maintenance)
- ✅ Checking CanopyWave's status page/dashboard
- ✅ Regenerating your API key

**Don't retry:**
- ❌ Immediately (may make it worse)
- ❌ Repeatedly in quick succession (may trigger rate limiting)

---

## 💡 Why This Happens

**CanopyWave is a cloud provider, and like all cloud services:**
- They occasionally have server issues
- They perform maintenance
- Their API may experience temporary outages
- This is normal and expected for any cloud service

**This is NOT:**
- ❌ A bug in Uni Trainer
- ❌ A problem with your computer
- ❌ An issue with your API key (unless it's invalid)

---

## 📝 Latest Build Info

**Build:** 2026-01-18 01:40 AM  
**Asar size:** 15.4 MB  
**Location:** `dist\Uni Trainer-win32-x64\Uni Trainer.exe`

**Improvements:**
- Better 500 error detection and messaging
- Clear user feedback in UI
- Detailed error logging
- Suggestions for resolution

---

## 🎯 Quick Checklist

- [ ] Waited 2-5 minutes and retried
- [ ] Checked CanopyWave dashboard (https://cloud.canopywave.io)
- [ ] Verified API key is valid
- [ ] Checked account status
- [ ] Tested API manually (PowerShell command above)
- [ ] Contacted CanopyWave support (if issue persists > 30 min)

---

**Most likely:** This is a temporary issue that will resolve itself within a few minutes. Just wait and retry! ⏳
