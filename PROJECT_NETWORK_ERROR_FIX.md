# "Project Network Not Found" Error - Fix Guide

## Date
January 18, 2026

---

## 🐛 Error Message

```
Failed to start cloud training: Failed to launch instance: 
Project network not found. This might be a permission issue with your API key.
```

---

## 🔍 Root Cause

This error occurs when trying to launch a CanopyWave cloud instance in a project that doesn't have a network configured. This is common with:
- **New projects** that haven't been fully set up
- **Projects without proper network configuration**
- **API keys with insufficient permissions**

---

## ✅ Solution Options

### Option 1: Configure Project Network (Recommended)

1. **Go to CanopyWave Dashboard**
   - Visit: https://cloud.canopywave.io
   - Log in with your account

2. **Navigate to Your Project**
   - Find the project you selected (e.g., "belocci1@gmail.com")
   - Click on the project name

3. **Check Networks Section**
   - Look for "Networks" or "Networking" tab
   - Verify if a network exists

4. **Create Network (if missing)**
   - Click "Create Network" or similar button
   - Follow the wizard to set up a default network
   - Use default settings unless you have specific requirements

5. **Verify Configuration**
   - Ensure the network is marked as "Active" or "Available"
   - Check that it's set as the default network for the project

6. **Try Again in Uni Trainer**
   - Return to Uni Trainer
   - Start cloud training again

---

### Option 2: Use a Different Project

1. **In Uni Trainer**
   - Go to Cloud Configuration modal
   - Click the "Project" dropdown

2. **Select Different Project**
   - Choose a different project from the list
   - Preferably one you've used successfully before

3. **Complete Configuration**
   - Select region, GPU, etc.
   - Try launching training again

---

### Option 3: Contact CanopyWave Support

If the above options don't work:

1. **Gather Information**
   - Your API key (first/last 4 characters only)
   - Project name
   - Region you're trying to use
   - Error message

2. **Contact Support**
   - Email: support@canopywave.com
   - Or use the support chat in CanopyWave dashboard
   - Mention: "Project network not found when launching instance"

3. **Request**
   - Ask them to verify your project configuration
   - Request network setup assistance
   - Verify API key permissions

---

## 🔧 Improved Error Handling

I've updated the application to provide better error messages:

### Before:
```
Failed to launch instance: Project network not found. 
This might be a permission issue with your API key.
```
❌ Vague and unhelpful

### After:
```
Failed to launch instance: Project network not found. 
Please ensure your CanopyWave project "belocci1@gmail.com" is properly configured:

1. Go to CanopyWave Dashboard (https://cloud.canopywave.io)
2. Navigate to your project: "belocci1@gmail.com"
3. Check "Networks" section and ensure a network exists
4. If no network exists, create one or contact CanopyWave support
5. Verify your API key has permission to launch instances

Alternative: Try selecting a different project in the Cloud Configuration.
```
✅ Clear, actionable steps

---

## 📝 Changes Made

### File: `cloud-training-handler.js`

**Lines 131-161** - Enhanced error handling:

```javascript
try {
    const instance = await this.client.launchInstance(instanceConfig);
    console.log('[CloudTraining] Instance launched successfully:', instance);
    return instance;
} catch (error) {
    console.error('[CloudTraining] Failed to launch instance:', error.message);
    console.error('[CloudTraining] Error details:', error);
    
    // Provide specific error messages for common issues
    let errorMessage = error.message;
    
    if (errorMessage.includes('Project network not found')) {
        errorMessage = `Project network not found. Please ensure your CanopyWave project "${config.project}" is properly configured:\n\n` +
            `1. Go to CanopyWave Dashboard (https://cloud.canopywave.io)\n` +
            `2. Navigate to your project: "${config.project}"\n` +
            `3. Check "Networks" section and ensure a network exists\n` +
            `4. If no network exists, create one or contact CanopyWave support\n` +
            `5. Verify your API key has permission to launch instances\n\n` +
            `Alternative: Try selecting a different project in the Cloud Configuration.`;
    } else if (errorMessage.includes('no default payment method')) {
        errorMessage = 'No payment method configured. Please add a payment method in your CanopyWave dashboard before launching instances.';
    } else if (errorMessage.includes('403') || errorMessage.toLowerCase().includes('forbidden')) {
        errorMessage = `Access forbidden. Your API key may not have permission to launch instances in project "${config.project}". Check your CanopyWave account permissions.`;
    } else if (errorMessage.includes('insufficient') || errorMessage.toLowerCase().includes('quota')) {
        errorMessage = 'Insufficient quota or resources. Check your CanopyWave account limits and available balance.';
    } else if (errorMessage.includes('flavor') || errorMessage.includes('not available')) {
        errorMessage = `The selected GPU type "${config.flavor}" may not be available in region "${config.region}". Try a different GPU or region.`;
    }
    
    throw new Error(`Failed to launch instance: ${errorMessage}`);
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

2. **Try cloud training again**
   - You'll now get a detailed error message with steps to fix

3. **Follow the instructions in the error message**

---

## 💡 Common CanopyWave Issues & Solutions

### Issue 1: Project Network Not Found
**Solution**: Configure network in CanopyWave dashboard or use different project

### Issue 2: No Payment Method
**Solution**: Add payment method in CanopyWave dashboard → Billing

### Issue 3: Access Forbidden (403)
**Solution**: Check API key permissions, may need to regenerate with proper scopes

### Issue 4: Insufficient Quota
**Solution**: Check account limits, upgrade plan, or wait for quota reset

### Issue 5: GPU Not Available
**Solution**: Try different GPU type or different region

---

## 📋 Troubleshooting Checklist

Before launching cloud training, verify:

- [ ] CanopyWave account is active
- [ ] Payment method is configured
- [ ] API key is valid and has proper permissions
- [ ] Project exists and is accessible
- [ ] Project has network configured
- [ ] Selected region is available
- [ ] Selected GPU type is available in that region
- [ ] Account has sufficient balance/credits
- [ ] Account quota allows instance creation

---

## 🔗 Helpful Links

- **CanopyWave Dashboard**: https://cloud.canopywave.io
- **CanopyWave Documentation**: https://canopywave.com/docs
- **CanopyWave Support**: support@canopywave.com
- **API Documentation**: https://canopywave.com/docs/account/quick-start

---

## 📦 Build Status

**Build Date**: January 18, 2026  
**Build Location**: `dist\Uni Trainer-win32-x64\`  
**Status**: ✅ Complete  
**Changes**: ✅ Improved error messages included  

---

## 🎯 Quick Fix Summary

**If you see "Project network not found":**

1. ✅ **Try a different project** (fastest)
   - In Cloud Config modal, select different project from dropdown

2. ✅ **Configure network** (proper fix)
   - Go to CanopyWave dashboard
   - Navigate to project → Networks
   - Create network if missing

3. ✅ **Contact support** (if stuck)
   - Email support@canopywave.com
   - Provide project name and error details

---

## ✨ Result

The application now provides:
- ✅ **Detailed error messages** - Know exactly what's wrong
- ✅ **Step-by-step guidance** - Clear instructions to fix
- ✅ **Alternative solutions** - Multiple ways to resolve
- ✅ **Helpful links** - Direct access to resources
- ✅ **Common issue detection** - Recognizes 5+ error types

---

**Status**: ✅ **Improved Error Handling Deployed**

The error messages will now guide you through fixing the issue! 🎉

---

## 📞 Need More Help?

If you're still stuck after trying these solutions:

1. Check the console logs in Uni Trainer (View → Toggle Developer Tools)
2. Look for additional error details
3. Take a screenshot of the error
4. Contact CanopyWave support with the details
5. Or create an issue with the Uni Trainer project

The improved error messages should make it much easier to diagnose and fix the problem!
