# CanopyWave API Checklist - Finding the Public IP Parameter

## 🎯 Goal
Find the correct parameter to automatically associate a public IP when launching a CanopyWave instance.

---

## 📋 Method 1: Check API Documentation

### Where to Look
- **Main docs:** https://canopywave.com/docs or https://cloud.canopywave.io/docs
- **Look for:** "Launch Instance", "Create Instance", or "Deploy Instance"
- **Endpoint:** `/instance-operations/launch` (this is what we're using)

### What to Find
The parameter that requests a public IP. It might be called:
- `associate_public_ip`
- `assign_public_ip`
- `public_ip`
- `network_config`
- `floating_ip`
- `auto_assign_ip`
- Or something similar

### Example of What We Need
If the docs show something like:
```json
{
  "project": "my-project",
  "region": "KEF-2",
  "flavor": "H100-2",
  "image": "GPU-Ubuntu.22.04",
  "password": "mypassword",
  "associate_public_ip": true  // ← THIS IS WHAT WE NEED
}
```

Then tell me: **"The parameter is `associate_public_ip: true`"**

---

## 📋 Method 2: Capture the Request from Dashboard

### Steps
1. Open CanopyWave dashboard in Chrome/Edge
2. Press **F12** to open Developer Tools
3. Go to **Network** tab
4. Click **"Preserve log"** checkbox
5. Manually create a new instance with these settings:
   - Enable "Associate Public IP" or similar option
   - Use any GPU/region
6. Look for a request to `/instance-operations/launch` or similar
7. Click on it and go to **Payload** or **Request** tab
8. Copy the entire JSON payload

### What to Share
Paste the entire request body here (remove your password/API key first).

Example:
```json
{
  "project": "my-project",
  "region": "KEF-2",
  "name": "test-instance",
  "flavor": "H100-2",
  "image": "GPU-Ubuntu.22.04",
  "password": "REDACTED",
  "network": "default",
  "assign_ip": true  // ← We need to know this parameter
}
```

---

## 📋 Method 3: Ask CanopyWave Support

### Question to Ask
> "What parameter should I include in the `/instance-operations/launch` API request to automatically associate a public IP address to the instance? I'm currently getting instances without public IPs and have to manually click 'Associate IP' in the dashboard."

### Share Their Response
Copy and paste their entire response here.

---

## 🔍 Current Launch Request (What We're Sending)

This is what Uni Trainer currently sends to CanopyWave:

```javascript
{
    project: "your-project",
    region: "KEF-2",
    name: "unitrainer-1234567890",
    flavor: "H100-2",
    image: "GPU-Ubuntu.22.04",
    password: "your-password",
    is_monitoring: true
}
```

**Missing:** The parameter to request a public IP.

---

## ✅ Once You Find It

Tell me one of these:

**Option A:** "Add `associate_public_ip: true` to the payload"

**Option B:** "Add `network_config: { public_ip: true }` to the payload"

**Option C:** "It requires a separate API call after launch: POST /floating-ips/associate"

**Option D:** Paste the full example from their docs

Then I'll implement it and rebuild!

---

## 🚨 Common Mistakes to Avoid

❌ Don't just guess - we need the exact parameter name from CanopyWave
❌ Don't confuse "monitoring" with "public IP"
❌ Don't assume it's the same as AWS/Azure/GCP (each cloud is different)

✅ Get the exact parameter from docs, dashboard capture, or support
✅ Share the full example if possible
✅ Include any nested objects (like `network_config: { ... }`)

---

## 📞 CanopyWave Contact Info

- **Website:** https://canopywave.com
- **Dashboard:** https://cloud.canopywave.io
- **Support:** Look for "Contact" or "Support" in their dashboard
- **Discord/Slack:** Check if they have a community channel

---

**Take your time to find the correct parameter. Once we have it, the fix will take 2 minutes!** 🚀
