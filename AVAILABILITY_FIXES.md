# GPU Availability False Positive Fixes

## ✅ All Issues Fixed

---

## **Issue #1: Frontend Defaults to "Available" on Any Failure** ❌→✅

### **Problem:**
```javascript
// OLD CODE - WRONG
cloudConfigGPUAvailable = true; // assume available to not block users
```

Any failure (API down, network error, parse error) would show GPU as "available" → **false positive**.

### **Fix:**
```javascript
// NEW CODE - CORRECT
cloudConfigGPUAvailable = null; // unknown state
```

**Tri-state system**:
- `true` = Available ✅
- `false` = Unavailable ❌
- `null` = Unknown ⚠️

**Behavior**:
- ✅ `true` → Continue button **enabled**
- ❌ `false` → Continue button **disabled**
- ⚠️ `null` → Continue button **disabled** (no false positives)

**UI Messages**:
- `null` from API failure: "⚠️ Unable to verify availability right now - check your API key or try again"
- `null` from network error: "⚠️ Unable to verify availability right now - network or API error"
- `null` from ambiguous data: "⚠️ Unable to verify availability - API returned ambiguous data"

---

## **Issue #2: Backend "OR" Logic Overrides Definitive "Unavailable"** ❌→✅

### **Problem:**
```javascript
// OLD CODE - WRONG
const isAvailable = availability && (
  availability.available === true ||
  availability.status === 'available' ||
  availability.in_stock === true ||
  (availability.count !== undefined && availability.count > 0)
);
```

**False positive example**:
```json
{ "available": false, "count": 5 }
```
Returns `true` because `count > 0`, even though `available` is explicitly `false`.

### **Fix:**
```javascript
// NEW CODE - CORRECT (with precedence)
function parseAvailability(a) {
  if (!a || typeof a !== 'object') return null; // unknown

  // 1) If API provides explicit boolean, trust it (HIGHEST PRIORITY)
  if (typeof a.available === 'boolean') return a.available;

  // 2) Normalize status strings
  if (typeof a.status === 'string') {
    const s = a.status.trim().toLowerCase();
    if (['available', 'in_stock', 'in-stock', 'ok', 'active'].includes(s)) return true;
    if (['unavailable', 'out_of_stock', 'out-of-stock', 'none', 'capacity_exhausted', 'exhausted'].includes(s)) return false;
  }

  // 3) Only use counts if field clearly indicates "available/free"
  const availableCountFields = ['available_count', 'free', 'free_count', 'remaining', 'remaining_count', 'capacity_available'];
  for (const k of availableCountFields) {
    if (typeof a[k] === 'number') return a[k] > 0;
  }

  // 4) Ambiguous "count" field - treat as UNKNOWN, not available
  if (typeof a.count === 'number') return null;

  return null; // unknown
}
```

**Precedence order**:
1. **`available` boolean** (highest priority - always wins)
2. **`status` string** (normalized, case-insensitive)
3. **Specific count fields** (`available_count`, `free_count`, etc.)
4. **Ambiguous `count`** → returns `null` (unknown), not `true`

**Result**: No more false positives from contradictory fields.

---

## **Issue #3: Status Field Parsing Too Strict** ❌→✅

### **Problem:**
```javascript
// OLD CODE - WRONG
availability.status === 'available' // exact match only
```

Missed: `"AVAILABLE"`, `"Available"`, `"in_stock"`, etc. → false negatives → fail open → false positives.

### **Fix:**
```javascript
// NEW CODE - CORRECT
const s = a.status.trim().toLowerCase();
if (['available', 'in_stock', 'in-stock', 'ok', 'active'].includes(s)) return true;
if (['unavailable', 'out_of_stock', 'out-of-stock', 'none', 'capacity_exhausted', 'exhausted'].includes(s)) return false;
```

**Now handles**:
- Case variations: `"AVAILABLE"`, `"Available"`, `"available"`
- Multiple formats: `"in_stock"`, `"in-stock"`, `"ok"`, `"active"`
- Unavailable states: `"unavailable"`, `"out_of_stock"`, `"capacity_exhausted"`

---

## **Issue #4: Race Conditions - Stale Responses** ❌→✅

### **Problem:**
User changes GPU quickly:
1. Select GPU A → Request A sent
2. Select GPU B → Request B sent
3. Request B returns → UI shows "✅ available"
4. Request A returns (stale) → UI shows "❌ unavailable" (wrong!)

### **Fix:**
```javascript
// NEW CODE - CORRECT
let gpuAvailReqId = 0; // Global request counter

async function checkGPUAvailability() {
    const reqId = ++gpuAvailReqId; // Increment for this request
    
    // ... API call ...
    
    // Ignore stale responses
    if (reqId !== gpuAvailReqId) {
        console.log('[GPU Availability] Ignoring stale response', { reqId, current: gpuAvailReqId });
        return;
    }
    
    // Only update UI if this is the latest request
    cloudConfigGPUAvailable = result.available;
}
```

**How it works**:
- Each request gets a unique ID
- If another request starts, the counter increments
- When response arrives, check if it's still the latest
- Ignore stale responses (don't update UI)

**Result**: UI always shows availability for the currently selected GPU.

---

## **Issue #5: Renderer Assumes DOM Exists** ❌→✅

### **Problem:**
```javascript
// OLD CODE - WRONG
const gpuHelpText = gpuSelect.parentElement.querySelector(...);
```

If `parentElement` is `null` → throws → catch → `cloudConfigGPUAvailable = true` → false positive.

### **Fix:**
```javascript
// NEW CODE - CORRECT
const gpuHelpText = gpuSelect?.parentElement?.querySelector('.form-help .privacy-text-small');
```

**Optional chaining** (`?.`):
- If `gpuSelect` is `null` → `gpuHelpText` is `undefined` (no throw)
- If `parentElement` is `null` → `gpuHelpText` is `undefined` (no throw)
- No more accidental false positives from DOM errors

---

## **Summary of Changes**

### **Backend (main.js)**
✅ Deterministic parsing with precedence  
✅ Explicit `available` boolean always wins  
✅ Ambiguous `count` returns `null`, not `true`  
✅ Normalized status strings (case-insensitive)  
✅ Returns tri-state: `true | false | null`

### **Frontend (renderer.js)**
✅ Tri-state availability: `true | false | null`  
✅ Race condition prevention (request ID counter)  
✅ Null-safe DOM access (optional chaining)  
✅ Continue button disabled for `null` (unknown)  
✅ Clear UI messages for each state  
✅ No more "fail open" false positives

---

## **New Behavior**

### **Scenario 1: API Returns Available**
```json
{ "available": true }
```
**Result**: ✅ "GPU is available" → Continue **enabled**

### **Scenario 2: API Returns Unavailable**
```json
{ "available": false }
```
**Result**: ❌ "GPU is currently unavailable" → Continue **disabled**

### **Scenario 3: API Returns Ambiguous Data**
```json
{ "count": 5 }  // no "available" field
```
**Result**: ⚠️ "Unable to verify availability - API returned ambiguous data" → Continue **disabled**

### **Scenario 4: API Call Fails**
Network error, API down, rate limited, etc.

**Result**: ⚠️ "Unable to verify availability right now - check your API key or try again" → Continue **disabled**

### **Scenario 5: User Changes GPU Quickly**
1. Select GPU A → Request A sent
2. Select GPU B → Request B sent
3. Request A returns (stale) → **Ignored**
4. Request B returns → UI updated

**Result**: UI always shows correct availability for currently selected GPU

---

## **Zero False Positives**

**Before**: Any error → "available" → user launches instance → fails → wastes money

**After**: Any error → "unknown" → Continue disabled → user must retry or fix issue → no wasted money

**Trade-off**: Slightly more friction (user must wait for successful check), but **zero false positives** and **correct behavior**.

---

## **Console Output**

```javascript
[Cloud Config] Validation result: {
  isValid: false,
  gpuAvailable: false,
  gpuAvailabilityState: 'unknown'  // or 'available' or 'unavailable'
}
```

---

## **Files Modified**

1. **`main.js`** (lines 1094-1126)
   - Added `parseAvailability()` function with precedence
   - Returns tri-state: `true | false | null`

2. **`renderer.js`** (lines 4278-4430)
   - Added `gpuAvailReqId` counter for race condition prevention
   - Changed `cloudConfigGPUAvailable` to tri-state
   - Added null-safe DOM access (`?.`)
   - Updated validation to require `=== true` (not just truthy)
   - Added UI messages for all three states

---

## **Testing**

### **Test Case 1: Normal Available**
1. Select GPU
2. API returns `{ "available": true }`
3. ✅ Shows "GPU is available" (green)
4. Continue button enabled

### **Test Case 2: Normal Unavailable**
1. Select GPU
2. API returns `{ "available": false }`
3. ❌ Shows "GPU is currently unavailable" (red)
4. Continue button disabled

### **Test Case 3: API Failure**
1. Disconnect internet
2. Select GPU
3. ⚠️ Shows "Unable to verify availability right now - network or API error" (orange)
4. Continue button disabled

### **Test Case 4: Ambiguous Response**
1. Select GPU
2. API returns `{ "count": 5 }` (no `available` field)
3. ⚠️ Shows "Unable to verify availability - API returned ambiguous data" (orange)
4. Continue button disabled

### **Test Case 5: Race Condition**
1. Quickly select GPU A, then GPU B
2. Request A returns after Request B
3. UI shows availability for GPU B (correct)
4. Request A is ignored (logged to console)

---

## **Future Enhancement Option**

If you want to allow users to proceed with unknown availability:

```javascript
// Add checkbox in UI
<input type="checkbox" id="proceedAnywayCheckbox">
<label>Proceed anyway (availability could not be verified)</label>

// In validation
const proceedAnyway = document.getElementById('proceedAnywayCheckbox')?.checked;
const gpuAvailable = !hasGPU || cloudConfigGPUAvailable === true || 
                     (cloudConfigGPUAvailable === null && proceedAnyway);
```

This gives users explicit control while still preventing false positives.
