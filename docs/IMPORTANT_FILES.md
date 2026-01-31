# Important Files for Uni Trainer Project

This document lists all the critical files needed to understand and review the codebase.

## Core Application Files

### 1. **main.js** (207 lines)
- **Purpose**: Electron main process
- **Key Functions**:
  - Window creation and management
  - IPC handlers for model saving/loading
  - File system operations
  - Window controls (close button)
- **Location**: Root directory

### 2. **renderer.js** (~2,986 lines - VERY LARGE)
- **Purpose**: Main UI logic and event handling
- **Key Functions**:
  - System information display (CPU, GPU, Memory)
  - File upload handling
  - Training settings management (preset/manual modes)
  - Parameter calculation (architecture-based)
  - Training simulation and metrics
  - UI state management
  - Model configuration (framework, variant, format selection)
- **Location**: Root directory
- **Note**: This file is too large to read in one go. Key sections:
  - `calculateParametersFromArchitecture()` - Parameter calculation
  - `updateParameterCount()` - Updates parameter display
  - `applyPresetSettings()` / `applyManualSettings()` - Settings application
  - Training simulation logic
  - Event listeners for all UI controls

### 3. **neural-network.js** (1,354 lines)
- **Purpose**: Neural network visualization
- **Key Classes**:
  - `NeuralNetworkVisualization` - Main visualization class
- **Key Features**:
  - Canvas-based neural network rendering
  - Node and connection animations
  - Gradual appearance of nodes and connections
  - Weight-based brightness
  - Heartbeat sound effects
  - Tooltip system for node information
  - Smooth growth animations for connections
- **Location**: Root directory

### 4. **index.html** (467 lines)
- **Purpose**: HTML structure and UI layout
- **Key Sections**:
  - Splash screen
  - Main app container
  - Left panel (settings, controls)
  - Right panel (neural network visualization)
  - Inline script for splash screen handling
- **Location**: Root directory

### 5. **styles.css** (1,283 lines)
- **Purpose**: All styling and visual appearance
- **Key Features**:
  - Beige/nude color theme
  - Minimalistic design
  - Custom scrollbars
  - Button styles
  - Card layouts
  - Neural network visualization styles
- **Location**: Root directory

## Configuration Files

### 6. **package.json** (97 lines)
- **Purpose**: Project configuration and dependencies
- **Key Information**:
  - Dependencies: `systeminformation`
  - DevDependencies: `electron`, `electron-builder`, `electron-packager`, `sharp`, `to-ico`
  - Build scripts
  - Electron builder configuration
- **Location**: Root directory

## File Structure Summary

```
transfer package - uni trainer/
├── main.js                 # Electron main process
├── renderer.js             # UI logic (VERY LARGE - ~3K lines)
├── neural-network.js       # Neural network visualization
├── index.html              # HTML structure
├── styles.css              # All CSS styling
├── package.json            # Project configuration
├── package-lock.json       # Dependency lock file
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── BUILD.md               # Build instructions
├── BUILD_INSTRUCTIONS.md  # Build instructions
├── DISTRIBUTION.md        # Distribution info
└── generate-icons.js      # Icon generation script
```

## Key Functionality Areas

### Parameter Calculation
- **File**: `renderer.js`
- **Function**: `calculateParametersFromArchitecture()`
- **Logic**: Based on model type, framework, variant, and quality settings
- **Updates**: Triggered on slider changes and manual setting changes

### Neural Network Visualization
- **File**: `neural-network.js`
- **Class**: `NeuralNetworkVisualization`
- **Features**:
  - Gradual node appearance
  - Smooth connection growth
  - Weight-based brightness
  - Heartbeat animations
  - Tooltip system

### Training Settings
- **File**: `renderer.js`
- **Modes**: Preset (slider) and Manual (dropdowns)
- **Functions**: `applyPresetSettings()`, `applyManualSettings()`

### Model Management
- **File**: `main.js` (IPC handlers)
- **Features**: Save, load, list models
- **Storage**: `Documents/UniTrainer/Models/`

## Notes for Code Review

1. **renderer.js is very large** (~3,000 lines) - may need to be reviewed in sections
2. **Connection growth logic** in `neural-network.js` (lines 914-1002) - recently fixed for smooth growth
3. **Parameter calculation** in `renderer.js` - architecture-based, not dataset-based
4. **Settings application** - both preset and manual modes update parameters immediately

## Recent Changes

- Fixed connection growth to be smooth and gradual
- Parameter calculation now architecture-based
- Removed DevTools shortcut
- Improved connection visibility logic
- Enhanced node appearance animations
