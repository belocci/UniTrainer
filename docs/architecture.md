# Architecture Overview

Uni Trainer is an Electron desktop app with a Python ML backend.

## High-level Components
- Electron main process: app lifecycle, IPC, file system access, and Python process orchestration.
- Renderer (UI): user interface, training status, logs, and inference results.
- Python backend: training and inference for tabular and computer vision workflows.

## Data Flow
1. UI collects inputs and sends commands via IPC.
2. Main process validates inputs and spawns Python tasks.
3. Python writes logs and results back to the main process.
4. UI renders progress and outputs.
