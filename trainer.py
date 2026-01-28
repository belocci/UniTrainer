#!/usr/bin/env python3
"""
Uni Trainer - Real Model Training Script
Handles training for all supported model types and frameworks
"""

import sys
import json
import os
import traceback
import time
from pathlib import Path
from datetime import datetime

# CRITICAL: Add bundled site-packages to Python path if running from Electron app
# This ensures ultralytics and other packages can be found
if hasattr(sys, 'frozen') or getattr(sys, '_MEIPASS', None):
    # Running as bundled executable (PyInstaller, etc.)
    pass
else:
    # Try to find site-packages relative to this script or Python executable
    python_exe = sys.executable
    if python_exe:
        python_dir = os.path.dirname(os.path.abspath(python_exe))
        site_packages = os.path.join(python_dir, 'Lib', 'site-packages')
        if os.path.exists(site_packages) and site_packages not in sys.path:
            sys.path.insert(0, site_packages)
            # Also try parent directory structure (for Electron bundled Python)
            alt_site_packages = os.path.join(os.path.dirname(python_dir), 'Lib', 'site-packages')
            if os.path.exists(alt_site_packages) and alt_site_packages not in sys.path:
                sys.path.insert(0, alt_site_packages)

# Import training modules (will be installed via requirements)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Warning: PyTorch not available: {e}", file=sys.stderr)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"Warning: ultralytics not available: {e}", file=sys.stderr)
    print(f"Python path: {sys.executable}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import pickle
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def send_progress(epoch, total_epochs, loss, accuracy, status="training"):
    """Send training progress to Electron app"""
    progress_data = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": float(loss) if loss is not None else None,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "status": status,
        "progress": float(epoch / total_epochs) if total_epochs > 0 else 0.0
    }
    print(json.dumps({"type": "progress", "data": progress_data}))
    sys.stdout.flush()


def send_log(message, level="info"):
    """Send log message to Electron app"""
    log_data = {"message": str(message), "level": level}
    print(json.dumps({"type": "log", "data": log_data}))
    sys.stdout.flush()


def send_error(error_message, traceback_str=None):
    """Send error to Electron app"""
    error_data = {"error": str(error_message), "traceback": traceback_str}
    print(json.dumps({"type": "error", "data": error_data}))
    sys.stdout.flush()


def send_result(model_path, metrics, format_type):
    """Send training completion result"""
    result_data = {
        "model_path": str(model_path),
        "metrics": metrics,
        "format": format_type,
        "status": "completed"
    }
    print(json.dumps({"type": "result", "data": result_data}))
    sys.stdout.flush()


def prepare_yolo_dataset(dataset_path):
    """Prepare dataset for YOLO training - auto-organize if needed"""
    try:
        import yaml
        import shutil
    except ImportError:
        send_log("Warning: yaml module not available, using basic dataset preparation")
        yaml = None
    
    dataset_path = Path(dataset_path)
    send_log(f"Preparing dataset at: {dataset_path}")
    
    # 1. Check if already YOLO format (has data.yaml)
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        send_log(f"Found existing data.yaml at: {yaml_path}")
        # Validate and fix paths in data.yaml
        try:
            if yaml:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f) or {}
                
                # Always update path to current dataset location (CRITICAL - YOLO needs correct path)
                old_path = str(yaml_data.get('path', ''))
                new_path = str(dataset_path)
                yaml_data['path'] = new_path
                if old_path != new_path:
                    needs_update = True
                    send_log(f"Updating path from '{old_path}' to '{new_path}'")
                
                # Check if train/val paths exist relative to dataset_path
                if 'train' in yaml_data:
                    train_rel = str(yaml_data['train']).replace('/', os.sep)
                    train_abs = dataset_path / train_rel
                    # Check if path exists and has images
                    if not train_abs.exists():
                        # Try to find correct train directory
                        for possible in ['train', 'images/train', 'train/images']:
                            test_path = dataset_path / possible
                            if test_path.exists() and (any(test_path.rglob('*.jpg')) or any(test_path.rglob('*.png'))):
                                yaml_data['train'] = possible.replace(os.sep, '/')
                                needs_update = True
                                send_log(f"Found train directory at: {possible}")
                                break
                    else:
                        send_log(f"Train path exists: {train_abs}")
                
                if 'val' in yaml_data:
                    val_rel = str(yaml_data['val']).replace('/', os.sep)
                    val_abs = dataset_path / val_rel
                    if not val_abs.exists():
                        # Try to find correct val directory
                        for possible in ['valid', 'val', 'images/val', 'valid/images']:
                            test_path = dataset_path / possible
                            if test_path.exists():
                                yaml_data['val'] = possible.replace(os.sep, '/')
                                needs_update = True
                                send_log(f"Found val directory at: {possible}")
                                break
                        # If no val found, use train
                        final_val_rel = str(yaml_data.get('val', '')).replace('/', os.sep)
                        if not (dataset_path / final_val_rel).exists():
                            yaml_data['val'] = yaml_data.get('train', 'images/train')
                            needs_update = True
                    else:
                        send_log(f"Val path exists: {val_abs}")
                
                # Always write the YAML file to ensure path is updated
                send_log("Updating data.yaml with correct paths...")
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
                send_log(f"Updated data.yaml at: {yaml_path}")
                send_log(f"Path: {yaml_data.get('path')}, Train: {yaml_data.get('train')}, Val: {yaml_data.get('val')}")
            else:
                # If yaml module not available, read as text and try to fix path
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Always update path to current dataset location
                if 'path:' in content:
                    send_log("Updating path in data.yaml...")
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        if line.strip().startswith('path:'):
                            new_lines.append(f"path: {dataset_path}")
                        else:
                            new_lines.append(line)
                    with open(yaml_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_lines))
                    send_log(f"Updated data.yaml at: {yaml_path}")
        except Exception as e:
            send_log(f"Warning: Could not validate data.yaml paths: {e}")
            import traceback
            send_log(f"Traceback: {traceback.format_exc()}")
            send_log("Will attempt to use existing data.yaml as-is")
        
        return str(yaml_path)
    
    # 2. Check for existing YOLO structure (images/labels directories)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    if images_dir.exists() and labels_dir.exists():
        send_log("Found images/labels structure")
        # Determine train/val paths
        train_images = images_dir / 'train'
        val_images = images_dir / 'val'
        
        if train_images.exists():
            train_path = 'images/train'
        else:
            train_path = 'images'
        
        if val_images.exists():
            val_path = 'images/val'
        else:
            val_path = train_path  # Use same as train if no val
        
        # Create data.yaml
        if yaml:
            yaml_content = {
                'path': str(dataset_path),
                'train': train_path,
                'val': val_path,
                'nc': 1,  # Default, user should update
                'names': ['object']
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
        else:
            yaml_content = f"""path: {dataset_path}
train: {train_path}
val: {val_path}
nc: 1
names: ['object']"""
            yaml_path.write_text(yaml_content)
        
        send_log(f"Created data.yaml at: {yaml_path}")
        return str(yaml_path)
    
    # 3. Auto-organize dataset - find images and create structure
    send_log("Auto-organizing dataset...")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    images = []
    
    # Find images in root and subdirectories
    for ext in image_extensions:
        images.extend(list(dataset_path.glob(f'*{ext}')))
        images.extend(list(dataset_path.glob(f'*{ext.upper()}')))
        # Check subdirectories (one level deep)
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                images.extend(list(subdir.glob(f'*{ext}')))
                images.extend(list(subdir.glob(f'*{ext.upper()}')))
    
    send_log(f"Found {len(images)} images")
    
    if len(images) == 0:
        raise ValueError(f"No images found in {dataset_path}")
    
    # Create YOLO structure
    train_images_dir = dataset_path / 'images' / 'train'
    train_labels_dir = dataset_path / 'labels' / 'train'
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/move images to train directory
    copied_count = 0
    for img in images[:500]:  # Limit to first 500 to avoid issues
        dest = train_images_dir / img.name
        if img != dest:
            try:
                shutil.copy2(img, dest)
                copied_count += 1
            except Exception as e:
                send_log(f"Warning: Could not copy {img.name}: {e}")
    
    send_log(f"Copied {copied_count} images to images/train/")
    
    # Check for existing label files (.txt files that look like YOLO format)
    label_extensions = ['.txt']
    labels_found = 0
    
    for ext in label_extensions:
        label_files = list(dataset_path.glob(f'*{ext}'))
        for txt_file in label_files[:500]:  # Limit processing
            try:
                # Check if it looks like YOLO label (class_id x y w h)
                with open(txt_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        parts = content.split()
                        if len(parts) >= 5:
                            # Try to parse as YOLO format
                            try:
                                class_id = int(parts[0])
                                coords = [float(x) for x in parts[1:5]]
                                if all(0 <= coord <= 1 for coord in coords):
                                    # Looks like YOLO label
                                    dest = train_labels_dir / txt_file.name
                                    shutil.copy2(txt_file, dest)
                                    labels_found += 1
                            except (ValueError, IndexError):
                                pass
            except Exception:
                pass
    
    # If no labels found, create placeholder labels for a few images (for testing)
    if labels_found == 0:
        send_log("No label files found. Creating placeholder labels for testing...")
        placeholder_count = min(10, len(images))
        for img in images[:placeholder_count]:
            label_path = train_labels_dir / f'{img.stem}.txt'
            if not label_path.exists():
                # Create a dummy label (class 0, center of image, small box)
                label_path.write_text('0 0.5 0.5 0.1 0.1')
    
    # Create data.yaml
    if yaml:
        yaml_content = {
            'path': str(dataset_path),
            'train': 'images/train',
            'val': 'images/train',  # Same as train for now
            'nc': 1,  # User should update this
            'names': ['object']  # User should update this
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
    else:
        yaml_content = f"""path: {dataset_path}
train: images/train
val: images/train
nc: 1
names: ['object']"""
        yaml_path.write_text(yaml_content)
    
    send_log(f"Dataset prepared! Created data.yaml at: {yaml_path}")
    send_log(f"Structure: {dataset_path}/images/train/ - {copied_count} images")
    send_log(f"Labels: {dataset_path}/labels/train/ - {labels_found if labels_found > 0 else 'placeholder'} labels")
    
    return str(yaml_path)


def load_dataset(data_dir, model_purpose, framework):
    """Load and preprocess dataset based on model type"""
    data_path = Path(data_dir)
    
    if model_purpose == "computer_vision":
        # For CV: expect YOLO format (images + labels) or image folder
        # For now, return placeholder - YOLO training will handle its own dataset loading
        return None
    elif model_purpose in ["machine_learning", "time_series"]:
        # Load CSV or JSON files
        csv_files = list(data_path.glob("*.csv"))
        json_files = list(data_path.glob("*.json"))
        
        if csv_files:
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas required for CSV datasets")
            df = pd.read_csv(csv_files[0])
            # Assume last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return (X, y)
        elif json_files:
            # Try to load as structured data
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            # Placeholder - would need proper parsing
            return data
    elif model_purpose == "natural_language_processing":
        # Load text files
        text_files = list(data_path.glob("*.txt")) + list(data_path.glob("*.json"))
        texts = []
        for file in text_files:
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        return texts
    
    return None


def train_yolo_model(config):
    """Train YOLO model for computer vision"""
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
    
    data_dir = config["data_dir"]
    variant = config["variant"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    output_dir = config["output_dir"]
    model_format = config.get("format", "pt")
    device = config.get("device", "auto")
    
    # Map variant to YOLO model name
    variant_map = {
        "yolov11n": "yolo11n.pt",
        "yolov11s": "yolo11s.pt",
        "yolov11m": "yolo11m.pt",
        "yolov11l": "yolo11l.pt",
        "yolov11x": "yolo11x.pt"
    }
    
    model_name = variant_map.get(variant, "yolo11s.pt")
    
    send_log(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    # Prepare dataset - check for YOLO format and auto-organize if needed
    data_yaml = prepare_yolo_dataset(data_dir)
    
    send_log(f"Starting YOLO training with {epochs} epochs, batch size {batch_size}")
    
    # Determine device to use
    if device == "auto":
        # Auto-detect: use GPU if available, otherwise CPU
        train_device = 0 if torch.cuda.is_available() else 'cpu'
    elif device == "cuda":
        # Force GPU (will fail if not available, but that's user's choice)
        if torch.cuda.is_available():
            train_device = 0
        else:
            send_log("Warning: CUDA/GPU requested but not available. PyTorch CPU-only version is installed.", "warning")
        send_log("To use GPU: Install PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", "warning")
        train_device = 'cpu'
    else:
        # Force CPU
        train_device = 'cpu'
    
    device_name = "GPU" if train_device == 0 else "CPU"
    send_log(f"Using device: {device_name}")
    
    # Train the model - progress will be parsed from stdout output
    # Send initial progress to trigger UI update
    send_progress(0, epochs, 1.0, 0.0, "starting")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        lr0=learning_rate,
        device=train_device,
        project=str(output_dir),
        name="train",
        verbose=True
    )
    
    # Get best model path
    best_model = Path(output_dir) / "train" / "weights" / "best.pt"
    
    # Export to requested format
    if model_format != "pt" and best_model.exists():
        send_log(f"Exporting model to {model_format} format...")
        export_path = model.export(format=model_format.replace("torchscript", "torchscript"))
        best_model = Path(export_path)
    
    metrics = {
        "loss": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "accuracy": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)) * 100,
        "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0.0))
    }
    
    return best_model, metrics


def train_sklearn_model(config):
    """Train scikit-learn model"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
    
    data_dir = config["data_dir"]
    variant = config["variant"]
    output_dir = config["output_dir"]
    validation_split = config.get("validation_split", 0.2)
    model_format = config.get("format", "pkl")
    
    # Load dataset
    dataset = load_dataset(data_dir, "machine_learning", "sklearn")
    if dataset is None:
        raise ValueError("Could not load dataset. Ensure CSV or JSON files are in data directory")
    
    X, y = dataset
    
    # Encode labels if needed
    if isinstance(y[0], str):
        le = LabelEncoder()
        y = le.encode(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create model based on variant
    model_map = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    model = model_map.get(variant, RandomForestClassifier(n_estimators=100, random_state=42))
    
    send_log(f"Training {variant} model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    send_log(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    if model_format == "pkl":
        model_path = Path(output_dir) / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
    elif model_format == "joblib":
        model_path = Path(output_dir) / "model.joblib"
        joblib.dump({"model": model, "scaler": scaler}, model_path)
    else:
        raise ValueError(f"Unsupported format for sklearn: {model_format}")
    
    metrics = {"accuracy": float(accuracy * 100), "loss": None}
    return model_path, metrics


def train_xgboost_model(config):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost not installed. Install with: pip install xgboost")
    
    data_dir = config["data_dir"]
    variant = config["variant"]
    output_dir = config["output_dir"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    validation_split = config.get("validation_split", 0.2)
    model_format = config.get("format", "json")
    
    # Load dataset
    dataset = load_dataset(data_dir, "machine_learning", "xgboost")
    if dataset is None:
        raise ValueError("Could not load dataset")
    
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train model
    params = {
        "objective": "multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
        "num_class": len(np.unique(y)) if len(np.unique(y)) > 2 else None,
        "learning_rate": learning_rate,
        "max_depth": 6,
        "eval_metric": "mlogloss" if len(np.unique(y)) > 2 else "logloss"
    }
    params = {k: v for k, v in params.items() if v is not None}
    
    send_log(f"Training XGBoost model for {epochs} epochs...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=epochs,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False
    )
    
    # Evaluate
    predictions = model.predict(dtest)
    if len(np.unique(y)) > 2:
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions == y_test)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    if model_format == "json":
        model_path = Path(output_dir) / "model.json"
        model.save_model(str(model_path))
    elif model_format == "ubj":
        model_path = Path(output_dir) / "model.ubj"
        model.save_model(str(model_path))
    else:
        raise ValueError(f"Unsupported format for xgboost: {model_format}")
    
    metrics = {"accuracy": float(accuracy * 100), "loss": None}
    return model_path, metrics


def train_pytorch_mlp(config):
    """Train PyTorch MLP for machine learning"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Install with: pip install torch")
    
    # This is a simplified MLP trainer
    # In production, would need proper dataset loading and model architecture
    send_log("PyTorch MLP training - Basic implementation")
    send_log("Note: Full implementation requires dataset-specific preprocessing")
    
    # Placeholder implementation
    raise NotImplementedError("PyTorch MLP training requires dataset-specific implementation")


def train_lstm_nlp(config):
    """Train LSTM for NLP"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed")
    
    send_log("LSTM NLP training - Basic implementation")
    raise NotImplementedError("LSTM NLP training requires tokenization and dataset preparation")


def get_environment_metadata():
    """Get current Python environment version information"""
    import platform
    metadata = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    
    try:
        import numpy
        metadata["numpy_version"] = numpy.__version__
    except ImportError:
        metadata["numpy_version"] = None
    
    try:
        import sklearn
        metadata["sklearn_version"] = sklearn.__version__
    except ImportError:
        metadata["sklearn_version"] = None
    
    return metadata


def create_model_metadata(config, model_type, framework, algorithm):
    """Create metadata.json content for model artifact"""
    env_meta = get_environment_metadata()
    
    metadata = {
        "model_type": model_type,
        "framework": framework,
        "algorithm": algorithm,
        "trained_at": datetime.now().isoformat(),
        "python_version": env_meta["python_version"],
        "python_implementation": env_meta["python_implementation"],
        "numpy_version": env_meta["numpy_version"],
        "sklearn_version": env_meta["sklearn_version"],
    }
    
    return metadata


def is_label_like_column_name_only(column_name):
    """Check if column name looks like a label/target (STRICT name matching only)
    
    Returns: dict with 'is_label_like' (bool)
    """
    if not column_name or not isinstance(column_name, str):
        return {'is_label_like': False}
    
    # STRICT label/target tokens only
    strict_label_tokens = [
        'label', 'target', 'class', 'outcome', 'y',
        'bought', 'purchase', 'purchased', 'churn', 'clicked',
        'converted', 'conversion', 'fraud', 'default', 'response'
    ]
    
    lower_name = column_name.lower().strip()
    
    # Check for exact match (for short tokens like "y")
    if lower_name in ['y', 'label', 'target', 'class']:
        return {'is_label_like': True}
    
    # For longer tokens, check for substring match but only in specific contexts
    for pattern in strict_label_tokens:
        if len(pattern) <= 2:
            continue  # Skip short patterns (already handled above)
        
        # Exact match
        if lower_name == pattern:
            return {'is_label_like': True}
        
        # Starts with pattern_
        if lower_name.startswith(pattern + '_'):
            return {'is_label_like': True}
        
        # Ends with _pattern
        if lower_name.endswith('_' + pattern):
            return {'is_label_like': True}
        
        # Contains pattern as whole word (with word boundaries)
        import re
        word_boundary_pattern = r'\b' + re.escape(pattern) + r'\b'
        if re.search(word_boundary_pattern, lower_name, re.IGNORECASE):
            return {'is_label_like': True}
    
    return {'is_label_like': False}


def get_label_like_confidence(column_name, unique_value_count=None):
    """Get label-like confidence based on name match + unique count
    
    Returns: None if not label-like, or dict with 'is_label_like' (bool) and 'confidence' ('high'|'medium')
    """
    # First check if name is label-like (strict detection)
    name_result = is_label_like_column_name_only(column_name)
    if not name_result['is_label_like']:
        return None  # Not label-like, no warning
    
    # If label-like, determine confidence based on unique count
    if unique_value_count is not None and unique_value_count <= 2:
        return {'is_label_like': True, 'confidence': 'high'}
    else:
        return {'is_label_like': True, 'confidence': 'medium'}


def is_label_like_column(column_name, unique_value_count=None):
    """Check if column name looks like a label/target (STRICT matching)
    
    Returns: dict with 'is_label_like' (bool) and 'confidence' ('high'|'medium'|None)
    DEPRECATED: Use get_label_like_confidence() instead for better confidence calculation
    """
    result = get_label_like_confidence(column_name, unique_value_count)
    if result is None:
        return {'is_label_like': False, 'confidence': None}
    return result


def create_model_schema(config, X_columns, categorical_cols=None, encoding_type="one_hot"):
    """Create schema.json content for model artifact with validation"""
    target_column = config.get("target_column")
    feature_columns = list(X_columns) if hasattr(X_columns, '__iter__') and not isinstance(X_columns, str) else []
    
    # HARD INVARIANT: target_column must NOT be in feature_columns
    if target_column and target_column in feature_columns:
        raise ValueError(f"CRITICAL: Target column '{target_column}' cannot be in feature_columns. This would cause data leakage.")
    
    # Check for label-like features (warning, not error) - STRICT detection
    warnings = []
    label_like_features = []
    
    # Get unique counts for columns if available (from config)
    column_unique_counts = config.get('column_stats', {})
    
    for col in feature_columns:
        unique_count = column_unique_counts.get(col) if column_unique_counts else None
        result = get_label_like_confidence(col, unique_count)
        
        if result and result['is_label_like']:
            confidence = result.get('confidence') or 'medium'
            unique_info = f", unique={unique_count}" if unique_count is not None else ""
            label_like_features.append({
                "column": col,
                "confidence": confidence,
                "unique_count": unique_count
            })
            send_log(f"[Tabular] label-like feature: {col} (confidence={confidence}{unique_info})", 'warning')
    
    if label_like_features:
        # Group by confidence
        all_columns = [f['column'] for f in label_like_features]
        high_confidence_cols = [f['column'] for f in label_like_features if f['confidence'] == 'high']
        overall_confidence = 'high' if high_confidence_cols else 'medium'
        
        warnings.append({
            "type": "label_like_feature",
            "columns": all_columns,
            "confidence": overall_confidence,
            "message": f"Feature(s) {all_columns} look like label/target columns and may cause leakage or confusion."
        })
    
    schema = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "dropped_columns": config.get("dropped_columns", []),
        "categorical_columns": list(categorical_cols) if categorical_cols else [],
        "encoding_type": encoding_type,
    }
    
    # Add warnings field if any warnings exist
    if warnings:
        schema["warnings"] = warnings
    
    return schema


def save_model_artifact(model, scaler, output_dir, model_format, metadata, schema, model_name="model"):
    """Save model as artifact structure: model_artifact/ with model file, metadata.json, schema.json"""
    # Create artifact folder
    artifact_dir = Path(output_dir) / f"{model_name}_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model file
    if model_format == "pkl":
        model_path = artifact_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
    elif model_format == "joblib":
        model_path = artifact_dir / "model.joblib"
        joblib.dump({"model": model, "scaler": scaler}, model_path)
    else:
        raise ValueError(f"Unsupported format: {model_format}")
    
    # Save metadata.json
    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save schema.json
    schema_path = artifact_dir / "schema.json"
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    send_log(f"Model artifact saved to: {artifact_dir}")
    send_log(f"  - Model: {model_path.name}")
    send_log(f"  - Metadata: {metadata_path.name}")
    send_log(f"  - Schema: {schema_path.name}")
    
    return artifact_dir


def validate_model_artifact(artifact_path):
    """Validate model artifact compatibility with current environment"""
    artifact_path = Path(artifact_path)
    metadata_path = artifact_path / "metadata.json"
    
    if not metadata_path.exists():
        return {
            "valid": False,
            "error": "metadata.json not found in artifact",
            "can_load": False
        }
    
    try:
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to read metadata.json: {str(e)}",
            "can_load": False
        }
    
    current_meta = get_environment_metadata()
    issues = []
    warnings = []
    can_load = True
    
    # Check Python major version
    saved_py_major = int(saved_metadata.get("python_version", "0.0").split('.')[0])
    current_py_major = int(current_meta["python_version"].split('.')[0])
    
    if saved_py_major != current_py_major:
        issues.append(f"Python major version mismatch: saved={saved_metadata.get('python_version')}, current={current_meta['python_version']}")
        can_load = False
    
    # Check scikit-learn major version
    if saved_metadata.get("sklearn_version") and current_meta.get("sklearn_version"):
        saved_sklearn_major = int(saved_metadata["sklearn_version"].split('.')[0])
        current_sklearn_major = int(current_meta["sklearn_version"].split('.')[0])
        
        if saved_sklearn_major != current_sklearn_major:
            issues.append(f"scikit-learn major version mismatch: saved={saved_metadata['sklearn_version']}, current={current_meta['sklearn_version']}")
            can_load = False
        elif saved_metadata["sklearn_version"] != current_meta["sklearn_version"]:
            warnings.append(f"scikit-learn minor version difference: saved={saved_metadata['sklearn_version']}, current={current_meta['sklearn_version']}")
    
    # Check NumPy major version
    if saved_metadata.get("numpy_version") and current_meta.get("numpy_version"):
        saved_numpy_major = int(saved_metadata["numpy_version"].split('.')[0])
        current_numpy_major = int(current_meta["numpy_version"].split('.')[0])
        
        if saved_numpy_major != current_numpy_major:
            warnings.append(f"NumPy major version difference: saved={saved_metadata['numpy_version']}, current={current_meta['numpy_version']}")
        elif saved_metadata["numpy_version"] != current_meta["numpy_version"]:
            warnings.append(f"NumPy minor version difference: saved={saved_metadata['numpy_version']}, current={current_meta['numpy_version']}")
    
    return {
        "valid": len(issues) == 0,
        "can_load": can_load,
        "issues": issues,
        "warnings": warnings,
        "saved_metadata": saved_metadata,
        "current_metadata": current_meta
    }


def load_tabular_dataset(data_dir, dataset_file=None, target_column=None, feature_columns=None):
    """Load tabular dataset from CSV file(s) with explicit column selection
    
    Args:
        data_dir: Directory path (fallback if dataset_file not provided)
        dataset_file: Absolute path to specific CSV file (takes priority)
        target_column: Explicit target column name (required if feature_columns provided)
        feature_columns: Explicit list of feature column names (if None, uses all except target)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not installed. Install with: pip install pandas")
    
    # Priority: use dataset_file if provided
    if dataset_file and Path(dataset_file).exists():
        csv_file = Path(dataset_file)
        send_log(f"Using specified dataset_file: {csv_file}")
        send_log(f"Resolved dataset used: {csv_file}")
    else:
        # Fallback: search directory
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}. Tabular training requires CSV files.")
        
        # Use first CSV file found
        csv_file = csv_files[0]
        if len(csv_files) > 1:
            send_log(f"Multiple CSV files found. Using: {csv_file.name}")
        
        send_log(f"Resolved dataset used: {csv_file}")
    
    send_log(f"Loading tabular data from {csv_file.name}...")
    df = pd.read_csv(csv_file)
    
    send_log(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Use explicit target column if provided, otherwise infer
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {list(df.columns)}")
        target_col = target_column
        send_log(f"Using explicit target column: {target_col}")
    else:
        # Fallback: assume last column is target
        target_col = df.columns[-1]
        send_log(f"Using inferred target column: {target_col} (last column)")
    
    # Use explicit feature columns if provided, otherwise use all except target
    if feature_columns:
        # Validate all feature columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found in CSV: {missing_cols}. Available columns: {list(df.columns)}")
        
        # Ensure target is not in feature list
        feature_cols = [col for col in feature_columns if col != target_col]
        if len(feature_cols) != len(feature_columns):
            send_log(f"Removed target column from feature list", 'warning')
        
        X = df[feature_cols].copy()
        send_log(f"Using explicit feature columns: {len(feature_cols)} columns")
        send_log(f"Feature columns: {', '.join(feature_cols)}")
    else:
        # Use all columns except target
        X = df.drop(columns=[target_col])
        send_log(f"Using all columns except target: {len(X.columns)} features")
    
    y = df[target_col]
    
    # Convert to numpy (keep as DataFrame for now to handle categorical encoding properly)
    # We'll encode after train/test split to prevent data leakage
    send_log(f"Features shape: {X.shape}, Target shape: {y.shape}")
    send_log(f"Target column: {target_col}")
    
    # Check for potential data leakage (ID columns, duplicate columns, etc.)
    suspected_id_columns = []
    
    # Check if any feature column is identical to target (would cause perfect accuracy)
    for col in X.columns:
        if X[col].equals(y):
            send_log(f"WARNING: Feature '{col}' is identical to target - this will cause data leakage!", 'warning')
            suspected_id_columns.append(col)
    
    # Check for ID-like columns (high cardinality, unique values)
    for col in X.columns:
        unique_ratio = X[col].nunique() / len(X)
        if unique_ratio > 0.95:  # More than 95% unique values
            suspected_id_columns.append(col)
            send_log(f"WARNING: Column '{col}' has {X[col].nunique()}/{len(X)} unique values ({unique_ratio*100:.1f}%) - may be an ID column", 'warning')
            send_log(f"Potential ID column detected: {col} ({unique_ratio*100:.1f}% unique)", 'warning')
    
    # Store suspected ID columns for potential exclusion
    # Return as DataFrame to handle encoding after split (prevents data leakage)
    return X, y, suspected_id_columns


def train_tabular_sklearn(config):
    """Train scikit-learn model for tabular data"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
    
    data_dir = config["data_dir"]
    dataset_file = config.get("dataset_file")  # Get dataset_file if provided
    variant = config["variant"]
    output_dir = config["output_dir"]
    validation_split = config.get("validation_split", 0.2)
    model_format = config.get("format", "pkl")
    
    # Log effective settings (only the ones that actually apply to this model)
    effective_settings = []
    if "n_estimators" in config:
        effective_settings.append(f"n_estimators={config['n_estimators']}")
    if "max_depth" in config:
        effective_settings.append(f"max_depth={config.get('max_depth') or 'None'}")
    if "min_samples_split" in config:
        effective_settings.append(f"min_samples_split={config['min_samples_split']}")
    if "min_samples_leaf" in config:
        effective_settings.append(f"min_samples_leaf={config['min_samples_leaf']}")
    if "max_features" in config:
        effective_settings.append(f"max_features={config.get('max_features') or 'None'}")
    if "validation_split" in config:
        effective_settings.append(f"validation_split={config['validation_split']}")
    
    if effective_settings:
        send_log(f"Effective settings: {', '.join(effective_settings)}", 'log')
    
    # Log dataset selection
    send_log(f"Selected dataset_file: {dataset_file if dataset_file else 'None'}")
    
    # Get explicit column selections from config (if provided)
    target_column = config.get("target_column")
    feature_columns = config.get("feature_columns")
    
    # Load tabular dataset with explicit column selection
    # Returns X, y as DataFrames (not encoded yet to prevent data leakage)
    X, y, suspected_id_columns = load_tabular_dataset(
        data_dir, 
        dataset_file, 
        target_column=target_column,
        feature_columns=feature_columns
    )
    
    send_log(f"Dataset loaded: {len(X)} samples, {len(X.columns)} features")
    
    # Calculate unique counts from FULL dataset BEFORE split (for confidence calculation)
    # This gives us accurate binary detection
    column_stats = {}
    for col in X.columns:
        unique_count = X[col].nunique()
        column_stats[col] = unique_count
    send_log(f"Calculated unique counts for confidence: {column_stats}", 'log')
    
    # Note: If explicit feature_columns were provided, suspected_id_columns are already excluded
    # Only log warnings about remaining suspicious columns
    if suspected_id_columns:
        remaining_suspicious = [col for col in suspected_id_columns if col in X.columns]
        if remaining_suspicious:
            send_log(f"WARNING: {len(remaining_suspicious)} suspicious columns still in features: {', '.join(remaining_suspicious)}", 'warning')
    
    # Determine if classification or regression based on target column type BEFORE split
    # Check the original y before encoding to get accurate type information
    target_col_name = config.get("target_column")
    is_classification = None
    
    if target_col_name and hasattr(y, 'dtype'):
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(y):
            unique_count = y.nunique()
            total_count = len(y)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # If numeric with many unique values (>20) or high uniqueness ratio (>0.5), it's regression
            if unique_count > 20 or unique_ratio > 0.5:
                is_classification = False
                send_log(f"Target '{target_col_name}' is numeric with {unique_count} unique values ({unique_ratio*100:.1f}%) - using regression", 'log')
            else:
                # Numeric but few unique values - could be classification (e.g., rating 1-5)
                is_classification = True
                send_log(f"Target '{target_col_name}' is numeric with {unique_count} unique values - using classification", 'log')
        else:
            # Categorical/string target - always classification
            is_classification = True
            send_log(f"Target '{target_col_name}' is categorical - using classification", 'log')
    else:
        # Fallback heuristic if we can't determine from column name
        unique_count = len(np.unique(y))
        is_classification = unique_count < 20
        send_log(f"Target has {unique_count} unique values - using {'classification' if is_classification else 'regression'}", 'log')
    
    # SPLIT FIRST to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y if is_classification and len(np.unique(y)) < 20 else None
    )
    send_log(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Capture original feature columns and categorical info BEFORE encoding (for schema)
    original_feature_cols = feature_columns if feature_columns else (list(X_train.columns) if hasattr(X_train, 'columns') else [])
    categorical_cols = X_train.select_dtypes(include=['object']).columns if hasattr(X_train, 'select_dtypes') else []
    categorical_cols_list = list(categorical_cols) if len(categorical_cols) > 0 else []
    encoding_type = "one_hot" if len(categorical_cols_list) > 0 else "none"
    
    # Store original X for later use (before encoding)
    X_original = X_train.copy() if hasattr(X_train, 'copy') else X_train
    
    # Handle categorical features AFTER split (fit on train, transform test)
    if len(categorical_cols) > 0:
        send_log(f"Found {len(categorical_cols)} categorical columns. Encoding (fit on train only)...")
        # Get dummies on train
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        # Get dummies on test (may have different columns, so align)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        # Align columns (add missing columns with zeros)
        X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
        X_train = X_train_encoded
        X_test = X_test_encoded
        send_log(f"After encoding: {len(X_train.columns)} features")
    
    # Encode target if string (do this after split too)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        send_log("Encoded string labels to integers")
    
    # Capture encoded feature columns (post-encoding) for schema alignment in inference
    encoded_feature_columns = list(X_train.columns) if hasattr(X_train, 'columns') else []

    # Keep DataFrames for scaler fit to preserve feature names
    X_train_df = X_train.copy() if hasattr(X_train, 'copy') else X_train
    X_test_df = X_test.copy() if hasattr(X_test, 'copy') else X_test
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Scale features (fit on train, transform test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    
    # Create model based on variant
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    
    # is_classification was already determined before the split (see above)
    # Use that value here
    
    # Get RandomForest parameters from config if provided
    n_estimators = config.get("n_estimators", 100)
    max_depth = config.get("max_depth")
    min_samples_split = config.get("min_samples_split", 2)
    min_samples_leaf = config.get("min_samples_leaf", 1)
    max_features = config.get("max_features", "sqrt")
    
    # Build RandomForest with custom parameters
    rf_params = {
        "n_estimators": n_estimators,
        "random_state": 42
    }
    if max_depth is not None:
        rf_params["max_depth"] = max_depth
    rf_params["min_samples_split"] = min_samples_split
    rf_params["min_samples_leaf"] = min_samples_leaf
    if max_features is not None:
        rf_params["max_features"] = max_features
    
    model_map = {
        "random_forest": RandomForestClassifier(**rf_params) if is_classification 
                       else RandomForestRegressor(**rf_params),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42) if is_classification
                            else GradientBoostingRegressor(n_estimators=100, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42) if is_classification
                      else ExtraTreesRegressor(n_estimators=100, random_state=42),
        "svm": SVC(probability=True, random_state=42) if is_classification else None,
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000) if is_classification else None,
        "linear_regression": LinearRegression() if not is_classification else None
    }
    
    model = model_map.get(variant)
    if model is None:
        # Default fallback
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestClassifier(n_estimators=100, random_state=42) if is_classification else RandomForestRegressor(n_estimators=100, random_state=42)
    
    task_type = "classification" if is_classification else "regression"
    send_log(f"Training {variant} model for {task_type}...")
    
    # Simulate training progress
    total_epochs = 1  # sklearn models train in one go, but we'll show progress
    send_progress(0, total_epochs, None, None, "training")
    
    model.fit(X_train, y_train)
    
    send_progress(total_epochs, total_epochs, None, None, "training")
    
    # Evaluate on TEST SET ONLY (explicit to prevent confusion)
    if is_classification:
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        send_log(f"Test set accuracy: {accuracy:.4f} (evaluated on {len(X_test)} test samples)")
        
        # Also check training accuracy for comparison (but don't use it as the metric)
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        send_log(f"Train set accuracy: {train_accuracy:.4f} (for reference only)")
        
        if accuracy == 1.0:
            send_log("WARNING: Perfect test accuracy detected. This may indicate:", 'warning')
            send_log("  - Data leakage (target column in features, ID column, etc.)", 'warning')
            send_log("  - Overfitting (check train vs test accuracy)", 'warning')
            send_log("  - Trivial dataset (very few samples or perfect separation)", 'warning')
        
        # Send final progress update with accuracy for UI sync
        send_progress(total_epochs, total_epochs, None, float(accuracy), "completed")
        
        metrics = {"accuracy": float(accuracy * 100), "loss": None}
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        send_log(f"Test set R² score: {r2:.4f}, MSE: {mse:.4f} (evaluated on {len(X_test)} test samples)")
        
        # Send final progress update with loss (MSE) for UI sync
        send_progress(total_epochs, total_epochs, float(mse), None, "completed")
        
        metrics = {"r2_score": float(r2 * 100), "mse": float(mse), "loss": float(mse)}
    
    # Save model as artifact structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata
    # Use "tabular" as model_type for tabular models (not "sklearn_random_forest")
    metadata = create_model_metadata(config, "tabular", "sklearn", variant)
    
    # Add column stats to config for schema creation (already calculated before split)
    config_with_stats = config.copy()
    config_with_stats['column_stats'] = column_stats
    send_log(f"Using column stats for confidence calculation: {column_stats}", 'log')
    
    # Create schema (using captured info from before encoding)
    schema = create_model_schema(config_with_stats, original_feature_cols, categorical_cols_list, encoding_type)
    schema["encoded_feature_columns"] = encoded_feature_columns
    
    # Save as artifact
    artifact_dir = save_model_artifact(model, scaler, output_dir, model_format, metadata, schema)
    
    # Return artifact directory path (for compatibility)
    model_file = artifact_dir / f"model.{model_format}"
    
    send_log(f"Model exported (Python {metadata['python_version']} / scikit-learn {metadata['sklearn_version']})")
    send_log(f"This format requires a compatible Python environment.")
    
    return artifact_dir, metrics


def train_tabular_xgboost(config):
    """Train XGBoost model for tabular data"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost not installed. Install with: pip install xgboost")
    
    data_dir = config["data_dir"]
    dataset_file = config.get("dataset_file")  # Get dataset_file if provided
    variant = config["variant"]
    output_dir = config["output_dir"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    validation_split = config.get("validation_split", 0.2)
    model_format = config.get("format", "json")
    
    # Log dataset selection
    send_log(f"Selected dataset_file: {dataset_file if dataset_file else 'None'}")
    
    # Load tabular dataset (dataset_file takes priority)
    X, y, _ = load_tabular_dataset(data_dir, dataset_file)
    
    # Determine task type
    is_classification = len(np.unique(y)) < 20
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train model
    if is_classification:
        num_class = len(np.unique(y)) if len(np.unique(y)) > 2 else None
        objective = "multi:softprob" if num_class else "binary:logistic"
        eval_metric = "mlogloss" if num_class else "logloss"
    else:
        objective = "reg:squarederror"
        eval_metric = "rmse"
        num_class = None
    
    params = {
        "objective": objective,
        "learning_rate": learning_rate,
        "max_depth": 6,
        "eval_metric": eval_metric
    }
    if num_class:
        params["num_class"] = num_class
    
    send_log(f"Training XGBoost model for {epochs} epochs ({'classification' if is_classification else 'regression'})...")
    
    # Train model (XGBoost trains all at once, but we'll send progress updates)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=epochs,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False
    )
    
    # Send completion progress
    send_progress(epochs, epochs, None, None, "training")
    
    # Final evaluation
    predictions = model.predict(dtest)
    if is_classification:
        if num_class:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions == y_test)
        send_log(f"Model accuracy: {accuracy:.4f}")
        # Send final progress update with accuracy for UI sync
        send_progress(epochs, epochs, None, float(accuracy), "completed")
        metrics = {"accuracy": float(accuracy * 100), "loss": None}
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        send_log(f"Model R² score: {r2:.4f}, MSE: {mse:.4f}")
        # Send final progress update with loss (MSE) for UI sync
        send_progress(epochs, epochs, float(mse), None, "completed")
        metrics = {"r2_score": float(r2 * 100), "mse": float(mse), "loss": float(mse)}
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    if model_format == "json":
        model_path = Path(output_dir) / "model.json"
        model.save_model(str(model_path))
    elif model_format == "ubj":
        model_path = Path(output_dir) / "model.ubj"
        model.save_model(str(model_path))
    else:
        raise ValueError(f"Unsupported format for xgboost: {model_format}")
    
    return model_path, metrics


def train_tabular_lightgbm(config):
    """Train LightGBM model for tabular data"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm not installed. Install with: pip install lightgbm")
    
    data_dir = config["data_dir"]
    dataset_file = config.get("dataset_file")  # Get dataset_file if provided
    variant = config["variant"]
    output_dir = config["output_dir"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    validation_split = config.get("validation_split", 0.2)
    model_format = config.get("format", "txt")
    
    # Log dataset selection
    send_log(f"Selected dataset_file: {dataset_file if dataset_file else 'None'}")
    
    # Load tabular dataset (dataset_file takes priority)
    X, y, _ = load_tabular_dataset(data_dir, dataset_file)
    
    # Determine task type
    is_classification = len(np.unique(y)) < 20
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Set parameters
    if is_classification:
        num_class = len(np.unique(y)) if len(np.unique(y)) > 2 else None
        objective = "multiclass" if num_class else "binary"
        metric = "multi_logloss" if num_class else "binary_logloss"
    else:
        objective = "regression"
        metric = "rmse"
        num_class = None
    
    params = {
        "objective": objective,
        "metric": metric,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "verbose": -1
    }
    if num_class:
        params["num_class"] = num_class
    
    send_log(f"Training LightGBM model for {epochs} epochs ({'classification' if is_classification else 'regression'})...")
    
    # Train model (LightGBM trains all at once, but we'll send progress updates)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=epochs,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[lgb.log_evaluation(period=0)]  # Suppress default logging
    )
    
    # Send completion progress
    send_progress(epochs, epochs, None, None, "training")
    
    # Evaluate
    predictions = model.predict(X_test)
    if is_classification:
        if num_class:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions == y_test)
        send_log(f"Model accuracy: {accuracy:.4f}")
        # Send final progress update with accuracy for UI sync
        send_progress(epochs, epochs, None, float(accuracy), "completed")
        metrics = {"accuracy": float(accuracy * 100), "loss": None}
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        send_log(f"Model R² score: {r2:.4f}, MSE: {mse:.4f}")
        # Send final progress update with loss (MSE) for UI sync
        send_progress(epochs, epochs, float(mse), None, "completed")
        metrics = {"r2_score": float(r2 * 100), "mse": float(mse), "loss": float(mse)}
        metrics = {"r2_score": float(r2 * 100), "mse": float(mse), "loss": float(mse)}
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    if model_format == "txt":
        model_path = Path(output_dir) / "model.txt"
        model.save_model(str(model_path))
    else:
        raise ValueError(f"Unsupported format for lightgbm: {model_format}")
    
    return model_path, metrics


def train_tabular_pytorch(config):
    """Train PyTorch MLP for tabular data"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Install with: pip install torch")
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not installed. Install with: pip install pandas")
    
    data_dir = config["data_dir"]
    dataset_file = config.get("dataset_file")  # Get dataset_file if provided
    variant = config["variant"]
    output_dir = config["output_dir"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    validation_split = config.get("validation_split", 0.2)
    model_format = config.get("format", "pth")
    
    # Log dataset selection
    send_log(f"Selected dataset_file: {dataset_file if dataset_file else 'None'}")
    
    # Load tabular dataset (dataset_file takes priority)
    X, y, _ = load_tabular_dataset(data_dir, dataset_file)
    
    # Determine task type
    is_classification = len(np.unique(y)) < 20
    num_classes = len(np.unique(y)) if is_classification else 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train) if is_classification else torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test) if is_classification else torch.FloatTensor(y_test)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define model architecture based on variant
    input_size = X_train.shape[1]
    layer_sizes = {
        "mlp_small": [input_size, 64, num_classes if is_classification else 1],
        "mlp_medium": [input_size, 128, 64, num_classes if is_classification else 1],
        "mlp_large": [input_size, 256, 128, 64, num_classes if is_classification else 1],
        "mlp_deep": [input_size, 512, 256, 128, 64, num_classes if is_classification else 1]
    }
    
    sizes = layer_sizes.get(variant, layer_sizes["mlp_medium"])
    
    # Build model
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:  # Don't add activation after last layer
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
    
    model = nn.Sequential(*layers)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    send_log(f"Training PyTorch MLP ({variant}) for {epochs} epochs ({'classification' if is_classification else 'regression'})...")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if is_classification:
                loss = criterion(outputs, batch_y)
            else:
                outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            if is_classification:
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == y_test_tensor).float().mean().item()
                send_progress(epoch + 1, epochs, float(avg_loss), float(accuracy * 100), "training")
            else:
                test_outputs = test_outputs.squeeze()
                test_loss = criterion(test_outputs, y_test_tensor).item()
                send_progress(epoch + 1, epochs, float(test_loss), None, "training")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        if is_classification:
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean().item()
            send_log(f"Model accuracy: {accuracy:.4f}")
            metrics = {"accuracy": float(accuracy * 100), "loss": float(avg_loss)}
        else:
            test_outputs = test_outputs.squeeze()
            from sklearn.metrics import r2_score
            predictions_np = test_outputs.numpy()
            r2 = r2_score(y_test, predictions_np)
            send_log(f"Model R² score: {r2:.4f}")
            metrics = {"r2_score": float(r2 * 100), "loss": float(test_loss)}
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    if model_format == "pth":
        model_path = Path(output_dir) / "model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "input_size": input_size,
            "is_classification": is_classification,
            "num_classes": num_classes
        }, model_path)
    elif model_format == "pt":
        model_path = Path(output_dir) / "model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "input_size": input_size,
            "is_classification": is_classification,
            "num_classes": num_classes
        }, model_path)
    else:
        raise ValueError(f"Unsupported format for pytorch: {model_format}")
    
    return model_path, metrics


def main():
    """Main training function - reads config from stdin and trains model"""
    try:
        # Read configuration from stdin
        config_str = sys.stdin.read()
        config = json.loads(config_str)
        
        model_purpose = config["model_purpose"]
        framework = config.get("framework", "")
        variant = config.get("variant", "")
        data_dir = config["data_dir"]
        output_dir = config["output_dir"]
        epochs = config.get("epochs", 10)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 0.001)
        model_format = config.get("format", "pt")
        
        # Validate required fields
        if not framework:
            raise ValueError(f"Framework not selected. Please select a framework for {model_purpose} training.")
        if not variant:
            raise ValueError(f"Model variant not selected. Please select a model variant.")
        
        send_log(f"Starting training: {model_purpose}/{framework}/{variant}")
        send_log(f"Data directory: {data_dir}")
        send_log(f"Output directory: {output_dir}")
        
        # Route to appropriate trainer
        if model_purpose == "computer_vision" and framework == "yolo":
            model_path, metrics = train_yolo_model(config)
        elif model_purpose == "machine_learning" and framework == "sklearn":
            model_path, metrics = train_sklearn_model(config)
        elif model_purpose == "machine_learning" and framework == "xgboost":
            model_path, metrics = train_xgboost_model(config)
        elif model_purpose == "machine_learning" and framework == "lightgbm":
            send_log("LightGBM training - Not yet implemented")
            raise NotImplementedError("LightGBM training not yet implemented")
        elif model_purpose == "tabular" and framework == "sklearn":
            model_path, metrics = train_tabular_sklearn(config)
        elif model_purpose == "tabular" and framework == "xgboost":
            model_path, metrics = train_tabular_xgboost(config)
        elif model_purpose == "tabular" and framework == "lightgbm":
            model_path, metrics = train_tabular_lightgbm(config)
        elif model_purpose == "tabular" and framework == "pytorch":
            model_path, metrics = train_tabular_pytorch(config)
        else:
            send_log(f"Training not yet implemented for {model_purpose}/{framework}")
            raise NotImplementedError(f"Training for {model_purpose}/{framework}/{variant} not yet implemented")
        
        send_result(str(model_path), metrics, model_format)
        send_log("Training completed successfully!")
        
    except Exception as e:
        send_error(str(e), traceback.format_exc())
        sys.exit(1)


def infer_cv_yolo(model_path, image_path, confidence=0.25, output_dir=None):
    """Run YOLO inference on a single image
    
    Args:
        model_path: Path to model file (best.pt or model artifact)
        image_path: Path to input image
        confidence: Confidence threshold (0-1)
        output_dir: Directory to save results (default: creates inference_output folder)
    
    Returns:
        dict with detections, output_path, and preview_path
    """
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
    
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow not installed. Install with: pip install pillow")
    
    model_path = Path(model_path)
    image_path = Path(image_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path.cwd() / "inference_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    send_log(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    send_log(f"Running inference on: {image_path.name}")
    send_log(f"Confidence threshold: {confidence}")
    
    # Run prediction
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        save=True,
        project=str(output_dir),
        name="predict",
        exist_ok=True
    )
    
    # Get output image path
    predict_dir = output_dir / "predict"
    if results and len(results) > 0 and hasattr(results[0], 'save_dir'):
        try:
            predict_dir = Path(results[0].save_dir)
        except Exception:
            pass
    output_image_path = predict_dir / image_path.name

    # If image wasn't saved with original name, find the saved image
    if not output_image_path.exists():
        stem = image_path.stem
        candidates = list(predict_dir.glob(f"{stem}.*"))
        if candidates:
            output_image_path = candidates[0]
        else:
            saved_images = (
                list(predict_dir.glob("*.jpg")) +
                list(predict_dir.glob("*.jpeg")) +
                list(predict_dir.glob("*.png")) +
                list(predict_dir.glob("*.bmp")) +
                list(predict_dir.glob("*.webp"))
            )
            if saved_images:
                # pick most recent
                saved_images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                output_image_path = saved_images[0]

    if not output_image_path.exists():
        send_log(f"Warning: output image not found in {predict_dir}", "warning")
    
    # Extract detections
    detections = []
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i]) if hasattr(boxes, 'cls') else 0
                conf = float(boxes.conf[i]) if hasattr(boxes, 'conf') else 0.0
                class_name = model.names[cls] if hasattr(model, 'names') and cls in model.names else f"class_{cls}"
                
                # Get bounding box coordinates
                if hasattr(boxes, 'xyxy'):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                else:
                    x1, y1, x2, y2 = 0, 0, 0, 0
                
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]
                })
    
    send_log(f"Found {len(detections)} detections")
    
    return {
        "detections": detections,
        "output_dir": str(output_dir),
        "output_image_path": str(output_image_path),
        "num_detections": len(detections)
    }


def infer_tabular(model_artifact_path, csv_path, output_dir=None):
    """Run tabular model inference on CSV data
    
    Args:
        model_artifact_path: Path to model artifact folder (contains model.joblib, schema.json, metadata.json)
        csv_path: Path to input CSV file
        output_dir: Directory to save predictions (default: creates predictions_output folder)
    
    Returns:
        dict with predictions, probabilities (if available), and output_path
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not installed. Install with: pip install pandas")
    
    artifact_path = Path(model_artifact_path)
    csv_path = Path(csv_path)
    
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Validate artifact
    validation = validate_model_artifact(artifact_path)
    if not validation["can_load"]:
        error_msg = "Model artifact validation failed:\n" + "\n".join(validation.get("issues", []))
        raise ValueError(error_msg)
    
    if validation.get("warnings"):
        for warning in validation["warnings"]:
            send_log(f"Warning: {warning}", "warning")
    
    # Load schema
    schema_path = artifact_path / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.json not found in artifact: {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    target_column = schema.get("target_column")
    feature_columns = schema.get("feature_columns", [])
    categorical_cols = schema.get("categorical_columns", [])
    encoding_type = schema.get("encoding_type", "one_hot")
    encoded_feature_columns = schema.get("encoded_feature_columns")
    
    send_log(f"Loaded schema: {len(feature_columns)} features, target: {target_column}")
    
    # Load model - support both .pkl and .joblib formats
    model_path = None
    model_format = None
    
    # Try joblib first, then pkl
    if (artifact_path / "model.joblib").exists():
        model_path = artifact_path / "model.joblib"
        model_format = "joblib"
    elif (artifact_path / "model.pkl").exists():
        model_path = artifact_path / "model.pkl"
        model_format = "pkl"
    
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model file not found in artifact (expected model.joblib or model.pkl)")
    
    send_log(f"Loading model: {model_path} (format: {model_format})")
    
    # Load model - support both .pkl and .joblib formats
    # The saved file contains {"model": model, "scaler": scaler} dict
    if model_format == "joblib":
        model_data = joblib.load(str(model_path))
    else:  # pkl format
        # joblib can load pickle files, use it for consistency
        model_data = joblib.load(str(model_path))
    
    # Extract model and scaler from the saved dict
    if isinstance(model_data, dict) and "model" in model_data:
        model = model_data["model"]
        scaler = model_data.get("scaler", None)
        if scaler:
            send_log("Loaded model and scaler from artifact")
        else:
            send_log("Loaded model from artifact (no scaler found)")
    else:
        # Fallback: assume it's just the model (for backwards compatibility)
        model = model_data
        scaler = None
        send_log("Loaded model from artifact (legacy format)")
    
    # Also check for separate scaler file (for backwards compatibility)
    scaler_path = artifact_path / "scaler.joblib"
    if scaler is None and scaler_path.exists():
        scaler = joblib.load(str(scaler_path))
        send_log("Loaded scaler from separate file")
    
    # Load label encoder if available
    label_encoder_path = artifact_path / "label_encoder.joblib"
    label_encoder = None
    if label_encoder_path.exists():
        label_encoder = joblib.load(str(label_encoder_path))
        send_log("Loaded label encoder from artifact")
    
    def align_columns(df_in, expected_cols, context_label):
        missing = [col for col in expected_cols if col not in df_in.columns]
        extra = [col for col in df_in.columns if col not in expected_cols]
        if missing:
            send_log(f"{context_label}: missing columns added with 0 default: {missing}", "warning")
            for col in missing:
                df_in[col] = 0
        if extra:
            send_log(f"{context_label}: extra columns dropped: {extra}", "warning")
            df_in = df_in.drop(columns=extra)
        return df_in[expected_cols].copy()

    # Load input CSV
    send_log(f"Loading input CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    send_log(f"Input shape: {df.shape[0]} rows, {df.shape[1]} columns")

    if not feature_columns:
        feature_columns = list(df.columns)
        send_log("Schema missing feature_columns; using CSV columns as-is", "warning")

    # Enforce schema column order (add missing with defaults, drop extra)
    X = align_columns(df, feature_columns, "Schema alignment")
    
    # Handle categorical encoding (match training preprocessing)
    if categorical_cols and encoding_type == "one_hot":
        # One-hot encode categorical columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        send_log("Applied one-hot encoding to categorical features")

    # Align to encoded columns from schema (most reliable)
    if encoded_feature_columns:
        X = align_columns(X, encoded_feature_columns, "Encoded alignment (schema)")
    # Align to trained scaler feature order if available
    elif scaler is not None and hasattr(scaler, "feature_names_in_"):
        expected_encoded_cols = list(scaler.feature_names_in_)
        X = align_columns(X, expected_encoded_cols, "Encoded alignment")
    
    # Scale features if scaler available
    if scaler:
        # Backward compatibility: old scalers were fit on numpy arrays (no feature names)
        if not hasattr(scaler, "feature_names_in_"):
            X = X.values  # convert DataFrame -> numpy to match old scaler
            send_log("Scaler has no feature_names_in_; converting X to numpy for compatibility", "warning")
        X = scaler.transform(X)
        send_log("Applied feature scaling")
    
    # Run prediction
    send_log("Running predictions...")
    predictions = model.predict(X)
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        send_log("Prediction probabilities available")
    
    # Decode labels if encoder available
    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions)
        send_log("Decoded predictions using label encoder")
    
    # Create results DataFrame
    results_df = df.copy()
    results_df[f"prediction_{target_column}"] = predictions
    
    if probabilities is not None:
        # Add probability columns
        if hasattr(model, "classes_"):
            class_names = model.classes_
            if label_encoder:
                class_names = label_encoder.inverse_transform(class_names)
            for i, class_name in enumerate(class_names):
                results_df[f"prob_{class_name}"] = probabilities[:, i]
        else:
            for i in range(probabilities.shape[1]):
                results_df[f"prob_class_{i}"] = probabilities[:, i]
    
    # Save predictions
    if output_dir is None:
        output_dir = Path.cwd() / "predictions_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / "predictions.csv"
    results_df.to_csv(output_csv, index=False)
    send_log(f"Saved predictions to: {output_csv}")
    
    # Convert to list of dicts for JSON response
    results_list = results_df.head(100).to_dict('records')  # Limit to 100 rows for preview
    
    return {
        "predictions": results_list,
        "output_path": str(output_csv),
        "num_predictions": len(results_df),
        "has_probabilities": probabilities is not None
    }


def main_inference():
    """Main inference function - reads config from stdin and runs inference"""
    try:
        # Read configuration from stdin
        config_str = sys.stdin.read()
        config = json.loads(config_str)
        
        inference_type = config.get("inference_type")  # "cv" or "tabular"
        
        if inference_type == "cv":
            model_path = config["model_path"]
            image_path = config["image_path"]
            confidence = config.get("confidence", 0.25)
            output_dir = config.get("output_dir")
            
            send_log(f"Starting CV inference: {image_path}")
            result = infer_cv_yolo(model_path, image_path, confidence, output_dir)
            # Send result with all inference data
            send_result(
                str(result["output_image_path"]), 
                {
                    "detections": result["detections"], 
                    "num_detections": result["num_detections"],
                    "output_dir": result["output_dir"],
                    "output_image_path": result["output_image_path"]
                }, 
                "json"
            )
            send_log("CV inference completed successfully!")
            
        elif inference_type == "tabular":
            model_artifact_path = config["model_artifact_path"]
            csv_path = config["csv_path"]
            output_dir = config.get("output_dir")
            
            send_log(f"Starting tabular inference: {csv_path}")
            result = infer_tabular(model_artifact_path, csv_path, output_dir)
            # Send result with all inference data
            send_result(
                str(result["output_path"]), 
                {
                    "num_predictions": result["num_predictions"], 
                    "has_probabilities": result["has_probabilities"],
                    "output_path": result["output_path"],
                    "predictions": result["predictions"]  # First 100 rows for preview
                }, 
                "json"
            )
            send_log("Tabular inference completed successfully!")
            
        else:
            raise ValueError(f"Unknown inference_type: {inference_type}. Must be 'cv' or 'tabular'")
            
    except Exception as e:
        send_error(str(e), traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Check if running in inference mode (via command line arg)
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        main_inference()
    else:
        main()
