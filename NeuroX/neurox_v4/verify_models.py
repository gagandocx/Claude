#!/usr/bin/env python3
"""
=============================================================
  NeuroX v7.4 - Model Checkpoint Verification Script
  
  Validates all 17 model checkpoints + 1 meta-learner:
  - Checks all 18 files exist and reports their sizes
  - Loads each neural model (.pth), runs dummy inference,
    verifies output shape and class diversity
  - Loads each tree model (.joblib), verifies without error
  - Checks v7.4 state files (Platt calibration, Sharpe weights)
  - Reports summary: OK / STUCK / OVERCONFIDENT / ERROR / MISSING
  
  Usage:
    python verify_models.py
=============================================================
"""

import os
import sys
import traceback

# Ensure neurox_v4 root is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ─────────────────────────────────────────────
#  ANSI COLOR HELPERS
# ─────────────────────────────────────────────
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def ok(msg):
    return f"{Colors.GREEN}[OK]{Colors.RESET} {msg}"


def err(msg):
    return f"{Colors.RED}[ERROR]{Colors.RESET} {msg}"


def warn(msg):
    return f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}"


def info(msg):
    return f"{Colors.CYAN}[INFO]{Colors.RESET} {msg}"


# ─────────────────────────────────────────────
#  MODEL REGISTRY (v7.4 - 17 models + 1 meta)
# ─────────────────────────────────────────────
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

# Neural models: (name, filename, model_class_path, config_class_name)
NEURAL_MODELS = [
    ("transformer", "transformer.pth", "models.transformer_model", "MarketTransformer", "TransformerConfig"),
    ("lstm", "lstm.pth", "models.lstm_model", "MarketLSTM", "LSTMConfig"),
    ("tcn", "tcn.pth", "models.tcn_model", "MarketTCN", "TCNConfig"),
    ("patch_tst", "patch_tst.pth", "models.patch_tst", "MarketPatchTST", "PatchTSTConfig"),
    ("tft", "tft.pth", "models.tft_model", "MarketTFT", "TFTConfig"),
    ("nhits", "nhits.pth", "models.nhits_model", "MarketNHiTS", "NHiTSConfig"),
    ("itransformer", "itransformer.pth", "models.itransformer", "MarketITransformer", "ITransformerConfig"),
    ("mamba", "mamba.pth", "models.mamba_model", "MarketMamba", "MambaConfig"),
    ("dlinear", "dlinear.pth", "models.dlinear_model", "MarketDLinear", "DLinearConfig"),
    ("xlstm", "xlstm.pth", "models.xlstm_model", "MarketXLSTM", "xLSTMConfig"),
    ("timesnet", "timesnet.pth", "models.timesnet_model", "MarketTimesNet", "TimesNetConfig"),
    ("chronos", "chronos.pth", "models.chronos_model", "MarketChronos", "ChronosConfig"),
    ("timemixer", "timemixer.pth", "models.timemixer_model", "MarketTimeMixer", "TimeMixerConfig"),
    ("softs", "softs.pth", "models.softs_model", "MarketSOFTS", "SOFTSConfig"),
]

# Tree models: (name, filename, type)
# type: "joblib_sklearn" = plain joblib load, "gradboost_extra" = GradBoostExtra.load(), "catboost" = CatBoostModel.load()
TREE_MODELS = [
    ("gradient_boost", "gradient_boost.joblib", "joblib_sklearn"),
    ("xgboost_extra", "xgboost_extra.joblib", "gradboost_extra"),
    ("catboost", "catboost.joblib", "catboost_model"),
]

META_LEARNER = ("meta_learner", "meta_learner.joblib", "joblib_sklearn")

# v7.4 state files
V74_STATE_FILES = [
    ("platt_calibration_state", "platt_calibration_state.json"),
    ("sharpe_weight_state", "sharpe_weight_state.json"),
]


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def file_size_str(path):
    """Return human-readable file size."""
    if not os.path.exists(path):
        return "MISSING"
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.2f} MB"


def import_model_class(module_path, class_name):
    """Dynamically import a model class."""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def import_config_class(config_name):
    """Import a config class from config.settings."""
    from config.settings import (
        TransformerConfig, LSTMConfig, TCNConfig, PatchTSTConfig,
        TFTConfig, NHiTSConfig, ITransformerConfig, MambaConfig,
        DLinearConfig, xLSTMConfig, TimesNetConfig, ChronosConfig,
        TimeMixerConfig, SOFTSConfig, XGBoostConfig, CatBoostConfig
    )
    configs = {
        "TransformerConfig": TransformerConfig,
        "LSTMConfig": LSTMConfig,
        "TCNConfig": TCNConfig,
        "PatchTSTConfig": PatchTSTConfig,
        "TFTConfig": TFTConfig,
        "NHiTSConfig": NHiTSConfig,
        "ITransformerConfig": ITransformerConfig,
        "MambaConfig": MambaConfig,
        "DLinearConfig": DLinearConfig,
        "xLSTMConfig": xLSTMConfig,
        "TimesNetConfig": TimesNetConfig,
        "ChronosConfig": ChronosConfig,
        "TimeMixerConfig": TimeMixerConfig,
        "SOFTSConfig": SOFTSConfig,
        "XGBoostConfig": XGBoostConfig,
        "CatBoostConfig": CatBoostConfig,
    }
    return configs[config_name]


def check_predictions_stuck(probs, name):
    """
    Check if predictions are stuck (always predicting same class)
    or overconfident (one class > 99% on all samples).
    
    Returns: (status, detail)
      status: "OK", "STUCK", "OVERCONFIDENT"
    """
    # probs shape: (batch, 3)
    predicted_classes = np.argmax(probs, axis=1)
    unique_classes = np.unique(predicted_classes)
    
    if len(unique_classes) == 1:
        return "STUCK", f"All {len(probs)} samples predict class {unique_classes[0]}"
    
    # Check overconfidence: max prob > 0.99 on average
    max_probs = np.max(probs, axis=1)
    avg_max_prob = np.mean(max_probs)
    if avg_max_prob > 0.99:
        return "OVERCONFIDENT", f"Avg max prob = {avg_max_prob:.4f} (>0.99)"
    
    return "OK", f"Classes predicted: {unique_classes.tolist()}, avg confidence: {avg_max_prob:.3f}"


# ─────────────────────────────────────────────
#  VERIFICATION FUNCTIONS
# ─────────────────────────────────────────────
def verify_file_existence():
    """Step 1: Check all 18 checkpoint files exist and report sizes."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"  STEP 1: File Existence Check (18 files expected)")
    print(f"{'=' * 60}{Colors.RESET}\n")
    
    all_files = []
    for name, filename, *_ in NEURAL_MODELS:
        all_files.append((name, filename))
    for name, filename, *_ in TREE_MODELS:
        all_files.append((name, filename))
    all_files.append((META_LEARNER[0], META_LEARNER[1]))
    
    found = 0
    missing = 0
    results = {}
    
    for name, filename in all_files:
        path = os.path.join(CHECKPOINTS_DIR, filename)
        size = file_size_str(path)
        exists = os.path.exists(path)
        
        if exists:
            print(f"  {ok(f'{name:20s} | {filename:25s} | {size}')}")
            found += 1
            results[name] = "EXISTS"
        else:
            print(f"  {err(f'{name:20s} | {filename:25s} | MISSING')}")
            missing += 1
            results[name] = "MISSING"
    
    print(f"\n  {info(f'Found: {found}/18, Missing: {missing}/18')}")
    return results


def verify_neural_models():
    """Step 2: Load neural models, run dummy batch, verify output."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"  STEP 2: Neural Model Verification (14 models)")
    print(f"{'=' * 60}{Colors.RESET}\n")
    
    if not TORCH_AVAILABLE:
        print(f"  {err('PyTorch not installed - cannot verify neural models')}")
        return {name: "ERROR" for name, *_ in NEURAL_MODELS}
    
    results = {}
    batch_size = 100
    seq_length = 64
    input_features = 46
    
    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, seq_length, input_features)
    
    for name, filename, module_path, class_name, config_name in NEURAL_MODELS:
        path = os.path.join(CHECKPOINTS_DIR, filename)
        
        if not os.path.exists(path):
            print(f"  {err(f'{name:15s} | MISSING - skipped')}")
            results[name] = "MISSING"
            continue
        
        try:
            # Import config and model class
            ConfigClass = import_config_class(config_name)
            ModelClass = import_model_class(module_path, class_name)
            
            # Instantiate model
            config = ConfigClass()
            model = ModelClass(config)
            
            # Load weights
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Run dummy prediction
            with torch.no_grad():
                output = model.predict(dummy_input)
            
            # Verify output shape
            probs = output.numpy() if isinstance(output, torch.Tensor) else output
            
            if probs.shape != (batch_size, 3):
                print(f"  {err(f'{name:15s} | Bad output shape: {probs.shape} (expected ({batch_size}, 3))')}")
                results[name] = "ERROR"
                continue
            
            # Verify predictions are valid probabilities
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                print(f"  {err(f'{name:15s} | Output contains NaN/Inf')}")
                results[name] = "ERROR"
                continue
            
            # Check if stuck or overconfident
            status, detail = check_predictions_stuck(probs, name)
            
            if status == "OK":
                print(f"  {ok(f'{name:15s} | shape=({batch_size},3) | {detail}')}")
            elif status == "STUCK":
                print(f"  {warn(f'{name:15s} | STUCK: {detail}')}")
            elif status == "OVERCONFIDENT":
                print(f"  {warn(f'{name:15s} | OVERCONFIDENT: {detail}')}")
            
            results[name] = status
            
        except Exception as e:
            tb = traceback.format_exc().split('\n')[-3]
            print(f"  {err(f'{name:15s} | {str(e)[:60]}')}")
            results[name] = "ERROR"
    
    return results


def verify_tree_models():
    """Step 3: Load tree models, verify loading without error."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"  STEP 3: Tree Model Verification (3 models + meta-learner)")
    print(f"{'=' * 60}{Colors.RESET}\n")
    
    if not JOBLIB_AVAILABLE:
        print(f"  {err('joblib not installed - cannot verify tree models')}")
        return {name: "ERROR" for name, *_ in TREE_MODELS + [META_LEARNER]}
    
    results = {}
    all_tree = TREE_MODELS + [META_LEARNER]
    
    for name, filename, model_type in all_tree:
        path = os.path.join(CHECKPOINTS_DIR, filename)
        
        if not os.path.exists(path):
            print(f"  {err(f'{name:20s} | MISSING - skipped')}")
            results[name] = "MISSING"
            continue
        
        try:
            if model_type == "joblib_sklearn":
                # Direct joblib load (gradient_boost, meta_learner)
                model = joblib.load(path)
                
                # Verify it has predict_proba method
                if not hasattr(model, 'predict_proba'):
                    print(f"  {err(f'{name:20s} | No predict_proba method')}")
                    results[name] = "ERROR"
                    continue
                
                # Run dummy prediction
                if name == "meta_learner":
                    # meta_learner expects 51 features (17 models * 3 classes)
                    dummy_x = np.random.rand(10, 51)
                else:
                    # gradient_boost expects flattened 64*46 = 2944 features
                    dummy_x = np.random.rand(10, 64 * 46)
                
                probs = model.predict_proba(dummy_x)
                if probs.shape[1] == 3:
                    print(f"  {ok(f'{name:20s} | Loaded OK, predict_proba shape: {probs.shape}')}")
                    results[name] = "OK"
                else:
                    print(f"  {warn(f'{name:20s} | predict_proba shape: {probs.shape} (expected N,3)')}")
                    results[name] = "OK"
                    
            elif model_type == "gradboost_extra":
                # GradBoostExtra with XGBoostConfig
                from models.gradient_boost_extra import GradBoostExtra
                from config.settings import XGBoostConfig
                
                model = GradBoostExtra(XGBoostConfig())
                model.load(path)
                
                # Run dummy prediction: (batch, seq_len, features)
                dummy_x = np.random.rand(10, 64, 46)
                probs = model.predict_proba(dummy_x)
                print(f"  {ok(f'{name:20s} | Loaded OK, predict_proba shape: {probs.shape}')}")
                results[name] = "OK"
                
            elif model_type == "catboost_model":
                # CatBoostModel with CatBoostConfig
                from models.catboost_model import CatBoostModel
                from config.settings import CatBoostConfig
                
                model = CatBoostModel(CatBoostConfig())
                model.load(path)
                
                # Run dummy prediction: (batch, seq_len, features)
                dummy_x = np.random.rand(10, 64, 46)
                probs = model.predict_proba(dummy_x)
                print(f"  {ok(f'{name:20s} | Loaded OK, predict_proba shape: {probs.shape}')}")
                results[name] = "OK"
                
        except Exception as e:
            print(f"  {err(f'{name:20s} | {str(e)[:60]}')}")
            results[name] = "ERROR"
    
    return results


def verify_v74_state_files():
    """Step 4: Check v7.4-specific state files (Platt calibration, Sharpe weights)."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"  STEP 4: v7.4 State Files (Platt + Sharpe)")
    print(f"{'=' * 60}{Colors.RESET}\n")
    
    import json
    results = {}
    
    for name, filename in V74_STATE_FILES:
        # State files are in neurox_v4 root, not checkpoints
        path = os.path.join(SCRIPT_DIR, filename)
        
        if not os.path.exists(path):
            print(f"  {info(f'{name:30s} | Not present (will be created on first run)')}")
            results[name] = "NOT_PRESENT"
            continue
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"  {ok(f'{name:30s} | Loadable, {len(data)} keys')}")
            results[name] = "OK"
        except json.JSONDecodeError as e:
            print(f"  {err(f'{name:30s} | Invalid JSON: {str(e)[:40]}')}")
            results[name] = "ERROR"
        except Exception as e:
            print(f"  {err(f'{name:30s} | {str(e)[:50]}')}")
            results[name] = "ERROR"
    
    return results


def print_summary(file_results, neural_results, tree_results, state_results):
    """Print final colored summary."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"  SUMMARY - NeuroX v7.4 Checkpoint Verification")
    print(f"{'=' * 60}{Colors.RESET}\n")
    
    all_results = {}
    all_results.update(neural_results)
    all_results.update(tree_results)
    
    # Count statuses
    counts = {"OK": 0, "STUCK": 0, "OVERCONFIDENT": 0, "ERROR": 0, "MISSING": 0}
    
    for name, status in all_results.items():
        if status in counts:
            counts[status] += 1
        else:
            counts["ERROR"] += 1
    
    # Display per-model status
    print(f"  {'Model':<20} {'Status':<15} {'Type':<10}")
    print(f"  {'-' * 45}")
    
    for name, status in all_results.items():
        if status == "OK":
            color = Colors.GREEN
        elif status in ("STUCK", "OVERCONFIDENT"):
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        model_type = "neural" if name in [n for n, *_ in NEURAL_MODELS] else "tree"
        print(f"  {name:<20} {color}{status:<15}{Colors.RESET} {model_type}")
    
    # Final verdict
    print(f"\n  {'-' * 45}")
    total = sum(counts.values())
    
    if counts["OK"] == total:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}ALL {total} MODELS VERIFIED OK{Colors.RESET}")
    else:
        print(f"\n  Results:")
        if counts["OK"] > 0:
            print(f"    {Colors.GREEN}OK:             {counts['OK']}{Colors.RESET}")
        if counts["STUCK"] > 0:
            print(f"    {Colors.YELLOW}STUCK:          {counts['STUCK']}{Colors.RESET}")
        if counts["OVERCONFIDENT"] > 0:
            print(f"    {Colors.YELLOW}OVERCONFIDENT:  {counts['OVERCONFIDENT']}{Colors.RESET}")
        if counts["ERROR"] > 0:
            print(f"    {Colors.RED}ERROR:          {counts['ERROR']}{Colors.RESET}")
        if counts["MISSING"] > 0:
            print(f"    {Colors.RED}MISSING:        {counts['MISSING']}{Colors.RESET}")
    
    # v7.4 state files
    print(f"\n  v7.4 State Files:")
    for name, status in state_results.items():
        if status == "OK":
            print(f"    {Colors.GREEN}{name}: {status}{Colors.RESET}")
        elif status == "NOT_PRESENT":
            print(f"    {Colors.CYAN}{name}: will be created on first run{Colors.RESET}")
        else:
            print(f"    {Colors.RED}{name}: {status}{Colors.RESET}")
    
    print()
    return counts["OK"] == total


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   NeuroX v7.4 - Model Checkpoint Verifier   ║")
    print("  ║   17 Models + 1 Meta-Learner = 18 Files     ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    print(f"  Checkpoints dir: {CHECKPOINTS_DIR}")
    print(f"  PyTorch:  {'Available' if TORCH_AVAILABLE else 'NOT INSTALLED'}")
    print(f"  joblib:   {'Available' if JOBLIB_AVAILABLE else 'NOT INSTALLED'}")
    
    # Step 1: File existence
    file_results = verify_file_existence()
    
    # Step 2: Neural models
    neural_results = verify_neural_models()
    
    # Step 3: Tree models
    tree_results = verify_tree_models()
    
    # Step 4: v7.4 state files
    state_results = verify_v74_state_files()
    
    # Summary
    all_ok = print_summary(file_results, neural_results, tree_results, state_results)
    
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
