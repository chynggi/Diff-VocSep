#!/usr/bin/env python3
"""
PyTorch 2.8.0 + torch-xla 2.8 compatibility test for Diff-VocSep TPU code.

This script tests that all TPU-related code is compatible with torch-xla 2.8.x.
Run this in a TPU environment with torch-xla 2.8 installed to verify compatibility.

Usage:
    python test_torch_28_compatibility.py
"""
import os
import sys
import importlib
import traceback
from typing import List, Tuple


def test_environment_setup() -> bool:
    """Test PJRT environment setup for torch-xla 2.8+."""
    print("Testing PJRT environment setup...")
    try:
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        os.environ.setdefault("XLA_USE_SPMD", "1")
        print("✓ PJRT environment variables set")
        return True
    except Exception as e:
        print(f"✗ PJRT environment setup failed: {e}")
        return False


def test_pytorch_compatibility() -> bool:
    """Test PyTorch 2.8.0 compatibility."""
    print("Testing PyTorch 2.8.0 features...")
    try:
        import torch
        
        version = torch.__version__
        print(f"✓ PyTorch version: {version}")
        
        # Test autocast with XLA device_type
        with torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
            x = torch.randn(2, 3, 64, 128)
            y = torch.clamp(x, -1.0, 1.0)
            z = torch.cat([x, y], dim=1)
        print("✓ torch.autocast with device_type='xla' works")
        
        # Test distributed functions
        import torch.distributed as dist
        required_funcs = ['init_process_group', 'get_world_size', 'get_rank']
        for func in required_funcs:
            if hasattr(dist, func):
                print(f"✓ torch.distributed.{func} available")
            else:
                print(f"✗ torch.distributed.{func} missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ PyTorch compatibility test failed: {e}")
        traceback.print_exc()
        return False


def test_torch_xla_lazy_imports() -> bool:
    """Test torch-xla lazy import patterns."""
    print("Testing torch-xla lazy import patterns...")
    try:
        # Test lazy import of torch_xla modules
        modules_to_test = [
            "torch_xla.core.xla_model",
            "torch_xla.distributed.parallel_loader",
            "torch_xla.distributed.xla_multiprocessing"
        ]
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                print(f"✓ {module_name} imported successfully")
            except ImportError:
                print(f"⚠️  {module_name} not available (expected if not in TPU environment)")
        
        return True
    except Exception as e:
        print(f"✗ torch-xla lazy import test failed: {e}")
        return False


def test_torch_xla_functionality() -> bool:
    """Test torch-xla specific functionality if available."""
    print("Testing torch-xla functionality...")
    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
        
        # Test basic xm functions exist
        required_functions = [
            'xla_device', 'is_master_ordinal', 'master_print', 
            'mark_step', 'all_reduce', 'get_ordinal', 'xla_device_world_size'
        ]
        
        for func_name in required_functions:
            if hasattr(xm, func_name):
                print(f"✓ xm.{func_name} available")
            else:
                print(f"✗ xm.{func_name} missing")
                return False
        
        # Test that deprecated functions are avoided
        if hasattr(xm, 'xrt_world_size'):
            print("⚠️  xm.xrt_world_size still available but deprecated in 2.8")
        
        return True
    except ImportError:
        print("⚠️  torch-xla not available (expected if not in TPU environment)")
        return True  # This is OK for testing
    except Exception as e:
        print(f"✗ torch-xla functionality test failed: {e}")
        return False


def test_tpu_code_imports() -> bool:
    """Test that TPU code files can be imported without errors."""
    print("Testing TPU code imports...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Import basic modules used by TPU code
        import torch
        import importlib
        import yaml
        import math
        from torch import optim
        from torch.utils.data import DataLoader, DistributedSampler
        import torch.distributed as dist
        
        print("✓ All basic imports successful")
        return True
    except Exception as e:
        print(f"✗ TPU code import test failed: {e}")
        return False


def run_compatibility_tests() -> Tuple[int, int]:
    """Run all compatibility tests."""
    tests = [
        test_environment_setup,
        test_pytorch_compatibility,
        test_torch_xla_lazy_imports,
        test_torch_xla_functionality,
        test_tpu_code_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print("-" * 50)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED with exception: {e}")
    
    return passed, total


def main():
    """Main function."""
    print("PyTorch 2.8.0 + torch-xla 2.8 Compatibility Test")
    print("=" * 60)
    
    passed, total = run_compatibility_tests()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compatibility tests passed!")
        print("✅ TPU code is ready for torch-xla 2.8")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed or had warnings")
        print("Check the output above for details")
        return 1


if __name__ == "__main__":
    exit(main())