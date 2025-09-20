#!/usr/bin/env python3
"""
Test script to verify core dashboard functionality without Streamlit dependencies
"""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "/Users/syedasif/duality_ai/runs/detect/train/weights/best.pt",
        "/Users/syedasif/duality_ai/Hackathon2_scripts/yolo_params.yaml",
        "/Users/syedasif/duality_ai/Hackathon2_scripts/classes.txt",
        "/Users/syedasif/duality_ai/runs/detect/train/results.csv"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist

def test_yaml_loading():
    """Test YAML configuration loading"""
    print("\n🔍 Testing YAML configuration...")
    
    try:
        import yaml
        with open("/Users/syedasif/duality_ai/Hackathon2_scripts/yolo_params.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        print(f"✅ YAML loaded successfully")
        print(f"   Classes: {config.get('names', 'Not found')}")
        print(f"   Number of classes: {config.get('nc', 'Not found')}")
        return True
    except Exception as e:
        print(f"❌ YAML loading failed: {e}")
        return False

def test_csv_loading():
    """Test CSV results loading"""
    print("\n🔍 Testing CSV results loading...")
    
    try:
        import pandas as pd
        results = pd.read_csv("/Users/syedasif/duality_ai/runs/detect/train/results.csv")
        
        print(f"✅ CSV loaded successfully")
        print(f"   Rows: {len(results)}")
        print(f"   Columns: {list(results.columns)}")
        
        if len(results) > 0:
            latest = results.iloc[-1]
            print(f"   Latest mAP@0.5: {latest.get('metrics/mAP50(B)', 'N/A'):.3f}")
            print(f"   Latest Precision: {latest.get('metrics/precision(B)', 'N/A'):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        return False

def test_model_loading():
    """Test YOLO model loading"""
    print("\n🔍 Testing YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        model_path = "/Users/syedasif/duality_ai/runs/detect/train/weights/best.pt"
        model = YOLO(model_path)
        
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🛡️ AI Safety Compliance Dashboard - Core Functionality Test")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_yaml_loading,
        test_csv_loading,
        test_model_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        print("\nTo run the dashboard:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: streamlit run safety_compliance_dashboard.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


