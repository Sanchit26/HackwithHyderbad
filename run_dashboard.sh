#!/bin/bash

# AI Safety Compliance Dashboard Launcher
echo "🛡️ Starting AI Safety Compliance Dashboard..."
echo "=============================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if model exists
MODEL_PATH="/Users/syedasif/duality_ai/runs/detect/train/weights/best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    echo "Please ensure your trained model is available."
    exit 1
fi

echo "✅ Model found at $MODEL_PATH"
echo "🚀 Launching dashboard..."

# Run the dashboard
streamlit run safety_compliance_dashboard.py --server.port 8501 --server.address localhost

echo "Dashboard stopped."


