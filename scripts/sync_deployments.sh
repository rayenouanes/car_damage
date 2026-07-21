#!/bin/bash
# Sync app.py to deployment directories

set -e

echo "📡 Synchronizing deployment files..."

# Copy to Hugging Face Space
echo "✅ Copying to hf_space/"
cp app.py hf_space/app.py
cp requirements.txt hf_space/requirements.txt
cp .streamlit/config.toml hf_space/

# Copy to Streamlit deployment (if exists)
if [ -d "frontend" ]; then
    echo "✅ Copying to frontend/"
    cp app.py frontend/streamlit_app.py
    cp requirements.txt frontend/requirements.txt
fi

echo "🎉 Sync complete! Push to deploy:"
echo ""
echo "For HF Space:"
echo "  cd hf_space && git add -A && git commit -m 'Sync app updates' && git push"
echo ""
echo "For Streamlit Cloud:"
echo "  Redeploy from: https://share.streamlit.io/"
