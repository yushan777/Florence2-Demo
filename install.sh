python3 -m venv venv
source venv/bin/activate

pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate pillow requests
