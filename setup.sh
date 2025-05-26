#!/bin/bash
echo "Setting up environment..."
sudo apt update
sudo apt install -y python3-pip
pip3 install -r requirements.txt
echo "Done! Run: python3 src/infer.py"
