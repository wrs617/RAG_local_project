@echo off
cd /d "%~dp0"
python -m streamlit run app_qa.py --server.address 0.0.0.0 --server.port 8510
