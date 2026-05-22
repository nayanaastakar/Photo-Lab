Set-Location (Split-Path -Path $MyInvocation.MyCommand.Definition -Parent)
python -m pip install -r requirements.txt
streamlit run app.py --server.port 8503 --server.headless true
