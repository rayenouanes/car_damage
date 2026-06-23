$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Environnement GPU absent: $python"
}

Set-Location $projectRoot
& $python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false

