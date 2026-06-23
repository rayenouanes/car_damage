$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$backend = Start-Process `
    -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru

try {
    Start-Sleep -Seconds 2
    python -m streamlit run frontend/streamlit_app.py
}
finally {
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
}
