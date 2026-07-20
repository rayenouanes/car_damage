# PowerShell helper to ensure opencv-python-headless is installed
try {
    python -m pip uninstall -y opencv-python | Out-Null
} catch {
    # ignore
}
python -m pip install --upgrade opencv-python-headless
Write-Host "opencv-python-headless installed"
