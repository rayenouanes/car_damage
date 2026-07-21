# Sync app.py to deployment directories (Windows PowerShell)

Write-Host "Synchronizing deployment files..." -ForegroundColor Cyan

# Copy to Hugging Face Space
Write-Host "Copying to hf_space/" -ForegroundColor Green
Copy-Item -Path "app.py" -Destination "hf_space/app.py" -Force
Copy-Item -Path "requirements.txt" -Destination "hf_space/requirements.txt" -Force
Copy-Item -Path ".streamlit/config.toml" -Destination "hf_space/.streamlit/config.toml" -Force

# Copy to frontend (Streamlit Cloud)
if (Test-Path "frontend") {
    Write-Host "Copying to frontend/" -ForegroundColor Green
    Copy-Item -Path "app.py" -Destination "frontend/streamlit_app.py" -Force
    Copy-Item -Path "requirements.txt" -Destination "frontend/requirements.txt" -Force
    Copy-Item -Path ".streamlit/config.toml" -Destination "frontend/.streamlit/config.toml" -Force
}

Write-Host ""
Write-Host "Sync complete! Next steps:" -ForegroundColor Green
Write-Host ""
Write-Host "For HF Space:" -ForegroundColor Yellow
Write-Host "  cd hf_space"
Write-Host "  git add -A; git commit -m 'Sync updates'; git push"
Write-Host ""
Write-Host "For Streamlit Cloud:" -ForegroundColor Yellow
Write-Host "  Visit https://share.streamlit.io/ and redeploy"
Write-Host ""

