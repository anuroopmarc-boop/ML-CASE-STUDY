# PowerShell script to push to GitHub for Streamlit Cloud / Hugging Face Spaces deployment
param(
    [string]$RepoUrl = ""
)

if (-not $RepoUrl) {
    Write-Host "Please enter your GitHub repository URL (e.g. https://github.com/DineshBhimaneni/ml-case-study.git):" -ForegroundColor Cyan
    $RepoUrl = Read-Host "GitHub Repo URL"
}

if (-not $RepoUrl) {
    Write-Host "Error: No repository URL provided." -ForegroundColor Red
    exit 1
}

Write-Host "Setting git remote origin..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin $RepoUrl

Write-Host "Pushing main branch to GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "Now deploy your app in 60 seconds:" -ForegroundColor Green
    Write-Host "  1. Open: https://share.streamlit.io" -ForegroundColor Cyan
    Write-Host "  2. Sign in with GitHub" -ForegroundColor Cyan
    Write-Host "  3. Select your repository: $RepoUrl" -ForegroundColor Cyan
    Write-Host "  4. Set Main file path: app.py" -ForegroundColor Cyan
    Write-Host "  5. Click 'Deploy!'" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "Push failed. Check your GitHub credentials / URL." -ForegroundColor Red
}
