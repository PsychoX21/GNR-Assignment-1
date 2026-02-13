# Download and install Ninja build system
$ninjaVersion = "1.12.1"
$ninjaUrl = "https://github.com/ninja-build/ninja/releases/download/v$ninjaVersion/ninja-win.zip"
$ninjaZip = "$env:TEMP\ninja.zip"
$ninjaDir = "$env:LOCALAPPDATA\Ninja"

Write-Host "Downloading Ninja v$ninjaVersion..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $ninjaUrl -OutFile $ninjaZip -UseBasicParsing
    Write-Host "✓ Downloaded successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Download failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "Extracting Ninja..." -ForegroundColor Cyan
if (!(Test-Path $ninjaDir)) {
    New-Item -ItemType Directory -Path $ninjaDir -Force | Out-Null
}

Expand-Archive -Path $ninjaZip -DestinationPath $ninjaDir -Force
Remove-Item $ninjaZip

Write-Host "✓ Ninja installed to: $ninjaDir" -ForegroundColor Green

# Add to PATH for current session
$env:Path = "$ninjaDir;$env:Path"

# Add to user PATH permanently
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$ninjaDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$ninjaDir;$userPath", "User")
    Write-Host "✓ Added to PATH (restart shell to use)" -ForegroundColor Green
}

# Verify installation
$ninjaPath = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninjaPath) {
    Write-Host "✓ Ninja is ready!" -ForegroundColor Green
    & ninja --version
} else {
    Write-Host "⚠ Ninja installed but not in current PATH" -ForegroundColor Yellow
    Write-Host "  Restart PowerShell or run:" -ForegroundColor Yellow
    Write-Host "  `$env:Path = `"$ninjaDir;`$env:Path`"" -ForegroundColor Cyan
}
