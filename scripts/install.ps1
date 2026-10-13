# METRICA installer for Windows (PowerShell). Creates .venv in the repository and installs METRICA.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Dev] [-Models]
param([switch]$Dev, [switch]$Models)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = $null
foreach ($v in @("3.12", "3.13", "3.11")) {
  try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break } } catch {}
}
if (-not $py) {
  try { $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"; if ($ver -in @("3.11","3.12","3.13")) { $py = @("python") } } catch {}
}
if (-not $py) { Write-Error "Python 3.11, 3.12 or 3.13 (64-bit) is required: https://www.python.org/downloads/windows/"; exit 1 }

$pyExe = $py[0]
$pyArgs = @()
if ($py.Length -gt 1) { $pyArgs = $py[1..($py.Length - 1)] }
& $pyExe @pyArgs -m venv .venv
$venvPy = Join-Path ".venv" "Scripts\python.exe"
& $venvPy -m pip install --upgrade pip wheel
if ($Dev) { & $venvPy -m pip install -e ".[dev]" } else { & $venvPy -m pip install -e . }

if (-not (Test-Path "frontend\dist\index.html")) {
  if (Get-Command npm -ErrorAction SilentlyContinue) { Push-Location frontend; npm ci; npm run build; Pop-Location }
  else { Write-Warning "frontend\dist is missing and npm is not installed: the web interface will not be available." }
}
$metrica = Join-Path ".venv" "Scripts\metrica.exe"
if ($Models) { & $metrica models download }
& $metrica doctor
Write-Host ""
Write-Host "Installed. Next steps:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  metrica models download     # once, ~280 MB"
Write-Host "  metrica serve"
