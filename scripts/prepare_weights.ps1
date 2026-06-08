param(
  [switch]$DownloadSAM2
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$WeightsDir = Join-Path $Root "backend\weights"

$dirs = @(
  "base",
  "ppe_detector",
  "person_detector",
  "sam2",
  "sam3"
)

foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path (Join-Path $WeightsDir $dir) | Out-Null
}

if ($DownloadSAM2) {
  $sam2Path = Join-Path $WeightsDir "sam2\sam2.1_hiera_base_plus.pt"
  if (-not (Test-Path -LiteralPath $sam2Path)) {
    $url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    Invoke-WebRequest -Uri $url -OutFile $sam2Path
  }
}

Write-Host "Weight directories prepared:"
Write-Host "  backend\weights\base\yolo11s.pt           YOLO11 base model for PPE training"
Write-Host "  backend\weights\ppe_detector\best.pt     custom PPE model goes here"
Write-Host "  backend\weights\person_detector\yolo11m.pt auto-downloads on first run"
Write-Host "  backend\weights\sam2\sam2.1_hiera_base_plus.pt optional SAM2 weights"
Write-Host ""
Write-Host "No training was started."
