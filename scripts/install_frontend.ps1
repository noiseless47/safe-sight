$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $Root "frontend"

if (Get-Command corepack -ErrorAction SilentlyContinue) {
  corepack enable
  corepack prepare pnpm@9.15.0 --activate
}
elseif (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
  if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Neither corepack nor npm was found. Install Node.js first."
  }
  npm install -g pnpm@9.15.0
}

pnpm --dir $FrontendDir install

Write-Host ""
Write-Host "Frontend dependencies installed."
