$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $Root "frontend"

if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
  if (Get-Command corepack -ErrorAction SilentlyContinue) {
    corepack enable
    corepack prepare pnpm@9.15.0 --activate
  }
  else {
    throw "pnpm/corepack not found. Run scripts/install_frontend.ps1 first."
  }
}

pnpm --dir $FrontendDir dev
