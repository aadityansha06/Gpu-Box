# build-wasm.ps1
# Compiles the cuda-sim C++ simulator to WebAssembly using Emscripten.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[cuda-sim] Compiling C++ simulator to WASM via Emscripten..." -ForegroundColor Cyan

cmd.exe /c "`"$ScriptDir\build-wasm-direct.bat`""

if ($LASTEXITCODE -ne 0) {
  Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[cuda-sim] WASM build completed successfully!" -ForegroundColor Green
