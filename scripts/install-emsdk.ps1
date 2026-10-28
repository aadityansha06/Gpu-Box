# install-emsdk.ps1
# Installs Emscripten SDK (emsdk) to C:\Users\pkuma\projects\cuda-sim\emsdk
# Then activates the latest version for use in build-wasm.ps1

$ErrorActionPreference = "Stop"
$EmsdkDir = "$PSScriptRoot\..\emsdk"

Write-Host "[cuda-sim] Installing Emscripten SDK..." -ForegroundColor Cyan

# Clone emsdk repo if not already present
if (-not (Test-Path "$EmsdkDir\.git")) {
  Write-Host "Cloning emsdk repository..."
  $gitExe = "C:\Users\pkuma\AppData\Local\Programs\Git\cmd\git.exe"
  & $gitExe clone https://github.com/emscripten-core/emsdk.git $EmsdkDir
} else {
  Write-Host "emsdk already cloned; pulling latest..."
  $gitExe = "C:\Users\pkuma\AppData\Local\Programs\Git\cmd\git.exe"
  Push-Location $EmsdkDir
  & $gitExe pull
  Pop-Location
}

$env:Path = "C:\Users\pkuma\AppData\Local\Programs\Python\Python312;C:\Users\pkuma\AppData\Local\Programs\Python\Python312\Scripts;C:\Users\pkuma\AppData\Local\Programs\Git\cmd;$env:Path"

# Run emsdk install + activate
Write-Host "Installing latest emsdk toolchain..."
Push-Location $EmsdkDir
& cmd /c "set PATH=C:\Users\pkuma\AppData\Local\Programs\Python\Python312;C:\Users\pkuma\AppData\Local\Programs\Git\cmd;%PATH% && emsdk.bat install latest"
Write-Host "Activating latest emsdk toolchain..."
& cmd /c "set PATH=C:\Users\pkuma\AppData\Local\Programs\Python\Python312;C:\Users\pkuma\AppData\Local\Programs\Git\cmd;%PATH% && emsdk.bat activate latest"
Pop-Location

Write-Host "[cuda-sim] Emscripten SDK installed successfully!" -ForegroundColor Green
Write-Host "Run scripts\build-wasm.ps1 to compile the simulator to WASM."
