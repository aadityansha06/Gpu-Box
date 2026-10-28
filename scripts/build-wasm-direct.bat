@echo off
setlocal
call "%~dp0..\emsdk\emsdk_env.bat"
if not exist "%~dp0..\sim\build-wasm" mkdir "%~dp0..\sim\build-wasm"
if not exist "%~dp0..\js\public" mkdir "%~dp0..\js\public"

cd /d "%~dp0..\sim"
echo [cuda-sim] Compiling C++ core to WebAssembly via em++...

em++ -O2 -std=c++17 --bind ^
  -s WASM=1 ^
  -s MODULARIZE=1 ^
  -s EXPORT_NAME="CudaSimModule" ^
  -s ALLOW_MEMORY_GROWTH=1 ^
  -s INITIAL_MEMORY=67108864 ^
  -s MAXIMUM_MEMORY=1073741824 ^
  -s STACK_SIZE=2097152 ^
  -s NO_EXIT_RUNTIME=1 ^
  -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue'] ^
  -fexceptions ^
  -Iinclude -Isrc ^
  src\runtime.cpp src\scheduler.cpp src\counters.cpp src\detectors.cpp src\multi_gpu.cpp src\rewriter.cpp src\api.cpp ^
  -o build-wasm\cuda_sim.js

if %ERRORLEVEL% NEQ 0 (
  echo [cuda-sim] WASM compilation failed!
  exit /b %ERRORLEVEL%
)

copy /y build-wasm\cuda_sim.js "%~dp0..\js\public\cuda_sim.js"
copy /y build-wasm\cuda_sim.wasm "%~dp0..\js\public\cuda_sim.wasm"

echo [cuda-sim] WebAssembly compilation succeeded!
echo Created: sim\build-wasm\cuda_sim.js
echo Created: sim\build-wasm\cuda_sim.wasm
echo Copied to: js\public\
