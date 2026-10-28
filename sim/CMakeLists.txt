cmake_minimum_required(VERSION 3.16)
project(cuda_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Source files ─────────────────────────────────────────────────────────────
set(SIM_SOURCES
  src/runtime.cpp
  src/scheduler.cpp
  src/counters.cpp
  src/detectors.cpp
  src/multi_gpu.cpp
  src/rewriter.cpp
  src/api.cpp
)

# ─── WASM build (via Emscripten) ──────────────────────────────────────────────
if(EMSCRIPTEN)
  add_executable(cuda_sim ${SIM_SOURCES})

  target_include_directories(cuda_sim PRIVATE include src)

  # Emscripten flags
  set(EM_FLAGS
    -O2
    -std=c++17
    --bind                          # Embind for JS↔C++ bridge
    -s WASM=1
    -s MODULARIZE=1
    -s EXPORT_NAME="CudaSimModule"
    -s ALLOW_MEMORY_GROWTH=1
    -s INITIAL_MEMORY=67108864      # 64 MB initial
    -s MAXIMUM_MEMORY=1073741824    # 1 GB max
    -s STACK_SIZE=2097152           # 2 MB stack
    -s NO_EXIT_RUNTIME=1
    -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue']
    -fexceptions                    # needed for std::exception
  )

  string(JOIN " " EM_FLAGS_STR ${EM_FLAGS})
  set_target_properties(cuda_sim PROPERTIES
    COMPILE_FLAGS "-O2 -fexceptions"
    LINK_FLAGS    "${EM_FLAGS_STR}"
    SUFFIX        ".js"             # output: cuda_sim.js + cuda_sim.wasm
  )

  target_compile_definitions(cuda_sim PRIVATE __EMSCRIPTEN__=1)

  install(FILES
    ${CMAKE_BINARY_DIR}/cuda_sim.js
    ${CMAKE_BINARY_DIR}/cuda_sim.wasm
    DESTINATION ${CMAKE_SOURCE_DIR}/../js/public
  )

# ─── Native build (for local testing without WASM) ───────────────────────────
else()
  add_library(cuda_sim_native STATIC ${SIM_SOURCES})
  target_include_directories(cuda_sim_native PUBLIC include PRIVATE src)
  target_compile_options(cuda_sim_native PRIVATE -O1 -fexceptions -g)

  # Native test runner
  add_executable(cuda_sim_test tests/test_runner.cpp)
  target_link_libraries(cuda_sim_test cuda_sim_native)
  target_include_directories(cuda_sim_test PRIVATE include src)
  target_compile_options(cuda_sim_test PRIVATE -O1 -fexceptions -g)
endif()
