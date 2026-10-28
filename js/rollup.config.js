import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import copy from 'rollup-plugin-copy';

const banner = `/*!
 * gpubox v1.0.0
 * Browser-based CUDA and multi-GPU simulation/pre-validation library.
 * Copyright (c) 2026 Aadityansha Verma. All rights reserved.
 * Commercial License.
 */`;

export default [
  // ESM build (for bundlers / import)
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/cuda-sim.esm.js',
      format: 'es',
      banner,
      sourcemap: true,
    },
    plugins: [
      resolve({ browser: true }),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json', declaration: true, declarationDir: 'dist/types' }),
    ],
  },
  // CJS build (for Node / SSR)
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/cuda-sim.cjs.js',
      format: 'cjs',
      banner,
      exports: 'named',
    },
    plugins: [
      resolve({ browser: false }),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json', declaration: false }),
    ],
  },
  // UMD browser bundle (for jsDelivr <script> tag)
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/cuda-sim.min.js',
      format: 'umd',
      name: 'Gpubox',
      banner,
      sourcemap: true,
    },
    plugins: [
      resolve({ browser: true }),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json', declaration: false }),
      copy({
        targets: [
          // Copy WASM artifacts from sim build into public/
          { src: '../sim/build-wasm/cuda_sim.js',   dest: 'public' },
          { src: '../sim/build-wasm/cuda_sim.wasm', dest: 'public' },
          // Copy runtime and compatibility files into dist/ for standalone packaging
          { src: '../demo/*.js', dest: 'dist/runtime' },
          { src: '../compat.json', dest: 'dist' }
        ],
        hook: 'writeBundle',
        verbose: true,
      }),
    ],
  },
];
