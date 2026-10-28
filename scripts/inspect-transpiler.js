const fs = require('fs');

function inspectTranspiler() {
  const code = fs.readFileSync('demo/cuda-transpiler.js', 'utf8');
  const lines = code.split('\n');
  console.log('Total lines in cuda-transpiler.js:', lines.length);

  lines.forEach((line, idx) => {
    if (line.toLowerCase().includes('nccl') || line.includes('cudaSetDevice') || line.includes('cudaMalloc')) {
      console.log(`Line ${idx + 1}: ${line.trim()}`);
    }
  });
}

inspectTranspiler();
