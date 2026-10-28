const fs = require('fs');

function inspectRuntime() {
  const code = fs.readFileSync('demo/cuda-runtime.js', 'utf8');
  const lines = code.split('\n');
  console.log('Total lines in cuda-runtime.js:', lines.length);

  lines.forEach((line, idx) => {
    if (line.includes('class MultiGPURuntime') || 
        line.includes('ncclAllReduce') || 
        line.includes('ncclAllGather') || 
        line.includes('ncclReduceScatter') || 
        line.includes('ncclBroadcast') ||
        line.includes('ncclReduce(')) {
      console.log(`Line ${idx + 1}: ${line.trim()}`);
    }
  });
}

inspectRuntime();
