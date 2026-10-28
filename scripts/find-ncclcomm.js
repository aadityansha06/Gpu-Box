const fs = require('fs');
const rt = fs.readFileSync('demo/cuda-runtime.js', 'utf8');
const lines = rt.split('\n');
lines.forEach((l, i) => {
  if (l.includes('class NcclComm') || l.includes('function NcclComm')) {
    console.log(`${i+1}: ${l}`);
  }
});
