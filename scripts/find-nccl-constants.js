const fs = require('fs');
const runtime = fs.readFileSync('demo/cuda-runtime.js', 'utf8');
const lines = runtime.split('\n');
lines.forEach((l, i) => {
  if (l.includes('ncclSum') || l.includes('ncclFloat')) {
    console.log(`${i+1}: ${l}`);
  }
});
