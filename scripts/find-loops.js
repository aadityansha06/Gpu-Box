const fs = require('fs');

const code = fs.readFileSync('demo/cuda-transpiler.js', 'utf8');
const lines = code.split('\n');
lines.forEach((l, i) => {
  if (l.includes('for (') || l.includes('for(') || l.includes('transpileHostCode')) {
    console.log(`${i+1}: ${l}`);
  }
});
