const fs = require('fs');

const runtime = fs.readFileSync('demo/cuda-runtime.js', 'utf8');
const transpiler = fs.readFileSync('demo/cuda-transpiler.js', 'utf8');

console.log('ncclSum in runtime:', runtime.includes('ncclSum'));
console.log('ncclSum in transpiler:', transpiler.includes('ncclSum'));
console.log('ncclFloat in runtime:', runtime.includes('ncclFloat'));
console.log('ncclFloat in transpiler:', transpiler.includes('ncclFloat'));
