const fs = require('fs');
const rt = fs.readFileSync('demo/cuda-runtime.js', 'utf8');
console.log('ncclCommInitAll in cuda-runtime.js:', rt.includes('ncclCommInitAll'));
console.log('ncclCommInitRank in cuda-runtime.js:', rt.includes('ncclCommInitRank'));
