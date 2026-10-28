// Test script for transpiler fixes
const testCode = `
__shared__ float warpSums[32];
int tid = threadIdx.x;
int lane = tid % 32;
int warp = tid / 32;
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
sum = (lane < (blockDim.x + 31) / 32) ? warpSums[lane] : 0.0f;
int blocks = ((N / 4) + THREADS - 1) / THREADS;
`;

let code = testCode;
code = code.replace(/__shared__\s+(\w+)\s+(\w+)\s*\[([^\]]*)\]\s*;/g, (m, type, name, size) => {
  const s = size.trim() || '256';
  const t = /int/i.test(type) ? 'int' : 'float';
  return `let ${name} = __shared.getArray ? __shared.getArray("${name}", ${s}, "${t}") : __shared;`;
});

// Integer declarations with division: int/size_t/uint x = expr_with_division;
code = code.replace(/(?:const\s+)?(int|size_t|unsigned\s+int|uint32_t|int32_t|short|long)\s+(\w+)\s*=\s*([^;]+);/g, (m, type, name, expr) => {
  const trimmed = expr.trim();
  if (trimmed.includes('/') && !trimmed.startsWith('Math.floor(')) {
    return `let ${name} = Math.floor(${trimmed});`;
  }
  return `let ${name} = ${trimmed};`;
});

code = code.replace(/\b(\w+)\s*\/=\s*(\d+)\b/g, '$1 = Math.floor($1 / $2)');
code = code.replace(/\(([a-zA-Z0-9_.]+\s*[+\-*]\s*[a-zA-Z0-9_.]+)\)\s*\/\s*(\d+)\b/g, 'Math.floor(($1) / $2)');
code = code.replace(/(?<!Math\.floor\()\b(\w+)\s*\/\s*32\b/g, 'Math.floor($1 / 32)');

console.log('Result:\n' + code);
