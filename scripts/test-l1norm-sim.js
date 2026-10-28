// Test L1 Norm computation manually to verify expected math
const N = 16;
const h_x = [
   1, -2,  3, -4,
   5, -6,  7, -8,
   9, -10, 11, -12,
  13, -14, 15, -16
];

let sum = 0;
for (let v of h_x) sum += Math.abs(v);
console.log('Expected L1 Norm =', sum); // 1 + 2 + ... + 16 = 136
console.log('First 4 normalized:', h_x.slice(0, 4).map(v => (v / sum).toFixed(6)));
