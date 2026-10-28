const fs = require('fs');
let content = fs.readFileSync('C:/Users/pkuma/projects/cuda-sim/demo/cuda-runtime.js', 'utf8');
const badPart = `    // constructor replaced
      this._counters = counters;
      this._deviceId = deviceId;
      /** @type {Map<number, {buffer: ArrayBuffer, sizeBytes: number}>} */
      this._allocations = new Map();
    }`;
content = content.replace(badPart, '');
fs.writeFileSync('C:/Users/pkuma/projects/cuda-sim/demo/cuda-runtime.js', content, 'utf8');
console.log('Removed leftover constructor lines successfully');
