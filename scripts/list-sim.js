const fs = require('fs');

console.log('sim files:');
function listDir(d) {
  const items = fs.readdirSync(d, { withFileTypes: true });
  for (const item of items) {
    const p = d + '/' + item.name;
    if (item.isDirectory()) {
      if (item.name !== 'build-wasm') listDir(p);
    } else {
      console.log(p);
    }
  }
}
listDir('sim');
