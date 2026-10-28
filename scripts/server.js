// cuda-sim dev server
// Serves from the project root so /demo/index.html, /js/dist, /sim etc. all work.

const http = require('http');
const fs   = require('fs');
const path = require('path');

// Hardcoded project root (not __dirname which would be %TEMP%)
const ROOT = 'C:\\Users\\pkuma\\projects\\cuda-sim';

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.mjs':  'application/javascript',
  '.ts':   'text/plain',
  '.wasm': 'application/wasm',
  '.css':  'text/css',
  '.cu':   'text/plain',
  '.md':   'text/plain',
  '.json': 'application/json',
  '.map':  'application/json',
};

http.createServer((req, res) => {
  let url = req.url.split('?')[0];

  // Root redirect
  if (url === '/' || url === '/demo' || url === '/demo/') {
    url = '/demo/index.html';
  }

  // Normalize path separators
  const rel = url.replace(/\//g, path.sep).replace(/^\\/, '');
  const filePath = path.join(ROOT, rel);

  // Safety: don't serve files outside root
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      console.log('Not found:', filePath);
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found: ' + url);
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const ct  = MIME[ext] || 'application/octet-stream';
    res.writeHead(200, {
      'Content-Type': ct,
      'Access-Control-Allow-Origin': '*',
    });
    res.end(data);
  });
}).listen(3456, '127.0.0.1', () => {
  console.log('cuda-sim dev server running at http://localhost:3456/demo/');
});
