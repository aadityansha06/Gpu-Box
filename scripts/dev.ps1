# dev.ps1
# Starts a local HTTP server serving the demo page at http://localhost:3456
# Watches for changes to demo/ and js/ (requires Node).

$ErrorActionPreference = "Stop"
$ServerScript = "$PSScriptRoot\server.js"

Write-Host "[cuda-sim] Starting demo server at http://localhost:3456" -ForegroundColor Cyan
Write-Host "Open http://localhost:3456/demo/ in your browser."
Write-Host "Press Ctrl+C to stop."

node $ServerScript
