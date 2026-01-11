# Check for processes that might be using the camera
Write-Host "Checking for applications that might be using the camera..." -ForegroundColor Yellow
Write-Host ""

$cameraApps = @(
    "Teams", "Zoom", "Skype", "Discord", "Slack", 
    "obs64", "obs32", "streamlabs", "xsplit",
    "CameraServer", "WindowsCamera", "YourPhone"
)

$found = $false
foreach ($app in $cameraApps) {
    $process = Get-Process -Name $app -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "[!] Found: $app (PID: $($process.Id))" -ForegroundColor Red
        $found = $true
    }
}

if (-not $found) {
    Write-Host "[OK] No common camera apps detected running" -ForegroundColor Green
}

Write-Host ""
Write-Host "TIP: Close any video call or streaming apps before testing the camera." -ForegroundColor Cyan
