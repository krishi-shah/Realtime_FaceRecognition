# List all video capture devices
Write-Host "Checking for camera devices..." -ForegroundColor Yellow
Write-Host ""

# Check Device Manager for cameras
Get-PnpDevice -Class "Camera" -Status "OK" | ForEach-Object {
    Write-Host "[OK] Camera found: $($_.FriendlyName)" -ForegroundColor Green
    Write-Host "     Status: $($_.Status)" -ForegroundColor White
}

Get-PnpDevice -Class "Camera" -Status "Error" | ForEach-Object {
    Write-Host "[!] Camera with error: $($_.FriendlyName)" -ForegroundColor Red
    Write-Host "     Status: $($_.Status)" -ForegroundColor White
}

# Also check for Image devices (webcams sometimes show here)
Get-PnpDevice -Class "Image" -Status "OK" | ForEach-Object {
    Write-Host "[OK] Image device found: $($_.FriendlyName)" -ForegroundColor Green
}

Write-Host ""
Write-Host "If no camera is listed above, your camera may not be properly connected or drivers are missing." -ForegroundColor Yellow
