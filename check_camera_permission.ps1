# Check Windows Camera Privacy Settings via Registry
Write-Host "Checking Windows Camera Privacy Settings..." -ForegroundColor Yellow
Write-Host ""

# Check global camera access
$cameraAccess = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam" -Name "Value" -ErrorAction SilentlyContinue

if ($cameraAccess) {
    $value = $cameraAccess.Value
    Write-Host "Global Camera Access: " -NoNewline
    if ($value -eq "Allow") {
        Write-Host "ALLOWED" -ForegroundColor Green
    } elseif ($value -eq "Deny") {
        Write-Host "DENIED" -ForegroundColor Red
        Write-Host "  >> You need to enable camera access in Windows Settings!" -ForegroundColor Red
    } else {
        Write-Host "UNKNOWN ($value)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Could not read registry setting" -ForegroundColor Yellow
}

Write-Host ""

# Check if camera service is running
$cameraService = Get-Service -Name "FrameServer" -ErrorAction SilentlyContinue
if ($cameraService) {
    Write-Host "Camera Frame Server Service: " -NoNewline
    if ($cameraService.Status -eq "Running") {
        Write-Host "RUNNING" -ForegroundColor Green
    } else {
        Write-Host "$($cameraService.Status)" -ForegroundColor Red
        Write-Host "  >> Trying to start service..." -ForegroundColor Yellow
        try {
            Start-Service -Name "FrameServer"
            Write-Host "  >> Service started!" -ForegroundColor Green
        } catch {
            Write-Host "  >> Failed to start service: $_" -ForegroundColor Red
        }
    }
} else {
    Write-Host "Camera Frame Server Service: NOT FOUND" -ForegroundColor Red
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Make sure camera privacy settings are enabled" -ForegroundColor White
Write-Host "2. Restart your computer to apply changes" -ForegroundColor White
Write-Host "3. Try the camera test again" -ForegroundColor White
