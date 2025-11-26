# Script to deploy server files to Raspberry Pi from Windows
# Usage: .\deploy_server.ps1 <RPI_IP> <RPI_USER>
param (
    [Parameter(Mandatory=$true)]
    [string]$RpiIp,
    [Parameter(Mandatory=$true)]
    [string]$RpiUser
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Deploying to $RpiUser@$RpiIp..."

# Check for scp
if (-not (Get-Command "scp" -ErrorAction SilentlyContinue)) {
    Write-Error "scp command not found. Please install OpenSSH Client (Settings > Apps > Optional features)."
    exit 1
}

# 1. Copy scripts (will prompt for password)
Write-Host "Copying scripts (you will be prompted for password)..."
scp "$ScriptDir\start_stream_sw.sh" "$ScriptDir\setup_server.sh" "$RpiUser@$RpiIp`:~/"

# 2. Make scripts executable (will prompt for password again)
Write-Host "Setting permissions (you will be prompted for password again)..."
ssh "$RpiUser@$RpiIp" "chmod +x ~/start_stream_sw.sh ~/setup_server.sh"

Write-Host "Deployment complete!"
Write-Host "To install dependencies on RPi, run: ssh $RpiUser@$RpiIp './setup_server.sh'"
