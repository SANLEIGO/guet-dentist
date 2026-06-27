param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [ValidateSet("2mini-turbo", "2.1")]
    [string]$ModelPreset = "2mini-turbo",
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8081,
    [switch]$SkipTorchInstall,
    [switch]$SkipDependencyInstall,
    [switch]$SkipModelDownload,
    [switch]$Foreground
)

$argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "Start-Hunyuan3D-Remote.ps1"),
    "-RepoRoot", $RepoRoot,
    "-ModelPreset", $ModelPreset,
    "-DevicePreset", "cpu",
    "-BindHost", $BindHost,
    "-Port", "$Port"
)

if ($SkipTorchInstall) { $argsList += "-SkipTorchInstall" }
if ($SkipDependencyInstall) { $argsList += "-SkipDependencyInstall" }
if ($SkipModelDownload) { $argsList += "-SkipModelDownload" }
if ($Foreground) { $argsList += "-Foreground" }

& powershell.exe @argsList
exit $LASTEXITCODE
