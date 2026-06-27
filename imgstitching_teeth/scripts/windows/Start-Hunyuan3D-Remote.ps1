param(
    [string]$RepoRoot = "",
    [ValidateSet("2mini-turbo", "2.1")]
    [string]$ModelPreset = "2mini-turbo",
    [ValidateSet("cpu", "cuda")]
    [string]$DevicePreset = "cpu",
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8081,
    [switch]$SkipTorchInstall,
    [switch]$SkipDependencyInstall,
    [switch]$SkipModelDownload,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-RepoRootCandidate {
    param(
        [string]$Candidate
    )
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $false
    }
    if (-not (Test-Path $Candidate)) {
        return $false
    }
    $resolved = (Resolve-Path $Candidate).Path
    $serviceScript = Join-Path $resolved "scripts\hunyuan3d_mv_service.py"
    return (Test-Path $serviceScript)
}

function Resolve-RepoRootPath {
    param(
        [string]$ExplicitRepoRoot,
        [string]$ScriptRoot
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($ExplicitRepoRoot)) {
        $candidates += $ExplicitRepoRoot
    }
    $candidates += @(
        (Join-Path $ScriptRoot "..\.."),
        (Get-Location).Path,
        $ScriptRoot,
        (Join-Path $ScriptRoot "..")
    )

    foreach ($candidate in $candidates) {
        if (Test-RepoRootCandidate $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $message = @"
Could not locate the repo root.

Expected to find:
  scripts\hunyuan3d_mv_service.py

Tried these locations:
  $($candidates -join "`n  ")

Fix:
1. Put this launcher back inside the repo's scripts\windows\ folder, or
2. Run the script with -RepoRoot D:\path\to\imgstitching_teeth
"@
    throw $message
}

function Get-PythonBootstrapCommand {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        return @("py", "-3.10")
    }
    if (Get-Command python.exe -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python 3.10+ was not found. Please install Python 3.10 or 3.11 first."
}

function Invoke-CommandChecked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

function Ensure-RepoLayout {
    param(
        [string]$Root
    )
    $thirdPartyRoot = Join-Path $Root "third_party"
    $hunyuanRepo = Join-Path $thirdPartyRoot "Hunyuan3D-2"
    if (-not (Test-Path $hunyuanRepo)) {
        Write-Step "Cloning Tencent-Hunyuan/Hunyuan3D-2 into third_party"
        New-Item -ItemType Directory -Force -Path $thirdPartyRoot | Out-Null
        Invoke-CommandChecked -FilePath "git" -Arguments @("clone", "--depth", "1", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git", $hunyuanRepo)
    }
    return $hunyuanRepo
}

function Patch-ShapegenInit {
    param(
        [string]$RepoPath
    )
    $initPath = Join-Path $RepoPath "hy3dgen\shapegen\__init__.py"
    if (-not (Test-Path $initPath)) {
        return
    }
    $patched = @"
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from .pipelines import Hunyuan3DDiTPipeline, Hunyuan3DDiTFlowMatchingPipeline
try:
    from .postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier
except Exception:
    FaceReducer = None
    FloaterRemover = None
    DegenerateFaceRemover = None
    MeshSimplifier = None
from .preprocessors import ImageProcessorV2, IMAGE_PROCESSORS, DEFAULT_IMAGEPROCESSOR
"@
    Set-Content -Path $initPath -Value $patched -Encoding UTF8
}

function Get-ModelConfig {
    param(
        [string]$Preset
    )
    switch ($Preset) {
        "2.1" {
            return @{
                ModelDirName = "Hunyuan3D-2.1"
                ModelRepoId = "tencent/Hunyuan3D-2.1"
                Subfolder = "hunyuan3d-dit-v2-1"
                CheckpointFile = "model.fp16.ckpt"
                EnableFlashVDM = $false
            }
        }
        "2mini-turbo" {
            return @{
                ModelDirName = "Hunyuan3D-2mini"
                ModelRepoId = "tencent/Hunyuan3D-2mini"
                Subfolder = "hunyuan3d-dit-v2-mini-turbo"
                CheckpointFile = "model.fp16.safetensors"
                EnableFlashVDM = $true
            }
        }
        default {
            throw "Unsupported model preset: $Preset"
        }
    }
}

$RepoRoot = Resolve-RepoRootPath -ExplicitRepoRoot $RepoRoot -ScriptRoot $PSScriptRoot
$pythonBootstrap = Get-PythonBootstrapCommand
$venvDir = Join-Path $RepoRoot ".venv-hunyuan"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$serviceScript = Join-Path $RepoRoot "scripts\hunyuan3d_mv_service.py"
$runtimeDir = Join-Path $RepoRoot ".runtime\hunyuan3d"
$logsDir = Join-Path $runtimeDir "logs"
$serviceLog = Join-Path $logsDir "service.windows.log"
$servicePidFile = Join-Path $runtimeDir "service.windows.pid"
$modelConfig = Get-ModelConfig -Preset $ModelPreset
$modelRoot = Join-Path $runtimeDir ("models\" + $modelConfig.ModelDirName)
$modelSubdir = Join-Path $modelRoot $modelConfig.Subfolder
$configFile = Join-Path $modelSubdir "config.yaml"
$checkpointFile = Join-Path $modelSubdir $modelConfig.CheckpointFile
$hunyuanRepo = Ensure-RepoLayout -Root $RepoRoot

Write-Step "Preparing directories"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $modelSubdir | Out-Null
Patch-ShapegenInit -RepoPath $hunyuanRepo

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment at $venvDir"
    $venvArgs = @()
    if ($pythonBootstrap.Length -gt 1) {
        $venvArgs += $pythonBootstrap[1..($pythonBootstrap.Length - 1)]
    }
    $venvArgs += @("-m", "venv", $venvDir)
    Invoke-CommandChecked -FilePath $pythonBootstrap[0] -Arguments $venvArgs
}

Write-Step "Upgrading pip/setuptools/wheel"
Invoke-CommandChecked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if (-not $SkipTorchInstall) {
    if ($DevicePreset -eq "cuda") {
        Write-Step "Installing PyTorch CUDA 12.4 wheels"
        Invoke-CommandChecked -FilePath $venvPython -Arguments @(
            "-m", "pip", "install",
            "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        )
    } else {
        Write-Step "Installing PyTorch CPU wheels"
        Invoke-CommandChecked -FilePath $venvPython -Arguments @(
            "-m", "pip", "install",
            "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        )
    }
}

if (-not $SkipDependencyInstall) {
    Write-Step "Installing runtime dependencies"
    Invoke-CommandChecked -FilePath $venvPython -Arguments @(
        "-m", "pip", "install",
        "ninja",
        "pybind11",
        "diffusers",
        "einops",
        "transformers",
        "omegaconf",
        "tqdm",
        "trimesh",
        "accelerate",
        "fastapi",
        "uvicorn",
        "huggingface_hub>=0.30",
        "pyyaml",
        "scikit-image",
        "opencv-python",
        "pillow",
        "rembg",
        "onnxruntime",
        "safetensors"
    )
    Write-Step "Installing editable hy3dgen package from vendored Hunyuan repo"
    Invoke-CommandChecked -FilePath $venvPython -Arguments @("-m", "pip", "install", "-e", $hunyuanRepo, "--no-deps")
}

if (-not $SkipModelDownload) {
    Write-Step "Downloading model preset $ModelPreset"
    $configUrl = "https://huggingface.co/$($modelConfig.ModelRepoId)/resolve/main/$($modelConfig.Subfolder)/config.yaml"
    $checkpointUrl = "https://huggingface.co/$($modelConfig.ModelRepoId)/resolve/main/$($modelConfig.Subfolder)/$($modelConfig.CheckpointFile)"
    Invoke-CommandChecked -FilePath "curl.exe" -Arguments @("-L", "--fail", "--retry", "5", "--retry-all-errors", "--retry-delay", "2", "-C", "-", "-o", $configFile, $configUrl)
    Invoke-CommandChecked -FilePath "curl.exe" -Arguments @("-L", "--fail", "--retry", "5", "--retry-all-errors", "--retry-delay", "2", "-C", "-", "-o", $checkpointFile, $checkpointUrl)
}

$serviceArgs = @(
    $serviceScript,
    "--host", $BindHost,
    "--port", "$Port",
    "--model_path", $modelRoot,
    "--subfolder", $modelConfig.Subfolder,
    "--device", $DevicePreset,
    "--keepalive-seconds", "0.0"
)
if ($modelConfig.EnableFlashVDM -and $DevicePreset -eq "cuda") {
    $serviceArgs += "--enable_flashvdm"
}

if ($Foreground) {
    Write-Step "Starting remote Hunyuan service in foreground"
    & $venvPython @serviceArgs
    exit $LASTEXITCODE
}

Write-Step "Starting remote Hunyuan service in background"
$existingPid = $null
if (Test-Path $servicePidFile) {
    try {
        $existingPid = Get-Content $servicePidFile -ErrorAction Stop
    } catch {
        $existingPid = $null
    }
}
if ($existingPid) {
    try {
        Stop-Process -Id ([int]$existingPid) -Force -ErrorAction Stop
        Start-Sleep -Seconds 1
    } catch {
    }
}

$proc = Start-Process -FilePath $venvPython -ArgumentList $serviceArgs -WorkingDirectory $RepoRoot -RedirectStandardOutput $serviceLog -RedirectStandardError $serviceLog -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $servicePidFile -Encoding ASCII

Write-Host ""
Write-Host "Remote service started." -ForegroundColor Green
Write-Host "  PID   : $($proc.Id)"
Write-Host "  URL   : http://$BindHost`:$Port/health"
Write-Host "  Log   : $serviceLog"
Write-Host "  Model : $ModelPreset"
Write-Host "  Device: $DevicePreset"
Write-Host ""
Write-Host "Recommended client .env values on the Mac side:" -ForegroundColor Yellow
Write-Host "  HUNYUAN3D_SERVICE_URL=http://100.69.194.152:$Port"
Write-Host "  HUNYUAN3D_SERVICE_MODE=single_image"
if ($ModelPreset -eq "2.1") {
    Write-Host "  HUNYUAN3D_MODEL_PATH=tencent/Hunyuan3D-2.1"
    Write-Host "  HUNYUAN3D_SUBFOLDER=hunyuan3d-dit-v2-1"
} else {
    Write-Host "  HUNYUAN3D_MODEL_PATH=tencent/Hunyuan3D-2mini"
    Write-Host "  HUNYUAN3D_SUBFOLDER=hunyuan3d-dit-v2-mini-turbo"
}
Write-Host "  # Remote device is controlled on Windows by -DevicePreset $DevicePreset"
