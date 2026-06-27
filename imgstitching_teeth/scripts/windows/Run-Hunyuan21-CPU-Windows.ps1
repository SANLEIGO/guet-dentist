param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-CommandChecked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $FilePath $($Arguments -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

function Ensure-PathExists([string]$PathToCheck, [string]$Label) {
    if (-not (Test-Path $PathToCheck)) {
        throw "$Label not found: $PathToCheck"
    }
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
$ServiceScript = Join-Path $RepoRoot "scripts\hunyuan3d_mv_service.py"
$ThirdPartyRepo = Join-Path $RepoRoot "third_party\Hunyuan3D-2"
$ModelDir = Join-Path $RepoRoot ".runtime\hunyuan3d\models\Hunyuan3D-2.1"
$ModelSubdir = Join-Path $ModelDir "hunyuan3d-dit-v2-1"
$ModelConfig = Join-Path $ModelSubdir "config.yaml"
$ModelWeights = Join-Path $ModelSubdir "model.fp16.ckpt"
$VenvDir = Join-Path $RepoRoot ".venv-hunyuan"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$LogDir = Join-Path $RepoRoot ".runtime\hunyuan3d\logs"
$LogFile = Join-Path $LogDir "service.windows.log"
$PidFile = Join-Path $RepoRoot ".runtime\hunyuan3d\service.windows.pid"

Write-Step "Checking repo layout"
Ensure-PathExists $ServiceScript "Service script"
Ensure-PathExists $ThirdPartyRepo "Vendored Hunyuan3D-2 repo"
Ensure-PathExists $ModelConfig "Model config"
Ensure-PathExists $ModelWeights "Model weights"

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating virtual environment"
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        & py -3.10 -m venv $VenvDir
    } elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
        & python -m venv $VenvDir
    } else {
        throw "Python 3.10+ not found. Install Python first."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment."
    }
}

Write-Step "Upgrading pip/setuptools/wheel"
Invoke-CommandChecked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -WorkingDirectory $RepoRoot

Write-Step "Installing PyTorch CPU wheels"
Invoke-CommandChecked -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install",
    "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
    "--index-url", "https://download.pytorch.org/whl/cpu"
) -WorkingDirectory $RepoRoot

Write-Step "Installing runtime dependencies"
Invoke-CommandChecked -FilePath $VenvPython -Arguments @(
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
) -WorkingDirectory $RepoRoot

Write-Step "Installing vendored hy3dgen package"
Invoke-CommandChecked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-e", $ThirdPartyRepo, "--no-deps") -WorkingDirectory $RepoRoot

Write-Step "Preparing logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path $PidFile) {
    try {
        $OldPid = Get-Content $PidFile -ErrorAction Stop
        if ($OldPid) {
            Stop-Process -Id ([int]$OldPid) -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
    } catch {
    }
}

Write-Step "Starting Hunyuan3D-2.1 CPU service in foreground"
Write-Host "RepoRoot : $RepoRoot"
Write-Host "ModelDir : $ModelDir"
Write-Host "Health   : http://127.0.0.1:8081/health"
Write-Host ""

Push-Location $RepoRoot
try {
    & $VenvPython $ServiceScript `
        --host 0.0.0.0 `
        --port 8081 `
        --model_path $ModelDir `
        --subfolder "hunyuan3d-dit-v2-1" `
        --device cpu `
        --keepalive-seconds 0.0 2>&1 | Tee-Object -FilePath $LogFile
    $ExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($ExitCode -ne 0) {
    throw "Service exited with code $ExitCode"
}
