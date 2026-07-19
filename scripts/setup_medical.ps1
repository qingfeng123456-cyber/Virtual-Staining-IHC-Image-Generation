[CmdletBinding()]
param(
    [string]$EnvironmentName = "MEDICAL",
    [string]$TorchVersion = "2.12.1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

conda create -n $EnvironmentName python=3.11 pip -y
conda run -n $EnvironmentName python -m pip install --upgrade pip setuptools wheel
conda run -n $EnvironmentName python -m pip install "torch==$TorchVersion" --index-url https://download.pytorch.org/whl/cu126
conda run -n $EnvironmentName python -m pip install -e "$ProjectRoot[dev]"
conda run -n $EnvironmentName python -m pip check
conda run -n $EnvironmentName python -c "import torch; print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available()})"

