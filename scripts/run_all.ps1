[CmdletBinding()]
param(
    [string]$EnvironmentName = "MEDICAL",
    [string]$Config = "configs/smoke.yaml",
    [string]$RunId = "smoke_pipeline_acceptance"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    conda run -n $EnvironmentName python -m compileall src tests
    conda run -n $EnvironmentName ruff check .
    conda run -n $EnvironmentName pytest -q
    conda run -n $EnvironmentName python -m virtual_staining.cli run-pipeline --config $Config --run-id $RunId
}
finally {
    Pop-Location
}

