param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = "python",
    [string]$Output = "artifacts/performance_v2/final_qa.json"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$outputPath = if ([IO.Path]::IsPathRooted($Output)) {
    [IO.Path]::GetFullPath($Output)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $Output))
}
$junitPath = Join-Path $root "artifacts/performance_v2/final_pytest.xml"

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-QACommand {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $started = [DateTimeOffset]::Now
    Push-Location -LiteralPath $root
    try {
        $lines = @(& $PythonPath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $finished = [DateTimeOffset]::Now
    $text = ($lines | ForEach-Object { $_.ToString() }) -join "`n"
    if ($text.Length -gt 20000) {
        $text = $text.Substring($text.Length - 20000)
    }
    [ordered]@{
        name = $Name
        command = @($PythonPath) + @($Arguments)
        started_at = $started.ToString("o")
        finished_at = $finished.ToString("o")
        duration_seconds = ($finished - $started).TotalSeconds
        exit_code = [int]$exitCode
        status = if ($exitCode -eq 0) { "passed" } else { "failed" }
        output_tail = $text
    }
}

$commands = [ordered]@{}
$commands.compileall = Invoke-QACommand "compileall" @(
    "-m", "compileall", "-q", "src", "tests", "scripts/check_empty_implementations.py"
)
$commands.ruff = Invoke-QACommand "ruff" @("-m", "ruff", "check", ".")
$commands.pytest = Invoke-QACommand "pytest" @(
    "-m", "pytest", "-q", "--junitxml=$junitPath"
)
$commands.cli_help = Invoke-QACommand "cli_help" @(
    "-m", "virtual_staining.cli", "--help"
)
$commands.pip_check = Invoke-QACommand "pip_check" @("-m", "pip", "check")
$commands.empty_implementation_scan = Invoke-QACommand "empty_implementation_scan" @(
    "scripts/check_empty_implementations.py"
)

$failed = @(
    $commands.GetEnumerator() |
        Where-Object { [int]$_.Value.exit_code -ne 0 } |
        ForEach-Object { [string]$_.Key }
)
$scopeRoots = @("src", "tests", "configs", "docs", "scripts")
$topLevelFiles = @(
    "AGENTS.md",
    "README.md",
    "PERFORMANCE_V2_CODEX_TASK.md",
    "pyproject.toml",
    "environment.yml",
    "requirements-core.txt",
    "requirements-dev.txt"
)
$implementationFiles = [Collections.Generic.List[IO.FileInfo]]::new()
foreach ($relative in $scopeRoots) {
    $path = Join-Path $root $relative
    if (Test-Path -LiteralPath $path -PathType Container) {
        Get-ChildItem -LiteralPath $path -File -Recurse |
            Where-Object {
                $_.FullName -notmatch "[\\/]__pycache__[\\/]" -and
                $_.Extension -notin @(".pyc", ".pyo")
            } |
            ForEach-Object { $implementationFiles.Add($_) }
    }
}
foreach ($relative in $topLevelFiles) {
    $path = Join-Path $root $relative
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $implementationFiles.Add((Get-Item -LiteralPath $path))
    }
}
$implementationRecords = @(
    $implementationFiles |
        Sort-Object { [IO.Path]::GetRelativePath($root, $_.FullName) } -Unique |
        ForEach-Object {
            [ordered]@{
                path = [IO.Path]::GetRelativePath($root, $_.FullName).Replace("\", "/")
                sha256 = Get-FileSha256 $_.FullName
                size_bytes = $_.Length
            }
        }
)
$aggregateText = (
    $implementationRecords | ForEach-Object { "{0}  {1}" -f $_.sha256, $_.path }
) -join "`n"
$aggregateBytes = [Text.UTF8Encoding]::new($false).GetBytes("$aggregateText`n")
$aggregateHash = [Convert]::ToHexString(
    [Security.Cryptography.SHA256]::HashData($aggregateBytes)
).ToLowerInvariant()
[int64]$implementationBytes = 0
foreach ($record in $implementationRecords) {
    $implementationBytes += [int64]$record.size_bytes
}
$report = [ordered]@{
    schema_version = 1
    captured_at = [DateTimeOffset]::Now.ToString("o")
    project_root = $root
    python = [IO.Path]::GetFullPath($PythonPath)
    status = if ($failed.Count -eq 0) { "passed" } else { "failed" }
    failed_commands = $failed
    implementation_tree = [ordered]@{
        sha256 = $aggregateHash
        file_count = $implementationRecords.Count
        total_bytes = $implementationBytes
    }
    commands = $commands
    junit = if (Test-Path -LiteralPath $junitPath -PathType Leaf) {
        [ordered]@{
            path = [IO.Path]::GetRelativePath($root, $junitPath).Replace("\", "/")
            sha256 = Get-FileSha256 $junitPath
            size_bytes = (Get-Item -LiteralPath $junitPath).Length
        }
    } else {
        $null
    }
}

[IO.Directory]::CreateDirectory((Split-Path -Parent $outputPath)) | Out-Null
$temporary = "$outputPath.tmp"
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
Move-Item -LiteralPath $temporary -Destination $outputPath -Force
if ($failed.Count -ne 0) {
    throw "Performance V2 final QA failed: $($failed -join ', ')"
}
$report
