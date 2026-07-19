param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Output = "artifacts/performance_v2/implementation_snapshot_20260716.json"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$outputPath = if ([IO.Path]::IsPathRooted($Output)) {
    [IO.Path]::GetFullPath($Output)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $Output))
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Immutable implementation snapshot already exists: $outputPath"
}

function Get-Evidence {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [bool]$Required = $true
    )

    $path = [IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        if ($Required) {
            throw "Required implementation evidence is missing: $path"
        }
        return [ordered]@{
            path = $RelativePath.Replace("\", "/")
            exists = $false
            sha256 = $null
            size_bytes = $null
        }
    }
    $item = Get-Item -LiteralPath $path
    [ordered]@{
        path = [IO.Path]::GetRelativePath($root, $path).Replace("\", "/")
        exists = $true
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        size_bytes = $item.Length
    }
}

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
$files = [Collections.Generic.List[IO.FileInfo]]::new()
foreach ($relative in $scopeRoots) {
    $path = Join-Path $root $relative
    if (Test-Path -LiteralPath $path -PathType Container) {
        Get-ChildItem -LiteralPath $path -File -Recurse |
            Where-Object {
                $_.FullName -notmatch "[\\/]__pycache__[\\/]" -and
                $_.Extension -notin @(".pyc", ".pyo")
            } |
            ForEach-Object { $files.Add($_) }
    }
}
foreach ($relative in $topLevelFiles) {
    $path = Join-Path $root $relative
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $files.Add((Get-Item -LiteralPath $path))
    }
}

$treeFiles = @(
    $files |
        Sort-Object { [IO.Path]::GetRelativePath($root, $_.FullName) } -Unique |
        ForEach-Object {
            [ordered]@{
                path = [IO.Path]::GetRelativePath($root, $_.FullName).Replace("\", "/")
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                size_bytes = $_.Length
            }
        }
)
$aggregateText = (
    $treeFiles | ForEach-Object { "{0}  {1}" -f $_.sha256, $_.path }
) -join "`n"
$aggregateBytes = [Text.UTF8Encoding]::new($false).GetBytes("$aggregateText`n")
$aggregateHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($aggregateBytes)).ToLowerInvariant()
[int64]$implementationBytes = 0
foreach ($record in $treeFiles) {
    $implementationBytes += [int64]$record.size_bytes
}

$junitPath = Join-Path $root "artifacts/performance_v2/final_pytest.xml"
[xml]$junit = Get-Content -LiteralPath $junitPath -Raw -Encoding UTF8
$suite = $junit.testsuites.testsuite
if (
    [int]$suite.tests -le 0 -or
    [int]$suite.failures -ne 0 -or
    [int]$suite.errors -ne 0 -or
    [int]$suite.skipped -ne 0
) {
    throw "Final pytest evidence contains failures, errors, or skipped tests."
}
$latestImplementationFile = $files |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
$junitItem = Get-Item -LiteralPath $junitPath
if ($null -eq $latestImplementationFile -or $junitItem.LastWriteTimeUtc -le $latestImplementationFile.LastWriteTimeUtc) {
    throw "Final pytest evidence is older than the implementation tree."
}
$qaPath = Join-Path $root "artifacts/performance_v2/final_qa.json"
$qa = Get-Content -LiteralPath $qaPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$qa.status -ne "passed") {
    throw "Final QA evidence is not marked passed."
}
foreach ($name in @("compileall", "ruff", "pytest", "cli_help", "pip_check", "empty_implementation_scan")) {
    $command = $qa.commands.$name
    if ($null -eq $command -or [int]$command.exit_code -ne 0 -or [string]$command.status -ne "passed") {
        throw "Final QA command did not pass: $name"
    }
}
$junitHash = (Get-FileHash -LiteralPath $junitPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($junitHash -ne ([string]$qa.junit.sha256).ToLowerInvariant()) {
    throw "Final QA JSON is not bound to the current JUnit evidence."
}
if (
    ([string]$qa.implementation_tree.sha256).ToLowerInvariant() -ne $aggregateHash -or
    [int]$qa.implementation_tree.file_count -ne $treeFiles.Count
) {
    throw "Final QA JSON is not bound to the current implementation tree."
}

$baselinePath = Join-Path $root "artifacts/performance_v2/baseline_snapshot.json"
$bindingPath = Join-Path $root "artifacts/performance_v2/baseline_snapshot_binding_v3_20260716.json"
$baseline = Get-Content -LiteralPath $baselinePath -Raw -Encoding UTF8 | ConvertFrom-Json
$binding = Get-Content -LiteralPath $bindingPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([int]$binding.schema_version -ne 3) {
    throw "The retained baseline binding must use schema version 3."
}
$baselineManifestNames = [ordered]@{
    train = "train"
    validation = "validation"
    official_test = "official_test"
    smoke_test = "smoke_test"
}
$verifiedBaselineManifests = [ordered]@{}
foreach ($entry in $baselineManifestNames.GetEnumerator()) {
    $name = [string]$entry.Key
    $originalName = [string]$entry.Value
    $evidence = $binding.manifests.frozen.$name
    $frozenPath = [IO.Path]::GetFullPath((Join-Path $root ([string]$evidence.path)))
    $actual = (Get-FileHash -LiteralPath $frozenPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $recorded = ([string]$evidence.sha256).ToLowerInvariant()
    $original = ([string]$baseline.manifest_sha256.$originalName).ToLowerInvariant()
    if ($actual -ne $recorded -or $actual -ne $original) {
        throw "Frozen baseline manifest hash mismatch: $name"
    }
    $verifiedBaselineManifests[$name] = [ordered]@{
        path = [string]$evidence.path
        sha256 = $actual
        matches_binding = $true
        matches_original_snapshot = $true
    }
}

$submissionReportPath = Join-Path $root "artifacts/performance_v2/final_smoke_submission_validation/submission_report.json"
$submissionReport = Get-Content -LiteralPath $submissionReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not [bool]$submissionReport.valid -or @($submissionReport.errors).Count -ne 0 -or @($submissionReport.zip_errors).Count -ne 0) {
    throw "The isolated smoke submission evidence is not valid."
}

$criticalPaths = @(
    "artifacts/performance_v2/baseline_snapshot.json",
    "artifacts/performance_v2/baseline_snapshot_binding_v2_20260716.json",
    "artifacts/performance_v2/baseline_snapshot_binding_v3_20260716.json",
    "artifacts/performance_v2/baseline_files.sha256",
    "artifacts/performance_v2/baseline_manifests/train_manifest.csv",
    "artifacts/performance_v2/baseline_manifests/val_manifest.csv",
    "artifacts/performance_v2/baseline_manifests/test_manifest.csv",
    "artifacts/performance_v2/baseline_manifests/smoke_test_manifest.csv",
    "artifacts/performance_v2/baseline_benchmark.json",
    "artifacts/performance_v2/strict_p0_screen_snapshot_20260716.json",
    "artifacts/performance_v2/strict_p0_screen_verification_20260716.json",
    "artifacts/performance_v2/strict_p0_historical/experiment_registry_at_screen.csv",
    "artifacts/performance_v2/A1_vs_A0_sample_promotion.json",
    "artifacts/performance_v2/A2_vs_A1_sample_promotion.json",
    "artifacts/performance_v2/roi_grid_audit.json",
    "artifacts/performance_v2/complexity_report.json",
    "artifacts/performance_v2/final_environment.json",
    "artifacts/performance_v2/final_qa.json",
    "artifacts/performance_v2/final_pytest.xml",
    "artifacts/performance_v2/final_smoke_submission_validation/submission_report.json",
    "artifacts/performance_v2/ablation_contract_registry.csv",
    "artifacts/performance_v2/ablation_contract_registry_performance_v2_p0_strict_smoke_fold0_seed2031_report.json",
    "configs/performance_v2/retained_unpromoted.yaml",
    "outputs/performance_v2/performance_v2_p0_strict_A0_screen_seed2026/checkpoints/best_ssim.ckpt",
    "outputs/performance_v2/performance_v2_p0_strict_A1_screen_seed2026/checkpoints/best_ssim.ckpt",
    "outputs/performance_v2/performance_v2_p0_strict_A2_screen_seed2026/checkpoints/best_ssim.ckpt",
    "outputs/performance_v2/dapi_mae_contract_smoke_20260716/checkpoints/dapi_mae_last.ckpt",
    "outputs/performance_v2/camp_pretrain_transfer_smoke_20260716/checkpoints/last.ckpt",
    "outputs/performance_v2/performance_v2_target_finetune_contract_20260716/checkpoints/last.ckpt",
    "outputs/performance_v2/performance_v2_organ_finetune_lineage_20260716/checkpoints/last.ckpt",
    "outputs/performance_v2/prototype_attention_visual_contract/validation/per_image_prototype_diagnostics/prototype_attention_visuals/manifest.json",
    "outputs/performance_v2/performance_v2_p0_strict_A2_smoke_seed2026/a7_gate_submission/submission_CD68.zip"
)
$critical = [ordered]@{}
foreach ($relative in $criticalPaths) {
    $critical[$relative] = Get-Evidence $relative $true
}

$snapshot = [ordered]@{
    schema_version = 1
    captured_at = [DateTimeOffset]::Now.ToString("o")
    project_root = $root
    git_commit = $null
    git_status = "not_a_git_repository"
    implementation_tree = [ordered]@{
        sha256 = $aggregateHash
        file_count = $treeFiles.Count
        total_bytes = $implementationBytes
        files = $treeFiles
    }
    qa = [ordered]@{
        evidence = Get-Evidence "artifacts/performance_v2/final_qa.json"
        compileall = [string]$qa.commands.compileall.status
        ruff = [string]$qa.commands.ruff.status
        cli_help = [string]$qa.commands.cli_help.status
        pip_check = [string]$qa.commands.pip_check.status
        pytest = [ordered]@{
            tests = [int]$suite.tests
            failures = [int]$suite.failures
            errors = [int]$suite.errors
            skipped = [int]$suite.skipped
            time_seconds = [double]$suite.time
            evidence = Get-Evidence "artifacts/performance_v2/final_pytest.xml"
        }
        static_empty_implementation_scan = [string]$qa.commands.empty_implementation_scan.status
    }
    baseline_binding = [ordered]@{
        schema_version = 3
        evidence = Get-Evidence "artifacts/performance_v2/baseline_snapshot_binding_v3_20260716.json"
        frozen_manifests = $verifiedBaselineManifests
        invalid_predecessor_retained_for_audit = Get-Evidence "artifacts/performance_v2/baseline_snapshot_binding_v2_20260716.json"
    }
    isolated_smoke_submission = [ordered]@{
        valid = $true
        expected_count = [int]$submissionReport.expected_count
        actual_count = [int]$submissionReport.actual_count
        zip_error_count = @($submissionReport.zip_errors).Count
        evidence = Get-Evidence "artifacts/performance_v2/final_smoke_submission_validation/submission_report.json"
    }
    critical_artifacts = $critical
    retained_configuration = "configs/performance_v2/retained_unpromoted.yaml"
    formal_limits = @(
        "No authoritative ROI_row_col grid is present locally.",
        "A3 was blocked before training and has no performance metric.",
        "No Performance V2 module was promoted into the retained default.",
        "Official test is empty; no official submission or leaderboard result exists.",
        "Confirm/full training was not executed."
    )
}

$parent = Split-Path -Parent $outputPath
[IO.Directory]::CreateDirectory($parent) | Out-Null
$temporary = "$outputPath.tmp"
$snapshot | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
Move-Item -LiteralPath $temporary -Destination $outputPath
Get-Evidence ([IO.Path]::GetRelativePath($root, $outputPath))
