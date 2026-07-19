param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Output = "artifacts/performance_v2/baseline_snapshot_binding_v3_20260716.json"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$outputPath = if ([IO.Path]::IsPathRooted($Output)) {
    [IO.Path]::GetFullPath($Output)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $Output))
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Immutable baseline binding already exists: $outputPath"
}

function Get-ProjectEvidence {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $path = [IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required baseline evidence is missing: $path"
    }
    $item = Get-Item -LiteralPath $path
    [ordered]@{
        path = [IO.Path]::GetRelativePath($root, $path).Replace("\", "/")
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        size_bytes = $item.Length
    }
}

function Read-ProjectJson {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $path = [IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Freeze-BaselineManifest {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRelativePath,
        [Parameter(Mandatory = $true)][string]$FrozenName,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    $source = [IO.Path]::GetFullPath((Join-Path $root $SourceRelativePath))
    $destination = [IO.Path]::GetFullPath(
        (Join-Path $root "artifacts/performance_v2/baseline_manifests/$FrozenName")
    )
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Baseline manifest source is missing: $source"
    }
    $expected = $ExpectedSha256.ToLowerInvariant()
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -ne $expected) {
        throw "Active manifest no longer matches the frozen baseline hash: $SourceRelativePath"
    }
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $frozenHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($frozenHash -ne $expected) {
            throw "Existing frozen baseline manifest has an invalid hash: $destination"
        }
    } else {
        [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }
    Get-ProjectEvidence ([IO.Path]::GetRelativePath($root, $destination))
}

$baseline = Read-ProjectJson "artifacts/performance_v2/baseline_snapshot.json"
$benchmark = Read-ProjectJson "artifacts/performance_v2/baseline_benchmark.json"
$roiAudit = Read-ProjectJson "artifacts/performance_v2/roi_grid_audit.json"
$inference = Read-ProjectJson "outputs/smoke_pipeline_final_acceptance/inference_report.json"
$rawD4 = $benchmark.combinations.raw_d4
$frozenManifests = [ordered]@{
    train = Freeze-BaselineManifest `
        "artifacts/manifests/train_manifest.csv" `
        "train_manifest.csv" `
        ([string]$baseline.manifest_sha256.train)
    validation = Freeze-BaselineManifest `
        "artifacts/manifests/val_manifest.csv" `
        "val_manifest.csv" `
        ([string]$baseline.manifest_sha256.validation)
    official_test = Freeze-BaselineManifest `
        "artifacts/manifests/test_manifest.csv" `
        "test_manifest.csv" `
        ([string]$baseline.manifest_sha256.official_test)
    smoke_test = Freeze-BaselineManifest `
        "artifacts/manifests/smoke_test_manifest.csv" `
        "smoke_test_manifest.csv" `
        ([string]$baseline.manifest_sha256.smoke_test)
}

$speed = [ordered]@{}
foreach ($name in @("raw_none", "raw_d4", "ema_none", "ema_d4")) {
    $entry = $benchmark.combinations.$name
    $speed[$name] = [ordered]@{
        device = "cpu"
        validation_images = [int]$benchmark.validation_count
        duration_seconds = [double]$entry.duration_seconds
        images_per_second = [double]$benchmark.validation_count / [double]$entry.duration_seconds
        tta = [string]$entry.tta
    }
}

$roiMetrics = [ordered]@{}
foreach ($domain in @("float", "uint8", "jpg")) {
    $macro = $rawD4.domains.$domain.macro
    $roiMetrics[$domain] = [ordered]@{
        surrogate_group_count = [int]$rawD4.domains.$domain.per_target.CD68.group_count
        mean_group_ssim = [double]$macro.roi_ssim
        mean_group_psnr = [double]$macro.roi_psnr
    }
}

$binding = [ordered]@{
    schema_version = 3
    captured_at = [DateTimeOffset]::Now.ToString("o")
    immutable_extension_of = "artifacts/performance_v2/baseline_snapshot.json"
    supersedes = [ordered]@{
        evidence = Get-ProjectEvidence "artifacts/performance_v2/baseline_snapshot_binding_v2_20260716.json"
        reason = "The v2 companion captured active manifests during a concurrent rebuild; v3 owns immutable byte copies matching the original snapshot hashes."
    }
    git_commit = $null
    git_status = "not_a_git_repository"
    environment = $baseline.environment
    data = $baseline.data
    checkpoint = [ordered]@{
        evidence = Get-ProjectEvidence "outputs/smoke_pipeline_final_acceptance/checkpoints/best_ssim.ckpt"
        architecture = $baseline.legacy_checkpoint.architecture
        parameters = $baseline.legacy_checkpoint.parameters
        approximate_macs = $baseline.legacy_checkpoint.approximate_macs
        epoch = $baseline.legacy_checkpoint.epoch
        global_step = $baseline.legacy_checkpoint.global_step
        seed = $baseline.legacy_checkpoint.seed
    }
    configuration = [ordered]@{
        evidence = Get-ProjectEvidence "outputs/smoke_pipeline_final_acceptance/effective_config.yaml"
        note = "The persisted config describes the original smoke invocation; the checkpoint records the resumed second epoch."
    }
    manifests = [ordered]@{
        frozen = $frozenManifests
        active_paths_at_capture = [ordered]@{
            train = "artifacts/manifests/train_manifest.csv"
            validation = "artifacts/manifests/val_manifest.csv"
            official_test = "artifacts/manifests/test_manifest.csv"
            smoke_test = "artifacts/manifests/smoke_test_manifest.csv"
        }
        hash_contract = "Every frozen copy must equal the manifest SHA256 stored in baseline_snapshot.json."
    }
    metrics = [ordered]@{
        selected_weights = "raw"
        selected_tta = "d4"
        domains = $rawD4.domains
        by_organ = [ordered]@{ colon = $rawD4.domains }
        by_marker = [ordered]@{ CD68 = $rawD4.domains }
        by_roi = $roiMetrics
        roi_warning = "ROI aggregates use surrogate numeric blocks and are not authoritative ROI metrics."
    }
    inference = [ordered]@{
        full_validation_speed = $speed
        smoke_peak_memory = [ordered]@{
            device = "cpu"
            count = [int]$inference.count
            duration_seconds = [double]$inference.duration_seconds
            images_per_second = [double]$inference.count / [double]$inference.duration_seconds
            peak_vram_bytes = [int64]$inference.peak_vram_bytes
            oom_retries = [int]$inference.oom_retries
            source = "outputs/smoke_pipeline_final_acceptance/inference_report.json"
        }
        gpu_peak_memory = [ordered]@{
            value = $null
            reason = "The immutable legacy baseline benchmark was CPU-only; no GPU peak was fabricated retroactively."
        }
    }
    roi_grid = [ordered]@{
        evidence = Get-ProjectEvidence "artifacts/performance_v2/roi_grid_audit.json"
        parsed_rows = [int]$roiAudit.parsed_rows
        filename_grid_verified = [bool]$roiAudit.filename_grid_verified
        context_enabled = [bool]$roiAudit.context_enabled
        gate_reasons = @($roiAudit.context_gate_reasons)
    }
    provenance = [ordered]@{
        original_snapshot = Get-ProjectEvidence "artifacts/performance_v2/baseline_snapshot.json"
        original_hash_list = Get-ProjectEvidence "artifacts/performance_v2/baseline_files.sha256"
        benchmark = Get-ProjectEvidence "artifacts/performance_v2/baseline_benchmark.json"
        inference_report = Get-ProjectEvidence "outputs/smoke_pipeline_final_acceptance/inference_report.json"
        claim_limit = "Legacy smoke baseline on a non-authoritative surrogate split; not a competition result."
    }
}

$parent = Split-Path -Parent $outputPath
[IO.Directory]::CreateDirectory($parent) | Out-Null
$temporary = "$outputPath.tmp"
$binding | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
Move-Item -LiteralPath $temporary -Destination $outputPath
Get-ProjectEvidence ([IO.Path]::GetRelativePath($root, $outputPath))
