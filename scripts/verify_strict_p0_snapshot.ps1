param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Output = "artifacts/performance_v2/strict_p0_screen_verification_20260716.json"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$snapshotPath = Join-Path $root "artifacts/performance_v2/strict_p0_screen_snapshot_20260716.json"
$outputPath = if ([IO.Path]::IsPathRooted($Output)) {
    [IO.Path]::GetFullPath($Output)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $Output))
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Immutable strict-P0 verification already exists: $outputPath"
}
$snapshot = Get-Content -LiteralPath $snapshotPath -Raw -Encoding UTF8 | ConvertFrom-Json
$capturedAt = [DateTimeOffset]::Parse([string]$snapshot.captured_at)

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-Evidence {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = [IO.Path]::GetFullPath($Path)
    $item = Get-Item -LiteralPath $resolved
    [ordered]@{
        path = [IO.Path]::GetRelativePath($root, $resolved).Replace("\", "/")
        sha256 = Get-Sha256 $resolved
        size_bytes = $item.Length
    }
}

function Get-HistoricalRunTree {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256,
        [Parameter(Mandatory = $true)][int]$ExpectedFileCount,
        [Parameter(Mandatory = $true)][Int64]$ExpectedBytes
    )

    $runDirectory = Join-Path $root "outputs/performance_v2/$RunId"
    $allFiles = @(Get-ChildItem -LiteralPath $runDirectory -File -Recurse)
    $historicalFiles = @(
        $allFiles |
            Where-Object { [DateTimeOffset]$_.LastWriteTime -le $capturedAt } |
            Sort-Object FullName
    )
    $records = @(
        $historicalFiles | ForEach-Object {
            [ordered]@{
                path = [IO.Path]::GetRelativePath($root, $_.FullName).Replace("\", "/")
                sha256 = Get-Sha256 $_.FullName
                size_bytes = $_.Length
            }
        }
    )
    $text = ($records | ForEach-Object { "{0}  {1}" -f $_.sha256, $_.path }) -join "`n"
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes("$text`n")
    $aggregate = [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($bytes)
    ).ToLowerInvariant()
    [int64]$totalBytes = 0
    foreach ($record in $records) {
        $totalBytes += [int64]$record.size_bytes
    }
    $verified = (
        $aggregate -eq $ExpectedSha256.ToLowerInvariant() -and
        $records.Count -eq $ExpectedFileCount -and
        $totalBytes -eq $ExpectedBytes
    )
    if (-not $verified) {
        throw (
            "Historical run tree no longer matches the frozen snapshot: $RunId " +
            "actual=$aggregate/$($records.Count)/$totalBytes " +
            "expected=$ExpectedSha256/$ExpectedFileCount/$ExpectedBytes"
        )
    }
    [ordered]@{
        run_id = $RunId
        verified = $true
        sha256 = $aggregate
        file_count = $records.Count
        total_bytes = $totalBytes
        selection_rule = "files with LastWriteTime <= immutable snapshot captured_at"
        post_capture_files = @(
            $allFiles |
                Where-Object { [DateTimeOffset]$_.LastWriteTime -gt $capturedAt } |
                Sort-Object FullName |
                ForEach-Object {
                    [IO.Path]::GetRelativePath($root, $_.FullName).Replace("\", "/")
                }
        )
    }
}

function Freeze-RegistryPrefix {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    $bytes = [IO.File]::ReadAllBytes($SourcePath)
    $expected = $ExpectedSha256.ToLowerInvariant()
    [byte[]]$matched = @()
    for ($index = 0; $index -lt $bytes.Length; $index++) {
        if ($bytes[$index] -ne 10) {
            continue
        }
        [byte[]]$candidate = $bytes[0..$index]
        $digest = [Convert]::ToHexString(
            [Security.Cryptography.SHA256]::HashData($candidate)
        ).ToLowerInvariant()
        if ($digest -eq $expected) {
            $matched = $candidate
            break
        }
    }
    if ($matched.Count -eq 0) {
        throw "The historical registry prefix cannot be recovered byte-for-byte."
    }
    if (Test-Path -LiteralPath $DestinationPath) {
        if ((Get-Sha256 $DestinationPath) -ne $expected) {
            throw "Existing historical registry copy has the wrong hash."
        }
    } else {
        [IO.Directory]::CreateDirectory((Split-Path -Parent $DestinationPath)) | Out-Null
        [IO.File]::WriteAllBytes($DestinationPath, $matched)
    }
    Get-Evidence $DestinationPath
}

$pathChecks = [ordered]@{}
foreach ($property in $snapshot.input_hashes.PSObject.Properties) {
    $relative = [string]$property.Name
    $path = Join-Path $root $relative
    $expected = ([string]$property.Value).ToLowerInvariant()
    $actual = Get-Sha256 $path
    $pathChecks[$relative] = [ordered]@{
        expected_sha256 = $expected
        current_sha256 = $actual
        current_matches_capture = $actual -eq $expected
        historical_bytes_available = $actual -eq $expected
    }
}
foreach ($property in $snapshot.output_hashes.PSObject.Properties) {
    $relative = [string]$property.Name
    $path = Join-Path $root $relative
    $expected = ([string]$property.Value).ToLowerInvariant()
    $actual = Get-Sha256 $path
    $pathChecks[$relative] = [ordered]@{
        expected_sha256 = $expected
        current_sha256 = $actual
        current_matches_capture = $actual -eq $expected
        historical_bytes_available = $actual -eq $expected
    }
}

$registryHistoricalPath = Join-Path $root "artifacts/performance_v2/strict_p0_historical/experiment_registry_at_screen.csv"
$registryEvidence = Freeze-RegistryPrefix `
    (Join-Path $root "artifacts/performance_v2/experiment_registry.csv") `
    ([string]$snapshot.output_hashes.'artifacts/performance_v2/experiment_registry.csv') `
    $registryHistoricalPath
$pathChecks["artifacts/performance_v2/experiment_registry.csv"].historical_bytes_available = $true
$pathChecks["artifacts/performance_v2/experiment_registry.csv"].historical_copy = $registryEvidence

$runTrees = [ordered]@{}
foreach ($stageName in @("A0", "A1", "A2")) {
    $stage = $snapshot.stages.$stageName
    $runTrees[$stageName] = Get-HistoricalRunTree `
        ([string]$stage.run_id) `
        ([string]$stage.run_tree_sha256) `
        ([int]$stage.run_file_count) `
        ([int64]$stage.run_total_bytes)
}

$unavailable = @(
    $pathChecks.GetEnumerator() |
        Where-Object { -not [bool]$_.Value.historical_bytes_available } |
        ForEach-Object { [string]$_.Key }
)
$verification = [ordered]@{
    schema_version = 1
    captured_at = [DateTimeOffset]::Now.ToString("o")
    immutable_snapshot = Get-Evidence $snapshotPath
    capture_time = $capturedAt.ToString("o")
    verified_historical_run_trees = $runTrees
    captured_path_checks = $pathChecks
    historical_bytes_unavailable = $unavailable
    interpretation = (
        "A0/A1/A2 capture-time run trees, train/val manifests, screen report, and " +
        "registry prefix remain byte-verifiable. The old p0 suite and ROI-audit bytes " +
        "were later replaced and are recorded as unavailable rather than reconstructed."
    )
}
[IO.Directory]::CreateDirectory((Split-Path -Parent $outputPath)) | Out-Null
$temporary = "$outputPath.tmp"
$verification | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
Move-Item -LiteralPath $temporary -Destination $outputPath
Get-Evidence $outputPath
