# Phase 5 — Run all remaining bandwidth sweep experiments (30 rounds each)
$ErrorActionPreference = "Stop"
$env:RURALMED_DATA_ROOT = "C:\Users\pavan\Documents\Programming\College\Sem7\DL\RuralMed_FL\data"

$runs = @(
    "bw_1024kbps_topk1pct",
    "bw_512kbps_topk1pct",
    "bw_256kbps_topk1pct",
    "bw_128kbps_topk1pct",
    "bw_64kbps_topk1pct",
    "bw_256kbps_topk10pct",
    "bw_256kbps_uncompressed"
)

$total = $runs.Count
$idx = 1

foreach ($runName in $runs) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "[$idx/$total] Starting run: $runName (30 rounds)" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""

    python research/scripts/run_phase5.py `
        --config research/configs/phase5_bandwidth_sweep.yml `
        --run-name $runName `
        --rounds 30

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Run $runName failed with exit code $LASTEXITCODE. Stopping batch." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "[$idx/$total] Completed: $runName" -ForegroundColor Green
    $idx++
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "All Phase 5 experiments completed successfully!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
