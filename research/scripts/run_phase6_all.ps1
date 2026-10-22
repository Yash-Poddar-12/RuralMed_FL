# Phase 6 — Joint Experiment Sweep (30 rounds)
param (
    [int]$Rounds = 30,
    [int]$Seed = 42,
    [switch]$NoTracking = $true,
    [switch]$SkipExisting = $true,
    [int]$MaxRuns = 0
)

$ErrorActionPreference = "Stop"
$env:RURALMED_DATA_ROOT = "C:\Users\pavan\Documents\Programming\College\Sem7\DL\RuralMed_FL\data"

$python = ".\research\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "Phase 6 — Joint Experiment Sweep" -ForegroundColor Green
Write-Host "  Rounds       : $Rounds" -ForegroundColor Green
Write-Host "  Seed         : $Seed" -ForegroundColor Green
Write-Host "  Skip Existing: $SkipExisting" -ForegroundColor Green
Write-Host "  No Tracking  : $NoTracking" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

$cmdArgs = @(
    "-m", "research.src.eval.experiment_runner",
    "--config", "research/configs/phase6_core_joint_sweep.yaml",
    "--output-dir", "results/phase6/raw",
    "--rounds", "$Rounds",
    "--seed", "$Seed"
)

if ($NoTracking) { $cmdArgs += "--no-tracking" }
if ($SkipExisting) { $cmdArgs += "--skip-existing" }
if ($MaxRuns -gt 0) { $cmdArgs += @("--max-runs", "$MaxRuns") }

& $python @cmdArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 6 experiment runner failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "Generating Phase 6 summaries, heatmaps and operating point..." -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
& $python -m research.src.eval.joint_metric_plots `
    --results-dir results/phase6/raw `
    --output-dir results/phase6

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "Generating Phase 6 Supabase batch payload..." -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
& $python -m research.src.eval.supabase_writer `
    --results-dir results/phase6/raw `
    --batch-json results/phase6/supabase_batch.json

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "All Phase 6 experiments and summaries completed successfully!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
