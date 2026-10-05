# Laptop storage inventory and cleanup commands

Measured 2026-10-05 by a read-only file-size inventory. The unrestricted inventory could read all directories (the first sandbox pass could not read six pytest folders). Values are logical file sizes, rounded to GiB; uv cache and virtual-environment files have hard links (verified on torch_cpu.dll), so summing them overstates physical space reclaimed if only one link is deleted.

| Location | GiB | Disposition |
|---|---:|---|
| `.cache/uv` | 6.333 | Regenerable package cache; shares some files with the venv |
| `.cache/wheels` | 2.530 | Regenerable CUDA wheel |
| `.cache/npm` | 0.206 | Regenerable npm cache |
| `.cache/node` | 0.135 | Regenerable Node runtime; delete if willing to reinstall it |
| `.cache` test fixtures | 0.094 | Regenerable test outputs; includes the old larger synthetic fixtures |
| `.cache` remaining scripts/probes | <0.001 | Disposable setup helpers; the worker launcher is now inert |
| `research/.venv` | 5.397 | Rebuild from requirements/environment spec; optional environment reset |
| `research/data/raw` | 5.553 | Existing COVID release, NIH pilot and partial transfer; teammate reacquires full data |
| `research/data/processed` | 0.861 | Old processed pilot; regenerable |
| `research/data/partitions/representative` | 0.004 | Old pilot indices; aggregate pilot evidence remains in Git |
| `research/data/dev_sample` | 0.008 | **Keep**: 300 real processed images and tiny indices, about 8.1 MiB |
| `research/artifacts` | <0.001 | Only Phase 1 setup/dev tracking receipts; aggregate verification is checked in |
| `web/frontend/node_modules` | 0.303 | Optional; regenerate with npm ci |
| `web/backend/node_modules` | 0.002 | Optional; regenerate with npm ci |

Totals: `.cache` about 9.30 GiB; `research` about 11.82 GiB including the venv and 6.42 GiB of data. No downloads, full-data processing or cleanup deletes were performed in this session. The sample uses independent saved PNG files and does not need raw inputs after creation.

## Remove caches and old local data; keep the dev sample and environment

These are commands for **you to run**, not commands already executed. They remove only the named regenerable directories. No parent `research/data` deletion appears. Tracked `.gitkeep` files, committable full indices, source/config files, `.env` and dev_sample are preserved. Read the optional environment reset below before deciding whether to remove the Node runtime too.

Run in PowerShell. The block resolves and checks every existing absolute target against the repository before using native `Remove-Item -LiteralPath`. It also refuses to remove dev_sample, its parents or children.

```powershell
$cleanupRepo = (Resolve-Path -LiteralPath 'D:\Projects_sem_7\RuralMed_FL').Path
$keepDev = (Resolve-Path -LiteralPath 'D:\Projects_sem_7\RuralMed_FL\research\data\dev_sample').Path
$cleanupTargets = @(
    'D:\Projects_sem_7\RuralMed_FL\.cache\uv',
    'D:\Projects_sem_7\RuralMed_FL\.cache\wheels',
    'D:\Projects_sem_7\RuralMed_FL\.cache\npm',
    'D:\Projects_sem_7\RuralMed_FL\.cache\source-probes',
    'D:\Projects_sem_7\RuralMed_FL\research\data\raw\covidqu',
    'D:\Projects_sem_7\RuralMed_FL\research\data\raw\nih',
    'D:\Projects_sem_7\RuralMed_FL\research\data\raw\nih-representative',
    'D:\Projects_sem_7\RuralMed_FL\research\data\processed\representative',
    'D:\Projects_sem_7\RuralMed_FL\research\data\partitions\representative',
    'D:\Projects_sem_7\RuralMed_FL\research\artifacts'
)
$cleanupTargets += Get-ChildItem -LiteralPath 'D:\Projects_sem_7\RuralMed_FL\.cache' -Directory |
    Where-Object { $_.Name -like 'pytest*' -or $_.Name -like 'test-tmp*' -or $_.Name -like 'tests-dev*' } |
    Select-Object -ExpandProperty FullName
foreach ($cleanupTarget in $cleanupTargets) {
    if (Test-Path -LiteralPath $cleanupTarget) {
        $resolvedTarget = (Resolve-Path -LiteralPath $cleanupTarget).Path
        if (-not $resolvedTarget.StartsWith($cleanupRepo + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Outside repository: $resolvedTarget"
        }
        if ($resolvedTarget -eq $keepDev -or
            $resolvedTarget.StartsWith($keepDev + '\', [StringComparison]::OrdinalIgnoreCase) -or
            $keepDev.StartsWith($resolvedTarget + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Would affect dev sample: $resolvedTarget"
        }
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force -ErrorAction Stop
    }
}
```

This removes about 6.42 GiB of raw/pilot data plus unshared cache bytes. Some of the 6.33 GiB uv cache stays allocated through hard links in the retained venv; the separate 2.53 GiB wheel can be reclaimed. Deleting Phase 1 tracking receipts clears only local dev/setup run history; [committed dev verification](reports/phase1-dev/verification.json) remains available. Keeping dev_sample allows the offline CLI and unit tests to work without raw sources. No source registration or full-data downloads should be restarted on this laptop.

## Optional: remove all remaining caches and rebuildable environments

Use this second block if you also want to discard the Python venv, custom Node runtime, web dependencies and remaining cache helper files. It frees the remaining shared package allocations once their final hard links are removed. The globally installed host Python used by the dev tests is outside this repo and remains available. Later restore the research environment from its specs, and Node/web packages from `.nvmrc` and the npm lockfiles on a machine with sufficient space.

```powershell
$cleanupRepo = (Resolve-Path -LiteralPath 'D:\Projects_sem_7\RuralMed_FL').Path
$cleanupTargets = @(
    'D:\Projects_sem_7\RuralMed_FL\.cache',
    'D:\Projects_sem_7\RuralMed_FL\research\.venv',
    'D:\Projects_sem_7\RuralMed_FL\web\frontend\node_modules',
    'D:\Projects_sem_7\RuralMed_FL\web\backend\node_modules'
)
foreach ($cleanupTarget in $cleanupTargets) {
    if (Test-Path -LiteralPath $cleanupTarget) {
        $resolvedTarget = (Resolve-Path -LiteralPath $cleanupTarget).Path
        if (-not $resolvedTarget.StartsWith($cleanupRepo + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Outside repository: $resolvedTarget"
        }
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force -ErrorAction Stop
    }
}
```

Optional bytecode cleanup, confined to code directories (never the retained sample):

```powershell
$cleanupRepo = (Resolve-Path -LiteralPath 'D:\Projects_sem_7\RuralMed_FL').Path
$bytecodeRoots = @(
    'D:\Projects_sem_7\RuralMed_FL\research\src',
    'D:\Projects_sem_7\RuralMed_FL\research\scripts',
    'D:\Projects_sem_7\RuralMed_FL\research\tests'
)
foreach ($bytecodeRoot in $bytecodeRoots) {
    $bytecodeTargets = @(Get-ChildItem -LiteralPath $bytecodeRoot -Directory -Recurse -Filter '__pycache__')
    foreach ($bytecodeTarget in $bytecodeTargets) {
        $resolvedTarget = (Resolve-Path -LiteralPath $bytecodeTarget.FullName).Path
        if (-not $resolvedTarget.StartsWith($cleanupRepo + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Outside repository: $resolvedTarget"
        }
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force -ErrorAction Stop
    }
}
```

Do not delete the retained sample, `.git`, source/config/requirements files, web lockfiles, phase specifications, committed reports or `.env` as part of this cleanup. The two main blocks together target about 21.4 GiB of logical files; physical reclaimed space is smaller because of the verified package hard links. Exact physical free space can be read before/after your cleanup with `Get-PSDrive -Name D`.
