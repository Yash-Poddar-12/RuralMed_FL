# Run from the repository root using an available Python with the host audit deps.
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = (Get-Command python).Source
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'research\artifacts') -Force | Out-Null
Start-Process -FilePath $taskPython -ArgumentList @('-u', 'research/scripts/resume_phase1.py') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput "$taskRoot\research\artifacts\continuation-launch.log" -RedirectStandardError "$taskRoot\research\artifacts\continuation-launch-errors.log" -PassThru | Select-Object Id
