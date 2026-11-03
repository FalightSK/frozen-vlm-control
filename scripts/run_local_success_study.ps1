[CmdletBinding()]
param(
    [ValidateSet('aloha', 'libero')]
    [string]$Dataset = 'aloha',
    [switch]$WaitForCachePreparation,
    [string]$Python = $env:ACTIONOMNI_BASE_PYTHON,
    [string]$LiberoPython = $env:ACTIONOMNI_LIBERO_PYTHON,
    [string]$StudyPlan = 'configs/runpod_fresh_cached_success_study.json',
    [string]$StudyId,
    [switch]$ThenLibero
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$retentionWorker = $null
Push-Location $repo
try {
    if (-not $Python) { throw 'Pass -Python or set ACTIONOMNI_BASE_PYTHON to a CUDA-enabled Python interpreter.' }
    $python = (Get-Command $Python -ErrorAction Stop).Source
    if (-not $LiberoPython) {
        $condaEnvs = Split-Path (Split-Path $python -Parent) -Parent
        $candidate = Join-Path $condaEnvs 'vla_libero/python.exe'
        $LiberoPython = if (Test-Path $candidate) { $candidate } else { $python }
    }
    $liberoPython = (Get-Command $LiberoPython -ErrorAction Stop).Source
    if ($Dataset -eq 'libero' -or $ThenLibero) {
        & $liberoPython -c 'import libero, robosuite, mujoco'
        if ($LASTEXITCODE -ne 0) { throw "LIBERO simulator dependencies are unavailable in $liberoPython" }
    }
    $overlay = Join-Path $repo '.venv/Lib/site-packages'
    if (-not (Test-Path (Join-Path $overlay 'transformers'))) {
        throw 'The local compatibility overlay is missing; create .venv and install the pinned Transformers/Hugging Face Hub versions first.'
    }
    $studyId = if ($StudyId) { $StudyId } else { "${Dataset}_progression_v1" }
    $cacheStudyId = "runpod_success_stopped_v6_$Dataset"
    $durable = Join-Path $repo "asset/durable/$Dataset"
    $runRoot = Join-Path $repo "asset/runs/$studyId"
    $statusPath = Join-Path $durable 'local_launcher_status.json'
    $logPath = Join-Path $durable 'local_launcher.log'
    $cacheReport = Join-Path $durable "success_studies/$cacheStudyId/cache_prepare_$Dataset.json"
    $cachePlan = 'configs/runpod_cached_success_study.json'
    $studyPlan = $StudyPlan
    $plan = Get-Content $studyPlan -Raw | ConvertFrom-Json
    $datasetPlan = $plan.datasets.PSObject.Properties[$Dataset].Value
    $stopping = @{}
    foreach ($item in $plan.stopping.PSObject.Properties) {
        $stopping[$item.Name] = $item.Value
    }
    foreach ($item in $datasetPlan.stopping.PSObject.Properties) {
        $stopping[$item.Name] = $item.Value
    }

    New-Item -ItemType Directory -Force $durable | Out-Null
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $env:MUJOCO_GL = 'glfw'
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$overlay;$env:PYTHONPATH" } else { $overlay }

    function Set-LauncherStatus([string]$Stage, [hashtable]$Fields = @{}) {
        $state = @{ stage = $Stage; dataset = $Dataset; study_id = $studyId;
                    updated_utc = [DateTime]::UtcNow.ToString('o') }
        foreach ($key in $Fields.Keys) { $state[$key] = $Fields[$key] }
        $temporary = "$statusPath.$PID.tmp"
        $state | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 $temporary
        for ($attempt = 1; $attempt -le 7; $attempt++) {
            try {
                Move-Item -LiteralPath $temporary -Destination $statusPath -Force -ErrorAction Stop
                return
            } catch {
                if ($_.Exception -isnot [System.IO.IOException] -and
                    $_.Exception -isnot [System.UnauthorizedAccessException]) { throw }
                if ($attempt -eq 7) {
                    Write-Warning "Could not refresh advisory launcher status '$statusPath' after 7 attempts: $($_.Exception.Message)"
                    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
                    return
                }
                Start-Sleep -Milliseconds ([Math]::Min(50 * [Math]::Pow(2, $attempt - 1), 500))
            }
        }
    }

    function Invoke-PythonStep([string]$Script, [string[]]$Arguments, [string]$Stage) {
        Set-LauncherStatus $Stage @{ command = @($Script) + $Arguments; log = $logPath }
        & $python -u $Script @Arguments 2>&1 | Tee-Object -Append -FilePath $logPath
        if ($LASTEXITCODE -ne 0) { throw "$Stage failed with exit code $LASTEXITCODE" }
    }

    Set-LauncherStatus 'starting' @{ durable_root = $durable; run_root = $runRoot }
    if ($WaitForCachePreparation) {
        Set-LauncherStatus 'waiting_for_feature_caches' @{ cache_report = $cacheReport }
        while (-not (Test-Path $cacheReport)) {
            $latestCacheLog = Get-ChildItem $durable -Filter "cache_prepare_$Dataset*.log" -File |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestCacheLog -and (Select-String -Path $latestCacheLog.FullName -SimpleMatch '"event": "cached_study_prepare.failed"' -Quiet)) {
                throw "Feature-cache preparation failed; see $($latestCacheLog.FullName)"
            }
            Start-Sleep -Seconds 20
        }
    } else {
        Invoke-PythonStep 'scripts/prepare_runpod_study.py' @(
            '--dataset', $Dataset, '--plan', $studyPlan,
            '--durable-root', $durable, '--local-assets') 'preparing_local_assets'
        Invoke-PythonStep 'scripts/prepare_cached_study.py' @(
            '--dataset', $Dataset, '--durable-root', $durable,
            '--study-id', $cacheStudyId, '--plan', $cachePlan) 'preparing_feature_caches'
    }

    if (-not (Test-Path $cacheReport)) { throw "Verified cache report missing: $cacheReport" }
    $retentionLog = Join-Path $durable 'checkpoint_retention_worker.log'
    $retentionError = Join-Path $durable 'checkpoint_retention_worker_error.log'
    $retentionArgs = @('-u', 'scripts/manage_checkpoint_retention.py',
        '--dataset', $Dataset, '--run-root', $runRoot, '--durable-root', $durable,
        '--study-id', $studyId, '--minimum-epochs', [string]$stopping.minimum_epochs,
        '--evaluation-every-epochs', [string]$stopping.evaluation_every_epochs,
        '--poll-seconds', '10')
    $retentionWorker = Start-Process -FilePath $python -ArgumentList $retentionArgs `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $retentionLog -RedirectStandardError $retentionError
    $evalLog = Join-Path $durable 'local_evaluator.log'
    $evalError = Join-Path $durable 'local_evaluator_error.log'
    $evalArgs = @('-u', 'scripts/runpod_eval_worker.py', '--durable-root', $durable,
        '--study-id', $studyId,
        '--assets-root', (Join-Path $repo 'asset'), '--aloha-python', $python,
        '--libero-python', $liberoPython, '--aloha-gl', 'glfw', '--poll-seconds', '15',
        '--work-root', (Join-Path $repo 'asset/evaluation_scratch'))
    $evaluator = Start-Process -FilePath $python -ArgumentList $evalArgs -WorkingDirectory $repo `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $evalLog -RedirectStandardError $evalError

    $runnerExit = 1
    try {
        Invoke-PythonStep 'scripts/runpod_success_study.py' @(
            '--dataset', $Dataset, '--plan', $studyPlan, '--run-root', $runRoot,
            '--durable-root', $durable, '--local', '--parallel-policies', '1', '--workers', '1') 'training'
        $runnerExit = 0

        Set-LauncherStatus 'waiting_for_final_evaluations' @{ run_root = $runRoot; evaluator_pid = $evaluator.Id }
        $taskRoot = Join-Path $durable 'evaluation_tasks'
        $statusRoot = Join-Path $durable 'evaluation_status'
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            while ($true) {
                $tasks = @()
                if (Test-Path $taskRoot) {
                    $tasks = @(Get-ChildItem $taskRoot -Filter '*.task.json' -File -Recurse | ForEach-Object {
                        $task = Get-Content $_.FullName -Raw | ConvertFrom-Json
                        if ($task.study_id -eq $studyId) { $task }
                    })
                }
                $pending = @($tasks | Where-Object {
                    $hash = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($_.task_id))).Replace('-', '').ToLowerInvariant()
                    $path = Join-Path $statusRoot "$hash.json"
                    if (-not (Test-Path $path)) { return $true }
                    (Get-Content $path -Raw | ConvertFrom-Json).status -ne 'complete'
                })
                Set-LauncherStatus 'waiting_for_final_evaluations' @{
                    run_root = $runRoot; evaluator_pid = $evaluator.Id;
                    completed_tasks = $tasks.Count - $pending.Count; pending_tasks = $pending.Count
                }
                if ($pending.Count -eq 0) { break }
                Start-Sleep -Seconds 30
            }
        } finally { $sha.Dispose() }
    } finally {
        if ($evaluator -and -not $evaluator.HasExited) { Stop-Process -Id $evaluator.Id -Force }
    }

    if ($runnerExit -eq 0) {
        if ($ThenLibero -and $Dataset -eq 'aloha') {
            $liberoDurable = Join-Path $repo 'asset/durable/libero'
            $liberoStatus = Join-Path $liberoDurable 'local_launcher_status.json'
            Set-LauncherStatus 'aloha_complete_starting_libero' @{ run_root = $runRoot; libero_status = $liberoStatus }
            & $PSCommandPath -Dataset libero -Python $python -LiberoPython $liberoPython `
                -StudyPlan $studyPlan
            $child = Get-Content $liberoStatus -Raw | ConvertFrom-Json
            if ($child.stage -ne 'complete') { throw "LIBERO campaign ended at stage $($child.stage)" }
            Set-LauncherStatus 'complete' @{ run_root = $runRoot; durable_root = $durable; libero_status = $liberoStatus }
        } else {
            Set-LauncherStatus 'complete' @{ run_root = $runRoot; durable_root = $durable }
        }
    }
} catch {
    Set-LauncherStatus 'failed' @{ error = $_.Exception.Message; log = $logPath }
    Add-Content -Encoding utf8 $logPath "[$([DateTime]::UtcNow.ToString('o'))] ERROR: $($_.Exception.Message)"
    throw
} finally {
    if ($retentionWorker) {
        Get-Process -Id $retentionWorker.Id -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
