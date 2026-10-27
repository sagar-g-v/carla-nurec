# CARLA server (-game, off-screen, no dialogs) -> client scripts in order -> server stopped (also on failure).
# Usage: .\run_carla.ps1 -Jobs @(,@("scenario\record.py", "plan.json", "out", "--probe")) [-Log name]
#        several jobs: -Jobs @(@("a.py", "x"), @("b.py", "y"))   (each job: script, then its arguments)
#   CARLA_UE_BIN / CARLA_PROJECT / CARLA_PY38 override the defaults below.
param([Parameter(Mandatory = $true)][object[]]$Jobs, [string]$Map = "LA_DTLA_ODR", [string]$Log = "server",
      [int]$JobTimeout = 7200)
$ErrorActionPreference = "Continue"
$ue = if ($env:CARLA_UE_BIN) { $env:CARLA_UE_BIN } else { "C:\mywork\git_repos\UnrealEngine5_carla\Engine\Binaries\Win64" }
$proj = if ($env:CARLA_PROJECT) { $env:CARLA_PROJECT } else { "C:\mywork\git_repos\CarlaUE5\Unreal\CarlaUnreal\CarlaUnreal.uproject" }
$py38 = if ($env:CARLA_PY38) { $env:CARLA_PY38 } else { "C:\mywork\osm2carla\venv38\Scripts\python.exe" }
$wait = "C:\mywork\osm2carla\scripts\carla_wait_server.py"
$logdir = Join-Path $PSScriptRoot "logs"; New-Item -ItemType Directory -Force $logdir | Out-Null

function Stop-CarlaServer {
    Get-Process UnrealEditor -ErrorAction SilentlyContinue | ForEach-Object {
        $c = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
        if ($c -match "-game" -and $c -match "carla-rpc-port") { Stop-Process -Id $_.Id -Force; "server stopped" } }
}

# load check: one Unreal process at a time, machine not busy
$other = Get-Process UnrealEditor, UnrealEditor-Cmd -ErrorAction SilentlyContinue
if ($other) { throw "an Unreal process is already running (pid $($other.Id -join ', ')) - not starting another" }
$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$gpu = (& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits) -join " "
"load before start: CPU $cpu %, free RAM $freeGB GB, GPU mem used/total MiB, util %: $gpu"
if ($cpu -gt 60 -or $freeGB -lt 16) { throw "machine busy (CPU $cpu %, free RAM $freeGB GB) - not starting" }

$t0 = Get-Date
try {
    Start-Process -FilePath "$ue\UnrealEditor.exe" -ArgumentList @("`"$proj`"", "/Game/Carla/Maps/$Map/$Map", "-game",
        "-RenderOffScreen", "-unattended", "-carla-rpc-port=2000", "-log", "-abslog=$logdir\$Log.log") | Out-Null
    & $py38 $wait --limit 1800
    if ($LASTEXITCODE) { throw "CARLA server not ready" }
    foreach ($job in $Jobs) {
        $job = @($job)
        "--- job: $($job -join ' ')"
        # the client runs as its own process with a time limit (a hung client must not keep the server alive)
        $argList = @("-u") + @($job | ForEach-Object { "`"$_`"" })
        $p = Start-Process -FilePath $py38 -ArgumentList $argList -NoNewWindow -PassThru `
            -RedirectStandardOutput "$logdir\$Log.job.out" -RedirectStandardError "$logdir\$Log.job.err"
        if (-not $p.WaitForExit($JobTimeout * 1000)) { Stop-Process -Id $p.Id -Force; "job timed out after $JobTimeout s" }
        Get-Content "$logdir\$Log.job.out"; Get-Content "$logdir\$Log.job.err" | Select-String -NotMatch "went out of the scope|^\(sensor"
        "client rc=$($p.ExitCode)"
    }
}
finally {
    Stop-CarlaServer
    "wall time {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
}
