# Whole pipeline after planning: CARLA recording + novel-view re-simulation (one server session), PNGs, instance map,
# reconstruction dataset, NCore V4, exposure check, training, evaluation, report.
# Usage: .\run_pipeline.ps1 -Build C:\mywork\osm2carla\build\nurec -Tag 02 [-Iters 30000] [-SkipCarla] [-SkipTrain]
param([string]$Build = "C:\mywork\osm2carla\build\nurec", [string]$Tag = "02", [int]$Iters = 30000,
      [switch]$SkipCarla, [switch]$SkipTrain)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$rec = "$Build\rec$Tag"; $nov = "$Build\novel$Tag"; $ds = "$Build\ds$Tag"; $nc = "$Build\ncore$Tag"
$tr = "$Build\train$Tag"; $ev = "$Build\eval$Tag"
function Step($name, [scriptblock]$b) { $t = Get-Date; "== $name"; & $b; if ($LASTEXITCODE) { throw "$name failed ($LASTEXITCODE)" }; "   {0:N0} s" -f ((Get-Date) - $t).TotalSeconds }

if (-not $SkipCarla) {
    Step "CARLA: record + re-simulate" { .\run_carla.ps1 -Log "rec${Tag}_server" -JobTimeout 5400 -Jobs @(
        @("scenario\record.py", "$Build\plan.json", $rec, "--raw"),
        @("scenario\record.py", "$Build\plan.json", $nov, "--raw", "--novel", $rec, "--every", "4")) }
}
Step "PNG (recording)" { & $py convert\raw_to_png.py $rec }
Step "PNG (novel views)" { & $py convert\raw_to_png.py $nov }
Step "instance map" { & $py convert\instance_map.py $rec --every 10 }
Step "dataset" { & $py convert\to_dataset.py $rec $ds }
Step "NCore V4" { & $py convert\to_ncore.py $ds $nc }
Step "exposure check" { & $py evaluate\exposure_check.py $ds --every 20 }
if (-not $SkipTrain) {
    Step "train" { & $py recon\train.py $ds $tr --iters $Iters --cap-bg 1200000 --cap-actor 30000 --log-every 500 --exposure }
}
Step "evaluate" { & $py evaluate\evaluate.py $ds "$tr\ckpt.pt" $nov $ev }
Step "report" { & $py evaluate\report.py $ev docs --extra "$ds\exposure_check.json" "$Build\replay01\recorder_replay_fidelity.json" }
