# _launch_gate.ps1 -- local helper, untracked. Runs the correctness suite.
#
# WHY A SCRIPT AND NOT A TYPED COMMAND. The Terminal-panel tool types ONE literal
# command line into a fresh PowerShell tab, so `export VAR=... && python ...`
# (Git Bash syntax) cannot be used and env vars set in one tab do not persist to
# the next. PYTHONIOENCODING=utf-8 is NOT optional: without it the suite raises
# UnicodeEncodeError under cp1252 on the arrow and +/- characters
# (PAPER_GUIDE.md section 1).
#
# Gate contract: RUNBOOK_ACL_4060.md section 4 -- expect TOTAL 356 356 0 0. If this is
# not 356/356 the matrix numbers are not trustworthy and nothing else matters
# (CLAUDE.md section 7: STOP, report, diagnose, fix, re-run).

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$repo = 'C:\Users\Hp\Desktop\Waste\MoRE'
$py   = 'C:\Users\Hp\anaconda3\envs\more_env\python.exe'
$log  = Join-Path $repo 'runs\_gate_console.log'

Set-Location $repo

Write-Host ''
Write-Host '=============================================================' -ForegroundColor Cyan
Write-Host ' GATE L0 -- code/run_correctness_suite.py' -ForegroundColor Cyan
Write-Host ' expect: TOTAL 356 356 0 0' -ForegroundColor Cyan
Write-Host (' started: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
Write-Host '=============================================================' -ForegroundColor Cyan
Write-Host ''

& $py -u (Join-Path $repo 'code\run_correctness_suite.py') 2>&1 |
    Tee-Object -FilePath $log

Write-Host ''
Write-Host (' finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
Write-Host (' console log: ' + $log) -ForegroundColor DarkGray
Write-Host ''
