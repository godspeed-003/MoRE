# _launch_mor_retrain.ps1 -- local helper, untracked. Retrains the MoR language arm.
#
# WHAT THIS RECOVERS AND WHAT IT DOES NOT REPLACE
# The five canonical MoR cells on disk carry metrics.json but NO checkpoint.pt:
# they were trained on the first rented V100 and the weights were never copied off
# before the instance was released (RUNBOOK_ACL_4060.md section 1). Every evaluation worth
# adding to the paper -- held-out test split, difficulty-stratified loss, OOD
# perplexity -- is a CHECKPOINT evaluation, so the MoR arm cannot take part in any
# of them until its weights exist again. This run recovers weights. The published
# val/task_loss mean (3.5202 +/- 0.0406) is the reproduction target, not a number
# being replaced.
#
# WHY --force IS REQUIRED, AND WHY OMITTING IT IS A SILENT NO-OP
# run_language_matrix.py:finished_run() treats a cell as complete on
# (metrics.json + resolved_config.json + canonical group + arch + seed + non-null
# val/task_loss). A checkpoint is NOT part of that test. The V100 MoR cells satisfy
# every clause, so a plain `--arch mor` prints
#     skip mor seed 42 -- already complete: langB_MoR_seed42__547e435b__r4
# for all five seeds and exits 0 having trained nothing. Verified by --dry-run
# before this script was written. RUNBOOK_ACL_4060.md section 4 says to launch without
# --force; that instruction is wrong for the weights-only case and following it
# costs a day of the window.
#
# WHY --force AND NOT `git mv`-FIRST (which section 4 implies)
# Archiving the V100 cells first would also unblock the runner, but it removes the
# admitted MoR arm from runs/ for the ~15 h the retrain takes, during which the
# exporter cannot regenerate the published table at all. --force writes the new
# cells alongside, so the provenance record stays live until a replacement exists.
# The duplicate-cell state this creates is EXPECTED and must be resolved AFTER the
# run completes: export_results.py refuses the whole export with
# `duplicate cell ('mor', 42)` while two canonical cells share one (arch, seed).
# Resolve with `git mv` -- never plain `mv`, which leaves the directory in the index
# at the old path so the commit carries the cell at BOTH paths and the duplication
# only surfaces on the next machine to pull (RUNBOOK_ACL_4060.md section 4, T-LX.19).
#
# DO NOT change batch_size, lr, weight_decay, dropout or epochs to use spare VRAM.
# Every one is frozen in canonical_spec_language.json and moves config_hash, which
# makes the new cells incomparable with the kept MoE/MoRE cells (T-L7.0/T-L7.1).
#
# Estimate: run_language_matrix.py's own HOURS table gives mor 1.03 h/epoch x 3
# epochs x 5 seeds ~= 15.5 h, measured on a 6 GB 3050 and labelled an upper bound
# for this 8 GB 4060. RUNBOOK_ACL_4060.md section 2 budgets ~20 h. Expect 13-20 h.

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
# W&B offline: preflight REFUSES to start otherwise ("no key and no WANDB_MODE"),
# because the first run would otherwise die in wandb.init after the guard passed.
# metrics.json is written either way; W&B is convenience, not the record.
$env:WANDB_MODE   = 'offline'
$env:WANDB_SILENT = 'true'

$repo = 'C:\Users\Hp\Desktop\Waste\MoRE'
$py   = 'C:\Users\Hp\anaconda3\envs\more_env\python.exe'
$log  = Join-Path $repo 'runs\_mor_retrain_console.log'

Set-Location $repo

Write-Host ''
Write-Host '=============================================================' -ForegroundColor Cyan
Write-Host ' MoR LANGUAGE RETRAIN -- 5 seeds, canonical_lang_b, --force' -ForegroundColor Cyan
Write-Host ' purpose: recover checkpoints (weights), not replace numbers' -ForegroundColor Cyan
Write-Host ' reproduction target: val/task_loss 3.5202 +/- 0.0406' -ForegroundColor Cyan
Write-Host ' estimate: 13-20 h. Leave this tab open.' -ForegroundColor Cyan
Write-Host (' started: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
Write-Host '=============================================================' -ForegroundColor Cyan
Write-Host ''

& $py -u (Join-Path $repo 'code\run_language_matrix.py') --arch mor --force 2>&1 |
    Tee-Object -FilePath $log

Write-Host ''
Write-Host '=============================================================' -ForegroundColor Green
Write-Host (' MoR retrain finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Green
Write-Host '=============================================================' -ForegroundColor Green
Write-Host ''
Write-Host ' NEXT, in order:' -ForegroundColor Yellow
Write-Host '  1. Reproduction gate. New mean inside +/-1 published std (0.0406) is a' -ForegroundColor Yellow
Write-Host '     reproduction; OUTSIDE +/-2 std, STOP -- that is a config/data/code' -ForegroundColor Yellow
Write-Host '     difference, not numerics (RUNBOOK_ACL_4060.md section 5).' -ForegroundColor Yellow
Write-Host '  2. Retire the 5 superseded V100 MoR cells with `git mv` into' -ForegroundColor Yellow
Write-Host '     archive/pre_finalization/lang_v100_no_checkpoints/, then confirm the' -ForegroundColor Yellow
Write-Host '     deletions are staged: git status --short | grep "^D"' -ForegroundColor Yellow
Write-Host '  3. Re-export: code/export_results.py --task language' -ForegroundColor Yellow
Write-Host (' console log: ' + $log) -ForegroundColor DarkGray
Write-Host ''
