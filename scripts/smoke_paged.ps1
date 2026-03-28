# Smoke test: run paged_kvcomm on a SINGLE GSM8K sample (Windows).
# Usage:
#   .\scripts\smoke_paged.ps1
#   .\scripts\smoke_paged.ps1 -Model "Qwen/Qwen3.5-2B"
param(
    [string]$Model = "meta-llama/Llama-3.1-8B-Instruct",
    [string]$ResultsDir = "",
    [switch]$UseLocalReference
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if (-not $ResultsDir) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $ResultsDir = Join-Path $ProjectRoot "result\smoke_$ts"
}

# Activate conda environment
conda activate cs590proj

$env:PYTHONPATH = "$ProjectRoot;$($env:PYTHONPATH)"

Write-Host "=== Smoke Test: paged_kvcomm x 1 sample ===" -ForegroundColor Cyan
Write-Host "Model:       $Model"
Write-Host "Results dir: $ResultsDir"
Write-Host "Start time:  $(Get-Date)"
Write-Host ""

# Create single-sample dataset
$FullDataset = Join-Path $ProjectRoot "datasets\gsm8k\gsm8k.jsonl"
$MiniDataset = Join-Path $env:TEMP "gsm8k_1sample_$PID.jsonl"
[System.IO.File]::WriteAllText($MiniDataset, (Get-Content $FullDataset -TotalCount 1) + "`n", [System.Text.UTF8Encoding]::new($false))
Write-Host "Mini dataset: $MiniDataset"

try {
    $LocalRefArg = @()
    if ($UseLocalReference) { $LocalRefArg = @("--use-local-reference") }

    python experiments/run_gsm8k.py `
        --llm_name $Model `
        --method paged_kvcomm `
        --agent_nums 3 `
        --dataset_json $MiniDataset `
        --output_dir "$ResultsDir\gsm8k\paged_kvcomm" `
        @LocalRefArg

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
    } else {
        Write-Host "SUCCESS" -ForegroundColor Green
    }
} finally {
    Remove-Item $MiniDataset -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "End time: $(Get-Date)"
