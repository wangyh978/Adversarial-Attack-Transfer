param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("nsl_kdd", "unsw_nb15")]
    [string]$Dataset = "nsl_kdd",

    [Parameter(Mandatory=$false)]
    [string[]]$TargetModels = @(),

    [Parameter(Mandatory=$false)]
    [int]$SeedSize = 1000,

    [Parameter(Mandatory=$false)]
    [double]$Alpha = 0.1,

    [Parameter(Mandatory=$false)]
    [int]$Depth = 3,

    [Parameter(Mandatory=$false)]
    [string[]]$Attacks = @("fgm", "pgd", "slide"),

    [switch]$IncludePreparation,
    [switch]$IncludeTargetTraining,
    [switch]$IncludeSurrogateTraining,
    [switch]$ReuseExistingArtifacts
)

$ErrorActionPreference = "Stop"

function Resolve-DefaultTargets {
    param([string]$Dataset)
    switch ($Dataset) {
        "nsl_kdd"   { return @("tabnet", "xgb", "gbdt") }
        "unsw_nb15" { return @("xgb") }
        default     { throw "Unsupported dataset: $Dataset" }
    }
}

function Resolve-LabelMode {
    param([string]$Dataset)
    switch ($Dataset) {
        "nsl_kdd"   { return "5class" }
        "unsw_nb15" { return "multiclass" }
        default     { throw "Unsupported dataset: $Dataset" }
    }
}

function Invoke-Step {
    param([string]$Command)
    Write-Host ""
    Write-Host ">> $Command" -ForegroundColor Cyan
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command"
    }
}

function Invoke-TargetTraining {
    param([string]$Dataset, [string]$TargetModel)

    switch ($TargetModel) {
        "xgb" {
            Invoke-Step "python -m src.models.train_xgb --dataset $Dataset"
        }
        "gbdt" {
            Invoke-Step "python -m src.models.train_gbdt --dataset $Dataset"
        }
        "tabnet" {
            Invoke-Step "python -m src.models.train_tabnet --dataset $Dataset"
        }
        "random_forest" {
            Invoke-Step "python -m src.models.train_sklearn_baseline --dataset $Dataset --model random_forest"
        }
        default {
            throw "Unsupported target model for training: $TargetModel"
        }
    }
}

if ($TargetModels.Count -eq 0) {
    $TargetModels = Resolve-DefaultTargets -Dataset $Dataset
}
$LabelMode = Resolve-LabelMode -Dataset $Dataset

if (-not $ReuseExistingArtifacts `
    -and -not $IncludePreparation `
    -and -not $IncludeTargetTraining `
    -and -not $IncludeSurrogateTraining) {

    $IncludeTargetTraining = $true
    $IncludeSurrogateTraining = $true

    Write-Host ""
    Write-Host "[safe-default] No include switches were provided." -ForegroundColor Yellow
    Write-Host "[safe-default] Automatically enabling target + surrogate retraining." -ForegroundColor Yellow
    Write-Host "[safe-default] Use -ReuseExistingArtifacts if you really want to reuse old checkpoints." -ForegroundColor Yellow
}

Write-Host "Dataset      : $Dataset" -ForegroundColor Yellow
Write-Host "TargetModels : $($TargetModels -join ', ')" -ForegroundColor Yellow
Write-Host "SeedSize     : $SeedSize" -ForegroundColor Yellow
Write-Host "Alpha        : $Alpha" -ForegroundColor Yellow
Write-Host "Depth        : $Depth" -ForegroundColor Yellow
Write-Host "Attacks      : $($Attacks -join ', ')" -ForegroundColor Yellow
Write-Host "Preparation  : $IncludePreparation" -ForegroundColor Yellow
Write-Host "TrainTarget  : $IncludeTargetTraining" -ForegroundColor Yellow
Write-Host "TrainSurr    : $IncludeSurrogateTraining" -ForegroundColor Yellow
Write-Host "ReuseOnly    : $ReuseExistingArtifacts" -ForegroundColor Yellow

if ($IncludePreparation) {
    Invoke-Step "python -m src.data.load_raw --dataset $Dataset"
    Invoke-Step "python -m src.data.clean_labels --dataset $Dataset --mode $LabelMode"
    Invoke-Step "python -m src.data.split_data --dataset $Dataset"
    Invoke-Step "python -m src.preprocess.run_preprocess_pipeline --dataset $Dataset"
}

foreach ($target in $TargetModels) {
    Write-Host ""
    Write-Host "==== Target: $target ====" -ForegroundColor Magenta

    if ($IncludeTargetTraining) {
        Invoke-TargetTraining -Dataset $Dataset -TargetModel $target
    }

    if ($IncludeSurrogateTraining) {
        Invoke-Step "python -m src.data.build_seed_set --dataset $Dataset --seed_size $SeedSize"
        Invoke-Step "python -m src.data.query_seed_labels --dataset $Dataset --target_model $target --seed_size $SeedSize"
        Invoke-Step "python -m src.augment.run_mixup --dataset $Dataset --target_model $target --seed_size $SeedSize --alpha $Alpha"
        Invoke-Step "python -m src.data.build_surrogate_trainset --dataset $Dataset --target_model $target --seed_size $SeedSize --alpha $Alpha"
        Invoke-Step "python -m src.models.train_surrogate_mlp --dataset $Dataset --target_model $target --seed_size $SeedSize --alpha $Alpha --depth $Depth"
        Invoke-Step "python -m src.evaluation.evaluate_surrogate --dataset $Dataset --target_model $target --seed_size $SeedSize --alpha $Alpha --depth $Depth"
    }

    foreach ($attack in $Attacks) {
        Invoke-Step "python -m src.transfer.generate_from_surrogate --dataset $Dataset --target_model $target --seed_size $SeedSize --alpha $Alpha --depth $Depth --attack $attack"
        Invoke-Step "python -m src.transfer.attack_target --dataset $Dataset --target_model $target --seed_size $SeedSize --alpha $Alpha --depth $Depth --attack $attack"
    }

    Invoke-Step "python scripts/summarize_transfer_matrix.py --dataset $Dataset --target-model $target"
}

Write-Host ""
Write-Host "Full attack matrix pipeline finished." -ForegroundColor Green
