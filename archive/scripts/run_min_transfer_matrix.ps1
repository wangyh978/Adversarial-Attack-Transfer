param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("nsl_kdd", "unsw_nb15")]
    [string]$Dataset = "nsl_kdd",

    [Parameter(Mandatory=$false)]
    [string]$TargetModel = "",

    [Parameter(Mandatory=$false)]
    [int]$SeedSize = 1000,

    [Parameter(Mandatory=$false)]
    [double]$Alpha = 0.1,

    [Parameter(Mandatory=$false)]
    [int]$Depth = 3,

    [Parameter(Mandatory=$false)]
    [string[]]$Attacks = @("fgm", "pgd"),

    [switch]$SkipDataPreparation,
    [switch]$SkipTargetTraining,
    [switch]$SkipSurrogateTraining
)

$ErrorActionPreference = "Stop"

function Resolve-TargetModel {
    param([string]$Dataset, [string]$TargetModel)
    if ($TargetModel -and $TargetModel.Trim().Length -gt 0) {
        return $TargetModel
    }
    switch ($Dataset) {
        "nsl_kdd"   { return "tabnet" }
        "unsw_nb15" { return "xgb" }
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

$TargetModel = Resolve-TargetModel -Dataset $Dataset -TargetModel $TargetModel
$LabelMode = Resolve-LabelMode -Dataset $Dataset

Write-Host "Dataset      : $Dataset" -ForegroundColor Yellow
Write-Host "TargetModel  : $TargetModel" -ForegroundColor Yellow
Write-Host "SeedSize     : $SeedSize" -ForegroundColor Yellow
Write-Host "Alpha        : $Alpha" -ForegroundColor Yellow
Write-Host "Depth        : $Depth" -ForegroundColor Yellow
Write-Host "Attacks      : $($Attacks -join ', ')" -ForegroundColor Yellow

if (-not $SkipDataPreparation) {
    Invoke-Step "python -m src.data.load_raw --dataset $Dataset"
    Invoke-Step "python -m src.data.clean_labels --dataset $Dataset --mode $LabelMode"
    Invoke-Step "python -m src.data.split_data --dataset $Dataset"
    Invoke-Step "python -m src.preprocess.run_preprocess_pipeline --dataset $Dataset"
}

if (-not $SkipTargetTraining) {
    Invoke-TargetTraining -Dataset $Dataset -TargetModel $TargetModel
    Invoke-Step "python -m src.reporting.compare_models --dataset $Dataset"
}

if (-not $SkipSurrogateTraining) {
    Invoke-Step "python -m src.data.build_seed_set --dataset $Dataset --seed_size $SeedSize"
    Invoke-Step "python -m src.data.query_seed_labels --dataset $Dataset --target_model $TargetModel --seed_size $SeedSize"
    Invoke-Step "python -m src.augment.run_mixup --dataset $Dataset --target_model $TargetModel --seed_size $SeedSize --alpha $Alpha"
    Invoke-Step "python -m src.data.build_surrogate_trainset --dataset $Dataset --target_model $TargetModel --seed_size $SeedSize --alpha $Alpha"
    Invoke-Step "python -m src.models.train_surrogate_mlp --dataset $Dataset --target_model $TargetModel --seed_size $SeedSize --alpha $Alpha --depth $Depth"
    Invoke-Step "python -m src.evaluation.evaluate_surrogate --dataset $Dataset --target_model $TargetModel --seed_size $SeedSize --alpha $Alpha --depth $Depth"
}

foreach ($attack in $Attacks) {
    Invoke-Step "python -m src.transfer.generate_from_surrogate --dataset $Dataset --target_model $TargetModel --seed_size $SeedSize --alpha $Alpha --depth $Depth --attack $attack"
    Invoke-Step "python -m src.transfer.attack_target --dataset $Dataset --target_model $TargetModel --seed_size $SeedSize --alpha $Alpha --depth $Depth --attack $attack"
}

Invoke-Step "python scripts/summarize_transfer_matrix.py --dataset $Dataset --target-model $TargetModel"

Write-Host ""
Write-Host "Minimal transfer pipeline finished." -ForegroundColor Green
