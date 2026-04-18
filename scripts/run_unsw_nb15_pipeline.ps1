param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Prepare", "Baseline", "Surrogate", "MinTransfer", "FullAttackMatrix", "FullPipeline", "ReuseArtifacts")]
    [string]$Stage = "MinTransfer",

    [Parameter(Mandatory=$false)]
    [string[]]$TargetModels = @("xgb", "gbdt", "tabnet"),

    [Parameter(Mandatory=$false)]
    [int]$SeedSize = 1000,

    [Parameter(Mandatory=$false)]
    [double]$Alpha = 0.1,

    [Parameter(Mandatory=$false)]
    [int]$Depth = 3,

    [Parameter(Mandatory=$false)]
    [string[]]$Attacks = @("fgm", "pgd", "slide")
)

$ErrorActionPreference = "Stop"

function Write-Stage {
    param([string]$Message)

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Get-PrimaryTarget {
    if ($null -eq $TargetModels -or $TargetModels.Count -eq 0) {
        throw "TargetModels is empty."
    }

    return $TargetModels[0]
}

Write-Stage ("UNSW-NB15 Pipeline | Stage = {0}" -f $Stage)

switch ($Stage) {
    "Prepare" {
        Write-Stage "Step 1/1: Prepare UNSW-NB15 dataset"
        python -m src.data.clean_unsw_nb15
        python -m src.features.preprocess --dataset unsw_nb15
        break
    }

    "Baseline" {
        Write-Stage "Step 1/1: Train baseline target models for UNSW-NB15"
        foreach ($target in $TargetModels) {
            Write-Host ("[Baseline] Training target model: {0}" -f $target) -ForegroundColor Yellow
            python -m src.training.train_target_model --dataset unsw_nb15 --model $target
        }
        break
    }

    "Surrogate" {
        $primaryTarget = Get-PrimaryTarget
        Write-Stage ("Step 1/1: Train surrogate for UNSW-NB15 | target = {0}" -f $primaryTarget)

        .\scripts\run_min_transfer_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModel $primaryTarget `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks @() `
            -SkipDataPreparation `
            -SkipTargetTraining

        break
    }

    "MinTransfer" {
        $primaryTarget = Get-PrimaryTarget
        Write-Stage ("Step 1/1: Run minimal transfer pipeline for UNSW-NB15 | target = {0}" -f $primaryTarget)

        .\scripts\run_min_transfer_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModel $primaryTarget `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks $Attacks

        break
    }

    "FullAttackMatrix" {
        Write-Stage "Step 1/1: Run full attack matrix for UNSW-NB15"

        .\scripts\run_full_attack_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModels $TargetModels `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks $Attacks

        break
    }

    "FullPipeline" {
        Write-Stage "Step 1/3: Prepare dataset"
        python -m src.data.clean_unsw_nb15
        python -m src.features.preprocess --dataset unsw_nb15

        Write-Stage "Step 2/3: Train target models"
        foreach ($target in $TargetModels) {
            Write-Host ("[FullPipeline] Training target model: {0}" -f $target) -ForegroundColor Yellow
            python -m src.training.train_target_model --dataset unsw_nb15 --model $target
        }

        Write-Stage "Step 3/3: Run full attack matrix"
        .\scripts\run_full_attack_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModels $TargetModels `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks $Attacks `
            -ReuseExistingArtifacts

        break
    }

    "ReuseArtifacts" {
        Write-Stage "Step 1/1: Reuse existing artifacts and run full attack matrix"

        .\scripts\run_full_attack_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModels $TargetModels `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks $Attacks `
            -ReuseExistingArtifacts

        break
    }

    default {
        throw ("Unsupported Stage: {0}" -f $Stage)
    }
}

Write-Host ""
Write-Host "[DONE] UNSW-NB15 pipeline finished." -ForegroundColor Green
Write-Host ""
