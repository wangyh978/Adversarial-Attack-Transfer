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
        python -m src.data.load_raw --dataset unsw_nb15
        python -m src.data.clean_labels --dataset unsw_nb15 --mode multiclass
        python -m src.data.split_data --dataset unsw_nb15
        python -m src.preprocess.run_preprocess_pipeline --dataset unsw_nb15
        break
    }

    "Baseline" {
        Write-Stage "Step 1/2: Prepare dataset"
        python -m src.data.load_raw --dataset unsw_nb15
        python -m src.data.clean_labels --dataset unsw_nb15 --mode multiclass
        python -m src.data.split_data --dataset unsw_nb15
        python -m src.preprocess.run_preprocess_pipeline --dataset unsw_nb15

        Write-Stage "Step 2/2: Train baseline target models"
        foreach ($target in $TargetModels) {
            switch ($target) {
                "xgb" {
                    python -m src.models.train_xgb --dataset unsw_nb15
                }
                "gbdt" {
                    python -m src.models.train_gbdt --dataset unsw_nb15
                }
                "tabnet" {
                    python -m src.models.train_tabnet --dataset unsw_nb15
                }
                default {
                    throw "Unsupported target model: $target"
                }
            }
        }
        python -m src.reporting.compare_models --dataset unsw_nb15
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
            -Attacks $Attacks `
            -IncludeTargetTraining `
            -IncludeSurrogateTraining
        break
    }

    "FullPipeline" {
        Write-Stage "Step 1/1: Run full pipeline for UNSW-NB15"

        .\scripts\run_full_attack_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModels $TargetModels `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks $Attacks `
            -IncludePreparation `
            -IncludeTargetTraining `
            -IncludeSurrogateTraining
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
