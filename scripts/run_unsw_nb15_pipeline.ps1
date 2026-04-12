param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Prepare", "Baseline", "Surrogate", "MinTransfer", "FullAttackMatrix", "FullPipeline", "ReuseArtifacts")]
    [string]$Stage = "MinTransfer",

    [Parameter(Mandatory=$false)]
    [string[]]$TargetModels = @("xgb"),

    [Parameter(Mandatory=$false)]
    [int]$SeedSize = 1000,

    [Parameter(Mandatory=$false)]
    [double]$Alpha = 0.1,

    [Parameter(Mandatory=$false)]
    [int]$Depth = 3,

    [Parameter(Mandatory=$false)]
    [string[]]$Attacks = @("fgm", "pgd")
)

$ErrorActionPreference = "Stop"

switch ($Stage) {
    "Prepare" {
        python -m src.data.load_raw --dataset unsw_nb15
        python -m src.data.clean_labels --dataset unsw_nb15 --mode multiclass
        python -m src.data.split_data --dataset unsw_nb15
        python -m src.preprocess.run_preprocess_pipeline --dataset unsw_nb15
        break
    }
    "Baseline" {
        python -m src.data.load_raw --dataset unsw_nb15
        python -m src.data.clean_labels --dataset unsw_nb15 --mode multiclass
        python -m src.data.split_data --dataset unsw_nb15
        python -m src.preprocess.run_preprocess_pipeline --dataset unsw_nb15
        python -m src.models.train_sklearn_baseline --dataset unsw_nb15 --model random_forest
        python -m src.models.train_xgb --dataset unsw_nb15
        python -m src.models.train_gbdt --dataset unsw_nb15
        python -m src.models.train_tabnet --dataset unsw_nb15
        python -m src.reporting.compare_models --dataset unsw_nb15
        break
    }
    "Surrogate" {
        .\scripts\run_min_transfer_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModel xgb `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks @() `
            -SkipDataPreparation `
            -SkipTargetTraining
        break
    }
    "MinTransfer" {
        .\scripts\run_min_transfer_matrix.ps1 `
            -Dataset unsw_nb15 `
            -TargetModel xgb `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks @("fgm", "pgd")
        break
    }
    "FullAttackMatrix" {
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
        throw "Unsupported Stage: $Stage"
    }
}
