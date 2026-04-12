param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Prepare", "Baseline", "Surrogate", "MinTransfer", "FullAttackMatrix", "FullPipeline")]
    [string]$Stage = "MinTransfer",

    [Parameter(Mandatory=$false)]
    [string[]]$TargetModels = @("tabnet", "xgb", "gbdt"),

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

switch ($Stage) {
    "Prepare" {
        .\scripts\run_min_transfer_matrix.ps1 `
            -Dataset nsl_kdd `
            -TargetModel tabnet `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks @() `
            -SkipTargetTraining `
            -SkipSurrogateTraining:$false
        break
    }
    "Baseline" {
        python -m src.data.load_raw --dataset nsl_kdd
        python -m src.data.clean_labels --dataset nsl_kdd --mode 5class
        python -m src.data.split_data --dataset nsl_kdd
        python -m src.preprocess.run_preprocess_pipeline --dataset nsl_kdd
        python -m src.models.train_sklearn_baseline --dataset nsl_kdd --model random_forest
        python -m src.models.train_xgb --dataset nsl_kdd
        python -m src.models.train_gbdt --dataset nsl_kdd
        python -m src.models.train_tabnet --dataset nsl_kdd
        python -m src.reporting.compare_models --dataset nsl_kdd
        break
    }
    "Surrogate" {
        .\scripts\run_min_transfer_matrix.ps1 `
            -Dataset nsl_kdd `
            -TargetModel tabnet `
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
            -Dataset nsl_kdd `
            -TargetModel tabnet `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks @("fgm", "pgd")
        break
    }
    "FullAttackMatrix" {
        .\scripts\run_full_attack_matrix.ps1 `
            -Dataset nsl_kdd `
            -TargetModels $TargetModels `
            -SeedSize $SeedSize `
            -Alpha $Alpha `
            -Depth $Depth `
            -Attacks $Attacks
        break
    }
    "FullPipeline" {
        .\scripts\run_full_attack_matrix.ps1 `
            -Dataset nsl_kdd `
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
    default {
        throw "Unsupported Stage: $Stage"
    }
}
