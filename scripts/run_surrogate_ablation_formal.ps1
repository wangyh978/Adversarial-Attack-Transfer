param(
    [string]$Dataset = "nsl_kdd",
    [string[]]$TargetModels = @("tabnet", "xgb", "gbdt"),
    [int[]]$SeedSizes = @(500, 1000, 2000),
    [double[]]$Alphas = @(0.1, 0.2, 0.5)
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param([string]$Command)
    Write-Host ">> $Command" -ForegroundColor Cyan
    Invoke-Expression $Command
}

foreach ($target in $TargetModels) {
    Write-Host ""
    Write-Host "===== Formal surrogate ablation for target: $target =====" -ForegroundColor Yellow

    foreach ($seed in $SeedSizes) {
        Run-Step "python -m src.data.build_seed_set --dataset $Dataset --seed_size $seed"
        Run-Step "python -m src.data.query_seed_labels --dataset $Dataset --target_model $target --seed_size $seed"

        foreach ($alpha in $Alphas) {
            Run-Step "python -m src.augment.run_mixup --dataset $Dataset --target_model $target --seed_size $seed --alpha $alpha"
            Run-Step "python -m src.data.build_surrogate_trainset --dataset $Dataset --target_model $target --seed_size $seed --alpha $alpha"
        }
    }

    Run-Step "python -m src.models.run_surrogate_ablation --dataset $Dataset --target_model $target --skip_existing"
    Run-Step "python -m src.evaluation.evaluate_surrogate_batch --dataset $Dataset --target_model $target"
    Run-Step "python -m src.reporting.summarize_surrogate_ablation --dataset $Dataset --target_model $target"
    Run-Step "python -m src.models.select_best_surrogate --dataset $Dataset --target_model $target"
}

Write-Host ""
Write-Host "All formal surrogate ablation jobs finished." -ForegroundColor Green
