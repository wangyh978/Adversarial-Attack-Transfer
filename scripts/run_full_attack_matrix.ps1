$ErrorActionPreference = "Stop"

$dataset = "nsl_kdd"
$targets = @("tabnet", "xgb", "gbdt")
$seedSizes = @(500, 1000, 2000)
$alphas = @(0.1, 0.2, 0.5)
$attacks = @("fgm", "pgd", "slide")   # 如果支持 C&W，改成 @("fgm","pgd","slide","cw")

function Run-Step([string]$cmd) {
    Write-Host ""
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
}

# 1) build seed sets
foreach ($seed in $seedSizes) {
    Run-Step "python -m src.data.build_seed_set --dataset $dataset --seed_size $seed"
}

# 2) prepare queried seeds + mixup + surrogate trainset
foreach ($target in $targets) {
    foreach ($seed in $seedSizes) {
        Run-Step "python -m src.data.query_seed_labels --dataset $dataset --target_model $target --seed_size $seed"

        foreach ($alpha in $alphas) {
            Run-Step "python -m src.augment.run_mixup --dataset $dataset --target_model $target --seed_size $seed --alpha $alpha"
            Run-Step "python -m src.data.build_surrogate_trainset --dataset $dataset --target_model $target --seed_size $seed --alpha $alpha"
        }
    }

    # 3) surrogate ablation / evaluate / summarize / select
    Run-Step "python -m src.models.run_surrogate_ablation --dataset $dataset --target_model $target"
    Run-Step "python -m src.evaluation.evaluate_surrogate_batch --dataset $dataset --target_model $target"
    Run-Step "python -m src.reporting.summarize_surrogate_ablation --dataset $dataset --target_model $target"
    Run-Step "python -m src.models.select_best_surrogate --dataset $dataset --target_model $target"

    # 4) attacks
    foreach ($attack in $attacks) {
        Run-Step "python -m src.transfer.generate_from_surrogate --dataset $dataset --target_model $target --attack $attack"
        Run-Step "python -m src.transfer.attack_target --dataset $dataset --target_model $target --attack $attack"
    }
}

# 5) summarize final transfer matrix
Run-Step "python scripts/summarize_transfer_matrix.py --dataset $dataset"
