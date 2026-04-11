$ErrorActionPreference = "Stop"

$dataset = "nsl_kdd"
$seed = 500
$alpha = 0.1
$targets = @("tabnet", "xgb", "gbdt")
$attacks = @("fgm", "pgd", "slide")

function Run-Step([string]$cmd) {
    Write-Host ""
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
}

# 0. 确保基础 seed set 存在
Run-Step "python -m src.data.build_seed_set --dataset $dataset --seed_size $seed"

foreach ($target in $targets) {
    Write-Host ""
    Write-Host "==============================" -ForegroundColor Yellow
    Write-Host "TARGET MODEL: $target" -ForegroundColor Yellow
    Write-Host "==============================" -ForegroundColor Yellow

    # 1. 查询 seed 标签
    Run-Step "python -m src.data.query_seed_labels --dataset $dataset --target_model $target --seed_size $seed"

    # 2. mixup
    Run-Step "python -m src.augment.run_mixup --dataset $dataset --target_model $target --seed_size $seed --alpha $alpha"

    # 3. surrogate 训练集
    Run-Step "python -m src.data.build_surrogate_trainset --dataset $dataset --target_model $target --seed_size $seed --alpha $alpha"

    # 4. 只训练最小一组 surrogate：depth=7
    Run-Step "python -m src.models.train_surrogate_mlp --dataset $dataset --target_model $target --seed_size $seed --alpha $alpha --depth 7"

    # 5. surrogate 评估
    Run-Step "python -m src.evaluation.evaluate_surrogate --dataset $dataset --target_model $target --seed_size $seed --alpha $alpha --depth 7"

    # 6. 依次跑三种攻击
    foreach ($attack in $attacks) {
        Run-Step "python -m src.transfer.generate_from_surrogate --dataset $dataset --target_model $target --attack $attack"
        Run-Step "python -m src.transfer.attack_target --dataset $dataset --target_model $target --attack $attack"
    }
}

# 7. 汇总最终矩阵
Run-Step "python scripts/summarize_transfer_matrix.py --dataset $dataset"

Write-Host ""
Write-Host "All done." -ForegroundColor Green
Write-Host "Check: results/tables/final_transfer_matrix_${dataset}.csv" -ForegroundColor Green
Write-Host "Check: results/tables/final_transfer_matrix_${dataset}.md" -ForegroundColor Green
