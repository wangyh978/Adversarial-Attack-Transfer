$targets = @("tabnet", "xgb", "gbdt")
$seed_sizes = @(500, 1000, 2000)
$alphas = @(0.1, 0.2, 0.5)

foreach ($target in $targets) {
    Write-Host ""
    Write-Host "===== Formal surrogate ablation for UNSW target: $target ====="

    foreach ($seed_size in $seed_sizes) {
        python -m src.data.build_seed_set --dataset unsw_nb15 --seed_size $seed_size
        python -m src.data.query_seed_labels --dataset unsw_nb15 --target_model $target --seed_size $seed_size

        foreach ($alpha in $alphas) {
            python -m src.augment.run_mixup --dataset unsw_nb15 --target_model $target --seed_size $seed_size --alpha $alpha
            python -m src.data.build_surrogate_trainset --dataset unsw_nb15 --target_model $target --seed_size $seed_size --alpha $alpha
        }
    }

    python -m src.models.run_surrogate_ablation --dataset unsw_nb15 --target_model $target --skip_existing
    python -m src.evaluation.evaluate_surrogate_batch --dataset unsw_nb15 --target_model $target
    python -m src.reporting.summarize_surrogate_ablation --dataset unsw_nb15 --target_model $target
    python -m src.models.select_best_surrogate --dataset unsw_nb15 --target_model $target
}

Write-Host ""
Write-Host "All UNSW-NB15 formal surrogate ablation jobs finished."
