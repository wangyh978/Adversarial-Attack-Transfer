$targets = @("tabnet", "xgb", "gbdt")
$attacks = @("fgm", "pgd", "slide")

foreach ($target in $targets) {
    Write-Host ""
    Write-Host "===== UNSW full attack matrix for target: $target ====="

    foreach ($attack in $attacks) {
        python -m src.transfer.generate_from_surrogate --dataset unsw_nb15 --target_model $target --attack $attack
        python -m src.transfer.attack_target --dataset unsw_nb15 --target_model $target --attack $attack
    }
}

Write-Host ""
Write-Host "UNSW-NB15 full attack matrix finished."
