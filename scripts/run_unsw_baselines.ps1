Write-Host "===== UNSW-NB15 baseline models ====="

Write-Host "[1/5] train_xgb"
python -m src.models.train_xgb --dataset unsw_nb15

Write-Host "[2/5] train_gbdt"
python -m src.models.train_gbdt --dataset unsw_nb15

Write-Host "[3/5] train_tabnet"
python -m src.models.train_tabnet --dataset unsw_nb15

Write-Host "[4/5] train_sklearn_baseline"
python -m src.models.train_sklearn_baseline --dataset unsw_nb15 --model random_forest

Write-Host "[5/5] compare_models"
python -m src.reporting.compare_models --dataset unsw_nb15

Write-Host "UNSW-NB15 baselines finished."
