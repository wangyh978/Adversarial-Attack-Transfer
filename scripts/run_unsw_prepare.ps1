Write-Host "===== UNSW-NB15 data preparation ====="

python -m src.data.load_raw --dataset unsw_nb15 --save
python -m src.data.clean_labels --dataset unsw_nb15 --mode multiclass
python -m src.data.split_data --dataset unsw_nb15 --val_size 0.2
python -m src.preprocess.run_preprocess_pipeline --dataset unsw_nb15

Write-Host "UNSW-NB15 prepare finished."
