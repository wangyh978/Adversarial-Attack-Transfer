# Transfer Attack Result Summary

## Overall conclusion

- Strongest setting: `unsw_nb15 / xgb / pgd`, transfer success rate = `0.5614`.
- Average strongest attack type: `pgd`, mean transfer success rate = `0.4521`.

## Best attack per dataset

- `unsw_nb15`: `pgd` on `xgb`, transfer success rate = `0.5614`.
- `nsl_kdd`: `pgd` on `xgb`, transfer success rate = `0.5339`.

## Best attack per target model

- `unsw_nb15 / xgb`: best attack = `pgd`, transfer success rate = `0.5614`, accuracy drop = `0.4126`, macro-F1 drop = `0.3871`.
- `unsw_nb15 / tabnet`: best attack = `pgd`, transfer success rate = `0.5424`, accuracy drop = `0.4011`, macro-F1 drop = `0.3045`.
- `nsl_kdd / xgb`: best attack = `pgd`, transfer success rate = `0.5339`, accuracy drop = `0.5257`, macro-F1 drop = `0.6561`.
- `unsw_nb15 / gbdt`: best attack = `pgd`, transfer success rate = `0.4793`, accuracy drop = `0.3127`, macro-F1 drop = `0.2870`.
- `nsl_kdd / gbdt`: best attack = `pgd`, transfer success rate = `0.3444`, accuracy drop = `0.3371`, macro-F1 drop = `0.3885`.
- `nsl_kdd / tabnet`: best attack = `pgd`, transfer success rate = `0.2511`, accuracy drop = `0.2445`, macro-F1 drop = `0.3683`.

## Attack average ranking

| attack   |   transfer_success_rate |
|:---------|------------------------:|
| pgd      |                0.452085 |
| fgm      |                0.321436 |
| slide    |                0.297141 |

## Perturbation anomaly check

- Some adversarial files contain extreme perturbation samples. Because the 99.9% quantiles are much smaller than the maximum values, the anomaly is concentrated in a small number of samples rather than the whole attack set.

| dataset   | target_model   | attack   |   max_l2_perturbation |   max_linf_perturbation |   l2_q0.999 |   linf_q0.999 |   num_linf_gt_1 |   num_l2_gt_5 |
|:----------|:---------------|:---------|----------------------:|------------------------:|------------:|--------------:|----------------:|--------------:|
| nsl_kdd   | gbdt           | fgm      |               82.7173 |                 82.2645 |    0.450933 |      0.356605 |               1 |             1 |
| nsl_kdd   | gbdt           | pgd      |               82.7164 |                 82.2645 |    0.884407 |      0.5      |               1 |             1 |
| nsl_kdd   | gbdt           | slide    |               82.7163 |                 82.2645 |    0.498881 |      0.412686 |               1 |             1 |
| nsl_kdd   | tabnet         | fgm      |               82.7174 |                 82.2645 |    0.467145 |      0.23837  |               1 |             1 |
| nsl_kdd   | tabnet         | pgd      |               82.7204 |                 82.2645 |    0.90073  |      0.447271 |               1 |             1 |
| nsl_kdd   | tabnet         | slide    |               82.7163 |                 82.2645 |    0.499245 |      0.337985 |               1 |             1 |
| nsl_kdd   | xgb            | fgm      |               82.7175 |                 82.2645 |    0.46165  |      0.351052 |               1 |             1 |
| nsl_kdd   | xgb            | pgd      |               82.7213 |                 82.2645 |    0.922729 |      0.5      |               1 |             1 |
| nsl_kdd   | xgb            | slide    |               82.7163 |                 82.2645 |    0.49961  |      0.415117 |               1 |             1 |
| unsw_nb15 | gbdt           | fgm      |              553.749  |                553.749  |    0.476373 |      0.222614 |             130 |            80 |
| unsw_nb15 | gbdt           | pgd      |              553.749  |                553.749  |    0.936157 |      0.426026 |             130 |            80 |
| unsw_nb15 | gbdt           | slide    |              553.749  |                553.749  |    0.503329 |      0.334238 |             130 |            80 |
| unsw_nb15 | tabnet         | fgm      |              553.749  |                553.749  |    0.486076 |      0.231983 |             130 |            80 |
| unsw_nb15 | tabnet         | pgd      |              553.749  |                553.749  |    0.940544 |      0.383813 |             130 |            80 |
| unsw_nb15 | tabnet         | slide    |              553.749  |                553.749  |    0.504125 |      0.349242 |             130 |            80 |
| unsw_nb15 | xgb            | fgm      |              553.749  |                553.749  |    0.478798 |      0.257343 |             130 |            80 |
| unsw_nb15 | xgb            | pgd      |              553.749  |                553.749  |    0.951643 |      0.452724 |             130 |            80 |
| unsw_nb15 | xgb            | slide    |              553.749  |                553.749  |    0.505077 |      0.330861 |             130 |            80 |

## Suggested wording for paper/report

> Most adversarial samples are constrained within a reasonable perturbation range. However, a small number of samples show unusually large maximum L2/Linf perturbations, which may be caused by feature normalization boundaries, inverse-scaling artifacts, or extreme original feature values. Therefore, this study reports both maximum perturbation and high-quantile perturbation statistics to avoid overestimating the global perturbation magnitude.

## Generated files

- `results/summary/all_transfer_matrix.csv`
- `results/summary/all_transfer_matrix.md`
- `results/summary/all_metrics_detail.csv`
- `results/summary/result_summary.md`
- `results/summary/plots/transfer_success_rate_bar.png`
- `results/summary/plots/accuracy_drop_bar.png`
- `results/summary/plots/macro_f1_drop_bar.png`
- `results/summary/plots/transfer_success_rate_grouped.png`
- `results/summary/plots/transfer_success_rate_heatmap.png`
- `results/summary/plots/perturbation_linf_999.png`