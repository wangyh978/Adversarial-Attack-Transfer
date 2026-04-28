# MSM Transfer Attack Result Summary

## Overall conclusion

- Best result: `nsl_kdd / xgb / mim`, transfer_success_rate = `0.8918`.

## Best attack per dataset

- `nsl_kdd`: `xgb / mim` = `0.8918`
- `unsw_nb15`: `xgb / pgd` = `0.5803`

## Best attack per target model

- `nsl_kdd / xgb`: `mim`, transfer = `0.8918`, accuracy_drop = `0.8797`, macro_f1_drop = `0.8870`
- `unsw_nb15 / xgb`: `pgd`, transfer = `0.5803`, accuracy_drop = `0.4367`, macro_f1_drop = `0.4015`
- `unsw_nb15 / tabnet`: `pgd`, transfer = `0.5764`, accuracy_drop = `0.4233`, macro_f1_drop = `0.3060`
- `unsw_nb15 / gbdt`: `pgd`, transfer = `0.5421`, accuracy_drop = `0.3625`, macro_f1_drop = `0.3103`
- `nsl_kdd / gbdt`: `pgd`, transfer = `0.3444`, accuracy_drop = `0.3371`, macro_f1_drop = `0.3885`
- `nsl_kdd / tabnet`: `pgd`, transfer = `0.2511`, accuracy_drop = `0.2445`, macro_f1_drop = `0.3683`

## Attack average ranking

| attack   |   transfer_success_rate |
|:---------|------------------------:|
| mim      |                0.891787 |
| ti       |                0.760192 |
| pgd      |                0.471374 |
| cw       |                0.470923 |
| fgm      |                0.318697 |
| slide    |                0.295325 |

## Perturbation anomaly check

- Some extreme perturbation samples exist. Since high quantiles are much smaller than maxima, these are concentrated outliers rather than global perturbation inflation.

| dataset   | target_model   | attack   |   max_l2_perturbation |   max_linf_perturbation |   l2_q0.999 |   linf_q0.999 |   num_linf_gt_1 |   num_l2_gt_5 |
|:----------|:---------------|:---------|----------------------:|------------------------:|------------:|--------------:|----------------:|--------------:|
| nsl_kdd   | gbdt           | fgm      |               82.7173 |                 82.2645 |    0.450933 |      0.356605 |               1 |             1 |
| nsl_kdd   | gbdt           | pgd      |               82.7164 |                 82.2645 |    0.884407 |      0.5      |               1 |             1 |
| nsl_kdd   | gbdt           | slide    |               82.7163 |                 82.2645 |    0.498881 |      0.412686 |               1 |             1 |
| nsl_kdd   | tabnet         | fgm      |               82.7174 |                 82.2645 |    0.467145 |      0.23837  |               1 |             1 |
| nsl_kdd   | tabnet         | pgd      |               82.7204 |                 82.2645 |    0.90073  |      0.447271 |               1 |             1 |
| nsl_kdd   | tabnet         | slide    |               82.7163 |                 82.2645 |    0.499245 |      0.337985 |               1 |             1 |
| nsl_kdd   | xgb            | cw       |               82.7164 |                 82.2645 |    0.807363 |      0.297868 |               1 |             1 |
| nsl_kdd   | xgb            | fgm      |               82.7175 |                 82.2645 |    0.46165  |      0.351052 |               1 |             1 |
| nsl_kdd   | xgb            | mim      |               82.8093 |                 82.2645 |    4.28928  |      0.5      |               1 |             1 |
| nsl_kdd   | xgb            | pgd      |               82.7213 |                 82.2645 |    0.922729 |      0.5      |               1 |             1 |
| nsl_kdd   | xgb            | slide    |               82.7163 |                 82.2645 |    0.49961  |      0.415117 |               1 |             1 |
| nsl_kdd   | xgb            | ti       |               82.8038 |                 82.2645 |    4.45242  |      0.5      |               1 |             1 |
| unsw_nb15 | gbdt           | fgm      |              553.749  |                553.749  |    0.468996 |      0.213279 |             130 |            80 |
| unsw_nb15 | gbdt           | pgd      |              553.749  |                553.749  |    0.927077 |      0.416118 |             130 |            80 |
| unsw_nb15 | gbdt           | slide    |              553.749  |                553.749  |    0.501396 |      0.309134 |             130 |            80 |
| unsw_nb15 | tabnet         | fgm      |              553.749  |                553.749  |    0.462153 |      0.206501 |             130 |            80 |
| unsw_nb15 | tabnet         | pgd      |              553.749  |                553.749  |    0.919938 |      0.41838  |             130 |            80 |
| unsw_nb15 | tabnet         | slide    |              553.749  |                553.749  |    0.503742 |      0.326826 |             130 |            80 |
| unsw_nb15 | xgb            | fgm      |              553.749  |                553.749  |    0.481789 |      0.228005 |             130 |            80 |
| unsw_nb15 | xgb            | pgd      |              553.749  |                553.749  |    0.95034  |      0.4145   |             130 |            80 |
| unsw_nb15 | xgb            | slide    |              553.749  |                553.749  |    0.501735 |      0.325692 |             130 |            80 |

## Suggested report wording

> Most adversarial samples are constrained within a reasonable perturbation range. A small number of samples show unusually large maximum L2/Linf perturbations, likely caused by normalization boundaries, inverse-scaling artifacts, or extreme original feature values. Therefore, both maximum perturbation and high-quantile perturbation statistics are reported.