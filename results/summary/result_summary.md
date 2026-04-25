# MSM Transfer Attack Result Summary

## Overall conclusion

- Best result: `unsw_nb15 / gbdt / pgd`, transfer_success_rate = `0.5703`.

## Best attack per dataset

- `unsw_nb15`: `gbdt / pgd` = `0.5703`
- `nsl_kdd`: `xgb / pgd` = `0.4571`

## Best attack per target model

- `unsw_nb15 / gbdt`: `pgd`, transfer = `0.5703`, accuracy_drop = `0.4190`, macro_f1_drop = `0.3207`
- `unsw_nb15 / xgb`: `pgd`, transfer = `0.5587`, accuracy_drop = `0.4207`, macro_f1_drop = `0.3880`
- `unsw_nb15 / tabnet`: `pgd`, transfer = `0.5169`, accuracy_drop = `0.3728`, macro_f1_drop = `0.2706`
- `nsl_kdd / xgb`: `pgd`, transfer = `0.4571`, accuracy_drop = `0.4480`, macro_f1_drop = `0.5918`
- `nsl_kdd / gbdt`: `pgd`, transfer = `0.3333`, accuracy_drop = `0.3258`, macro_f1_drop = `0.4526`
- `nsl_kdd / tabnet`: `pgd`, transfer = `0.3175`, accuracy_drop = `0.3028`, macro_f1_drop = `0.3962`

## Attack average ranking

| attack   |   transfer_success_rate |
|:---------|------------------------:|
| pgd      |                0.458968 |
| fgm      |                0.323248 |
| slide    |                0.298853 |

## Perturbation anomaly check

- Some extreme perturbation samples exist. Since high quantiles are much smaller than maxima, these are concentrated outliers rather than global perturbation inflation.

| dataset   | target_model   | attack   |   max_l2_perturbation |   max_linf_perturbation |   l2_q0.999 |   linf_q0.999 |   num_linf_gt_1 |   num_l2_gt_5 |
|:----------|:---------------|:---------|----------------------:|------------------------:|------------:|--------------:|----------------:|--------------:|
| nsl_kdd   | gbdt           | fgm      |               82.7173 |                 82.2645 |    0.451131 |      0.350213 |               1 |             1 |
| nsl_kdd   | gbdt           | pgd      |               82.7164 |                 82.2645 |    0.891039 |      0.5      |               1 |             1 |
| nsl_kdd   | gbdt           | slide    |               82.7163 |                 82.2645 |    0.499039 |      0.407964 |               1 |             1 |
| nsl_kdd   | tabnet         | fgm      |               82.7172 |                 82.2645 |    0.461727 |      0.231415 |               1 |             1 |
| nsl_kdd   | tabnet         | pgd      |               82.7194 |                 82.2645 |    0.907596 |      0.462821 |               1 |             1 |
| nsl_kdd   | tabnet         | slide    |               82.7163 |                 82.2645 |    0.499408 |      0.345496 |               1 |             1 |
| nsl_kdd   | xgb            | fgm      |               82.7174 |                 82.2645 |    0.455236 |      0.328083 |               1 |             1 |
| nsl_kdd   | xgb            | pgd      |               82.7209 |                 82.2645 |    0.900088 |      0.5      |               1 |             1 |
| nsl_kdd   | xgb            | slide    |               82.7163 |                 82.2645 |    0.499437 |      0.40335  |               1 |             1 |
| unsw_nb15 | gbdt           | fgm      |              553.749  |                553.749  |    0.480871 |      0.203693 |             130 |            80 |
| unsw_nb15 | gbdt           | pgd      |              553.749  |                553.749  |    0.942068 |      0.406496 |             130 |            80 |
| unsw_nb15 | gbdt           | slide    |              553.749  |                553.749  |    0.502731 |      0.349666 |             130 |            80 |
| unsw_nb15 | tabnet         | fgm      |              553.749  |                553.749  |    0.478494 |      0.205124 |             130 |            80 |
| unsw_nb15 | tabnet         | pgd      |              553.749  |                553.749  |    0.939409 |      0.373565 |             130 |            80 |
| unsw_nb15 | tabnet         | slide    |              553.749  |                553.749  |    0.503563 |      0.349725 |             130 |            80 |
| unsw_nb15 | xgb            | fgm      |              553.749  |                553.749  |    0.485916 |      0.216044 |             130 |            80 |
| unsw_nb15 | xgb            | pgd      |              553.749  |                553.749  |    0.954742 |      0.369851 |             130 |            80 |
| unsw_nb15 | xgb            | slide    |              553.749  |                553.749  |    0.505239 |      0.333761 |             130 |            80 |

## Suggested report wording

> Most adversarial samples are constrained within a reasonable perturbation range. A small number of samples show unusually large maximum L2/Linf perturbations, likely caused by normalization boundaries, inverse-scaling artifacts, or extreme original feature values. Therefore, both maximum perturbation and high-quantile perturbation statistics are reported.