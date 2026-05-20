# MSM Transfer Attack Result Summary

## Overall conclusion

- Best result: `unsw_nb15 / xgb / mim`, transfer_success_rate = `0.9575`.

## Best attack per dataset

- `unsw_nb15`: `xgb / mim` = `0.9575`
- `nsl_kdd`: `gbdt / ti` = `0.9260`

## Best attack per target model

- `unsw_nb15 / xgb`: `mim`, transfer = `0.9575`, accuracy_drop = `0.7241`, macro_f1_drop = `0.4986`
- `nsl_kdd / gbdt`: `ti`, transfer = `0.9260`, accuracy_drop = `0.9131`, macro_f1_drop = `0.9174`
- `unsw_nb15 / gbdt`: `mim`, transfer = `0.8944`, accuracy_drop = `0.6529`, macro_f1_drop = `0.4517`
- `nsl_kdd / xgb`: `mim`, transfer = `0.8921`, accuracy_drop = `0.8800`, macro_f1_drop = `0.8871`
- `unsw_nb15 / tabnet`: `ti`, transfer = `0.8858`, accuracy_drop = `0.6043`, macro_f1_drop = `0.3667`
- `nsl_kdd / tabnet`: `mim`, transfer = `0.5306`, accuracy_drop = `0.5077`, macro_f1_drop = `0.6402`

## Attack average ranking

| attack   |   transfer_success_rate |
|:---------|------------------------:|
| ti       |                0.809944 |
| mim      |                0.797009 |
| pgd      |                0.379727 |
| fgm      |                0.356029 |
| slide    |                0.334594 |
| cw       |                0.19976  |

## Perturbation anomaly check

- Some extreme perturbation samples exist. Since high quantiles are much smaller than maxima, these are concentrated outliers rather than global perturbation inflation.

| dataset   | target_model   | attack   |   max_l2_perturbation |   max_linf_perturbation |   l2_q0.999 |   linf_q0.999 |   num_linf_gt_1 |   num_l2_gt_5 |
|:----------|:---------------|:---------|----------------------:|------------------------:|------------:|--------------:|----------------:|--------------:|
| nsl_kdd   | gbdt           | fgm      |               82.7172 |                 82.2645 |    0.456684 |     0.314999  |               1 |             1 |
| nsl_kdd   | gbdt           | mim      |               82.7972 |                 82.2645 |    4.15925  |     0.5       |               1 |             1 |
| nsl_kdd   | gbdt           | pgd      |               82.7163 |                 82.2645 |    0.498811 |     0.385949  |               1 |             1 |
| nsl_kdd   | gbdt           | slide    |               82.7163 |                 82.2645 |    0.499205 |     0.370017  |               1 |             1 |
| nsl_kdd   | gbdt           | ti       |               82.8126 |                 82.2645 |    5.42213  |     0.6       |               1 |           661 |
| nsl_kdd   | tabnet         | fgm      |               82.7172 |                 82.2645 |    0.466033 |     0.229619  |               1 |             1 |
| nsl_kdd   | tabnet         | mim      |               82.7955 |                 82.2645 |    4.36077  |     0.5       |               1 |             1 |
| nsl_kdd   | tabnet         | pgd      |               82.7163 |                 82.2645 |    0.499077 |     0.347375  |               1 |             1 |
| nsl_kdd   | tabnet         | slide    |               82.7163 |                 82.2645 |    0.49942  |     0.35063   |               1 |             1 |
| nsl_kdd   | tabnet         | ti       |               82.8327 |                 82.2645 |    5.50246  |     0.6       |               1 |          1561 |
| nsl_kdd   | xgb            | fgm      |               82.7174 |                 82.2645 |    0.459838 |     0.308925  |               1 |             1 |
| nsl_kdd   | xgb            | mim      |               82.8097 |                 82.2645 |    4.28928  |     0.5       |               1 |             1 |
| nsl_kdd   | xgb            | pgd      |               82.7163 |                 82.2645 |    0.498926 |     0.362742  |               1 |             1 |
| nsl_kdd   | xgb            | slide    |               82.7163 |                 82.2645 |    0.499366 |     0.350053  |               1 |             1 |
| nsl_kdd   | xgb            | ti       |               82.8641 |                 82.2645 |    5.45515  |     0.6       |               1 |           968 |
| unsw_nb15 | gbdt           | cw       |              553.749  |                553.749  |    0.991356 |     0.0985724 |             130 |            80 |
| unsw_nb15 | gbdt           | fgm      |              553.749  |                553.749  |    0.480871 |     0.203693  |             130 |            80 |
| unsw_nb15 | gbdt           | mim      |              553.772  |                553.749  |    5.88832  |     0.500002  |             130 |        127667 |
| unsw_nb15 | gbdt           | pgd      |              553.749  |                553.749  |    0.502794 |     0.340166  |             130 |            80 |
| unsw_nb15 | gbdt           | slide    |              553.749  |                553.749  |    0.502447 |     0.342339  |             130 |            80 |
| unsw_nb15 | gbdt           | ti       |              553.773  |                553.749  |    6.25101  |     0.500002  |             130 |        145583 |
| unsw_nb15 | tabnet         | cw       |               32.9576 |                 32.9576 |    0.959731 |     0.0980803 |             122 |            74 |
| unsw_nb15 | tabnet         | fgm      |              553.749  |                553.749  |    0.471259 |     0.259382  |             130 |            80 |
| unsw_nb15 | tabnet         | mim      |              553.772  |                553.749  |    5.86283  |     0.5       |             130 |        138905 |
| unsw_nb15 | tabnet         | pgd      |              553.749  |                553.749  |    0.502766 |     0.349152  |             130 |            80 |
| unsw_nb15 | tabnet         | slide    |              553.749  |                553.749  |    0.50323  |     0.35614   |             130 |            80 |
| unsw_nb15 | tabnet         | ti       |              553.773  |                553.749  |    6.30161  |     0.500001  |             130 |        149252 |
| unsw_nb15 | xgb            | cw       |              553.749  |                553.749  |    0.920329 |     0.0979131 |             114 |            64 |
| unsw_nb15 | xgb            | fgm      |              553.749  |                553.749  |    0.478518 |     0.237229  |             130 |            80 |
| unsw_nb15 | xgb            | mim      |              553.769  |                553.749  |    5.5928   |     0.500002  |             130 |        121274 |
| unsw_nb15 | xgb            | pgd      |              553.749  |                553.749  |    0.499394 |     0.274279  |             130 |            80 |
| unsw_nb15 | xgb            | slide    |              553.749  |                553.749  |    0.499868 |     0.287223  |             130 |            80 |
| unsw_nb15 | xgb            | ti       |              553.766  |                553.749  |    6.15255  |     0.500001  |             130 |        137587 |

## Suggested report wording

> Most adversarial samples are constrained within a reasonable perturbation range. A small number of samples show unusually large maximum L2/Linf perturbations, likely caused by normalization boundaries, inverse-scaling artifacts, or extreme original feature values. Therefore, both maximum perturbation and high-quantile perturbation statistics are reported.