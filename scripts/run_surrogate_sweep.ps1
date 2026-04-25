param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("nsl_kdd", "unsw_nb15")]
    [string]$Dataset,

    [Parameter(Mandatory=$false)]
    [string[]]$Targets = @("xgb", "gbdt", "tabnet"),

    [Parameter(Mandatory=$false)]
    [int[]]$SeedSizes = @(500, 1000, 2000),

    [Parameter(Mandatory=$false)]
    [double[]]$Alphas = @(0.05, 0.1, 0.2),

    [Parameter(Mandatory=$false)]
    [int[]]$Depths = @(3, 4, 5),

    [Parameter(Mandatory=$false)]
    [string[]]$Attacks = @("fgm", "pgd", "slide"),

    [Parameter(Mandatory=$false)]
    [ValidateSet("full_attack_matrix", "reuse_artifacts")]
    [string]$Stage = "full_attack_matrix",

    [Parameter(Mandatory=$false)]
    [switch]$CoreOnly,

    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Fix accidental PowerShell variable typo risk for the default seed list.
if ($SeedSizes.Count -eq 0) {
    $SeedSizes = @(500, 1000, 2000)
}

$pyArgs = @(
    "scripts/run_surrogate_sweep.py",
    "--dataset", $Dataset,
    "--targets"
) + $Targets + @(
    "--seed-sizes"
) + ($SeedSizes | ForEach-Object { "$_" }) + @(
    "--alphas"
) + ($Alphas | ForEach-Object { "$_" }) + @(
    "--depths"
) + ($Depths | ForEach-Object { "$_" }) + @(
    "--attacks"
) + $Attacks + @(
    "--stage", $Stage
)

if ($CoreOnly) {
    $pyArgs += "--core-only"
}

if ($DryRun) {
    $pyArgs += "--dry-run"
}

python @pyArgs
