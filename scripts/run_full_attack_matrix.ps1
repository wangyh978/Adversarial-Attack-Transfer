param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("nsl_kdd", "unsw_nb15")]
    [string]$Dataset,

    [Parameter(Mandatory=$false)]
    [string[]]$TargetModels = @(),

    [Parameter(Mandatory=$false)]
    [int]$SeedSize = 1000,

    [Parameter(Mandatory=$false)]
    [double]$Alpha = 0.1,

    [Parameter(Mandatory=$false)]
    [int]$Depth = 3,

    [Parameter(Mandatory=$false)]
    [string[]]$Attacks = @("fgm", "pgd", "slide"),

    [Parameter(Mandatory=$false)]
    [switch]$IncludeTargetTraining,

    [Parameter(Mandatory=$false)]
    [switch]$IncludeSurrogateTraining,

    [Parameter(Mandatory=$false)]
    [switch]$ReuseExistingArtifacts
)

$ErrorActionPreference = "Stop"

function Write-Stage {
    param([string]$Message)

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Resolve-DefaultTargets {
    param([string]$DatasetName)

    switch ($DatasetName) {
        "nsl_kdd"   { return @("tabnet", "xgb", "gbdt") }
        "unsw_nb15" { return @("xgb", "gbdt", "tabnet") }
        default     { throw ("Unsupported dataset: {0}" -f $DatasetName) }
    }
}

function Get-TrainCommand {
    param(
        [string]$DatasetName,
        [string]$TargetModel
    )

    return ("python -m src.training.train_target_model --dataset {0} --model {1}" -f $DatasetName, $TargetModel)
}

function Invoke-TargetTraining {
    param(
        [string]$DatasetName,
        [string]$TargetModel
    )

    Write-Host ("[Target] Training target model: {0}" -f $TargetModel) -ForegroundColor Yellow
    $cmd = Get-TrainCommand -DatasetName $DatasetName -TargetModel $TargetModel
    Invoke-Expression $cmd
}

function Invoke-SeedSetBuild {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [int]$SeedSizeValue
    )

    Write-Host ("[Seed] build_seed_set | dataset={0} target={1} seed_size={2}" -f $DatasetName, $TargetModel, $SeedSizeValue) -ForegroundColor Yellow
    python -m src.data.build_seed_set --dataset $DatasetName --target-model $TargetModel --seed-size $SeedSizeValue
}

function Invoke-QuerySeedLabels {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [int]$SeedSizeValue
    )

    Write-Host ("[Seed] query_seed_labels | dataset={0} target={1} seed_size={2}" -f $DatasetName, $TargetModel, $SeedSizeValue) -ForegroundColor Yellow
    python -m src.data.query_seed_labels --dataset $DatasetName --target-model $TargetModel --seed-size $SeedSizeValue
}

function Invoke-Mixup {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [int]$SeedSizeValue,
        [double]$AlphaValue
    )

    Write-Host ("[Mixup] run_mixup | dataset={0} target={1} seed_size={2} alpha={3}" -f $DatasetName, $TargetModel, $SeedSizeValue, $AlphaValue) -ForegroundColor Yellow
    python -m src.data.run_mixup --dataset $DatasetName --target-model $TargetModel --seed-size $SeedSizeValue --alpha $AlphaValue
}

function Invoke-BuildSurrogateTrainset {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [int]$SeedSizeValue,
        [double]$AlphaValue
    )

    Write-Host ("[Surrogate] build_surrogate_trainset | dataset={0} target={1} seed_size={2} alpha={3}" -f $DatasetName, $TargetModel, $SeedSizeValue, $AlphaValue) -ForegroundColor Yellow
    python -m src.data.build_surrogate_trainset --dataset $DatasetName --target-model $TargetModel --seed-size $SeedSizeValue --alpha $AlphaValue
}

function Invoke-SurrogateTraining {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [int]$SeedSizeValue,
        [double]$AlphaValue,
        [int]$DepthValue
    )

    Write-Host ("[Surrogate] train_surrogate_model | dataset={0} target={1} seed_size={2} alpha={3} depth={4}" -f $DatasetName, $TargetModel, $SeedSizeValue, $AlphaValue, $DepthValue) -ForegroundColor Yellow
    python -m src.training.train_surrogate_model --dataset $DatasetName --target-model $TargetModel --seed-size $SeedSizeValue --alpha $AlphaValue --depth $DepthValue
}

function Invoke-SurrogateEvaluation {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [int]$SeedSizeValue,
        [double]$AlphaValue,
        [int]$DepthValue
    )

    Write-Host ("[Surrogate] evaluate_surrogate_model | dataset={0} target={1} seed_size={2} alpha={3} depth={4}" -f $DatasetName, $TargetModel, $SeedSizeValue, $AlphaValue, $DepthValue) -ForegroundColor Yellow
    python -m src.evaluation.evaluate_surrogate_model --dataset $DatasetName --target-model $TargetModel --seed-size $SeedSizeValue --alpha $AlphaValue --depth $DepthValue
}

function Invoke-GenerateFromSurrogate {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [string]$AttackName,
        [int]$SeedSizeValue,
        [double]$AlphaValue,
        [int]$DepthValue
    )

    Write-Host ("[Attack] generate_from_surrogate | dataset={0} target={1} attack={2}" -f $DatasetName, $TargetModel, $AttackName) -ForegroundColor Yellow
    python -m src.attacks.generate_from_surrogate --dataset $DatasetName --target-model $TargetModel --attack $AttackName --seed-size $SeedSizeValue --alpha $AlphaValue --depth $DepthValue
}

function Invoke-AttackTarget {
    param(
        [string]$DatasetName,
        [string]$TargetModel,
        [string]$AttackName,
        [int]$SeedSizeValue,
        [double]$AlphaValue,
        [int]$DepthValue
    )

    Write-Host ("[Transfer] attack_target | dataset={0} target={1} attack={2}" -f $DatasetName, $TargetModel, $AttackName) -ForegroundColor Yellow
    python -m src.transfer.attack_target --dataset $DatasetName --target-model $TargetModel --attack $AttackName --seed-size $SeedSizeValue --alpha $AlphaValue --depth $DepthValue
}

function Invoke-SummarizeTransferMatrix {
    param(
        [string]$DatasetName,
        [string]$TargetModel
    )

    Write-Host ("[Summary] summarize_transfer_matrix.py | dataset={0} target={1}" -f $DatasetName, $TargetModel) -ForegroundColor Yellow
    python .\scripts\summarize_transfer_matrix.py --dataset $DatasetName --target-model $TargetModel
}

if ($null -eq $TargetModels -or $TargetModels.Count -eq 0) {
    $TargetModels = Resolve-DefaultTargets -DatasetName $Dataset
}

if (-not $ReuseExistingArtifacts) {
    if (-not $PSBoundParameters.ContainsKey("IncludeTargetTraining")) {
        $IncludeTargetTraining = $true
    }

    if (-not $PSBoundParameters.ContainsKey("IncludeSurrogateTraining")) {
        $IncludeSurrogateTraining = $true
    }
}

Write-Stage ("Full Attack Matrix | dataset = {0}" -f $Dataset)
Write-Host ("Target models : {0}" -f ($TargetModels -join ", ")) -ForegroundColor Magenta
Write-Host ("Attacks       : {0}" -f ($Attacks -join ", ")) -ForegroundColor Magenta
Write-Host ("Seed size     : {0}" -f $SeedSize) -ForegroundColor Magenta
Write-Host ("Alpha         : {0}" -f $Alpha) -ForegroundColor Magenta
Write-Host ("Depth         : {0}" -f $Depth) -ForegroundColor Magenta
Write-Host ("Reuse existing: {0}" -f $ReuseExistingArtifacts) -ForegroundColor Magenta
Write-Host ("Train target  : {0}" -f $IncludeTargetTraining) -ForegroundColor Magenta
Write-Host ("Train surro   : {0}" -f $IncludeSurrogateTraining) -ForegroundColor Magenta

foreach ($target in $TargetModels) {
    Write-Stage ("Processing target model: {0}" -f $target)

    if ($IncludeTargetTraining) {
        Invoke-TargetTraining -DatasetName $Dataset -TargetModel $target
    } else {
        Write-Host ("[Skip] target training for {0}" -f $target) -ForegroundColor DarkYellow
    }

    Invoke-SeedSetBuild -DatasetName $Dataset -TargetModel $target -SeedSizeValue $SeedSize
    Invoke-QuerySeedLabels -DatasetName $Dataset -TargetModel $target -SeedSizeValue $SeedSize
    Invoke-Mixup -DatasetName $Dataset -TargetModel $target -SeedSizeValue $SeedSize -AlphaValue $Alpha
    Invoke-BuildSurrogateTrainset -DatasetName $Dataset -TargetModel $target -SeedSizeValue $SeedSize -AlphaValue $Alpha

    if ($IncludeSurrogateTraining) {
        Invoke-SurrogateTraining -DatasetName $Dataset -TargetModel $target -SeedSizeValue $SeedSize -AlphaValue $Alpha -DepthValue $Depth
    } else {
        Write-Host ("[Skip] surrogate training for {0}" -f $target) -ForegroundColor DarkYellow
    }

    Invoke-SurrogateEvaluation -DatasetName $Dataset -TargetModel $target -SeedSizeValue $SeedSize -AlphaValue $Alpha -DepthValue $Depth

    foreach ($attack in $Attacks) {
        Write-Stage ("Target = {0} | Attack = {1}" -f $target, $attack)

        Invoke-GenerateFromSurrogate `
            -DatasetName $Dataset `
            -TargetModel $target `
            -AttackName $attack `
            -SeedSizeValue $SeedSize `
            -AlphaValue $Alpha `
            -DepthValue $Depth

        Invoke-AttackTarget `
            -DatasetName $Dataset `
            -TargetModel $target `
            -AttackName $attack `
            -SeedSizeValue $SeedSize `
            -AlphaValue $Alpha `
            -DepthValue $Depth
    }

    Invoke-SummarizeTransferMatrix -DatasetName $Dataset -TargetModel $target
}

Write-Host ""
Write-Host "[DONE] Full attack matrix completed." -ForegroundColor Green
Write-Host ""
