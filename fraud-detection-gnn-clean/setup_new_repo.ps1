param(
    [string]$Name = "fraud-detection-gnn-clean"
)

$root = Join-Path (Get-Location) $Name
Write-Host "creating: $root"

$dirs = @(
    "configs",
    "src\fraud_gnn\data",
    "src\fraud_gnn\models\archived",
    "src\fraud_gnn\evaluation",
    "src\fraud_gnn\utils",
    "experiments\elliptic\archived",
    "experiments\samld\archived",
    "outputs\logs",
    "outputs\checkpoints",
    "outputs\metrics",
    "outputs\figures",
    "dashboard",
    "thesis-exports",
    "docs"
)

foreach ($d in $dirs) {
    $full = Join-Path $root $d
    New-Item -ItemType Directory -Force -Path $full | Out-Null
}

$initDirs = @(
    "src\fraud_gnn",
    "src\fraud_gnn\data",
    "src\fraud_gnn\models",
    "src\fraud_gnn\evaluation",
    "src\fraud_gnn\utils"
)
foreach ($d in $initDirs) {
    $initFile = Join-Path $root "$d\__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Force -Path $initFile | Out-Null
    }
}

Write-Host "done: $root"
Write-Host "cd $root"
Write-Host "git init"
Write-Host "python -m venv .venv"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "pip install -e ."
