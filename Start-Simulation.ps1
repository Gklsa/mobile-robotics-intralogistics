param([switch]$SmokeTest, [switch]$Analysis)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$candidates = @(
    (Join-Path $projectPath '.venv\Scripts\python.exe')
)
$pythonPath = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $pythonPath = $candidate
        break
    }
}
if (-not $pythonPath) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $pythonPath = $pythonCommand.Source }
}
if (-not $pythonPath) {
    throw 'Install Python 3.10 or later, then run: python -m pip install -r requirements.txt'
}
Push-Location -LiteralPath $projectPath
try {
    & $pythonPath -c 'import tkinter; import PIL'
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependencies missing. Run: python -m pip install -r requirements.txt'
    }
    if ($SmokeTest) {
        & $pythonPath 'main.py' '--smoke-test'
        if ($LASTEXITCODE -ne 0) { throw 'Simulation smoke test failed.' }
    } else {
        $simArgs = @('main.py')
        if (-not $Analysis) { $simArgs += '--studio' }
        Start-Process -FilePath $pythonPath -ArgumentList $simArgs -WorkingDirectory $projectPath -WindowStyle Hidden
    }
} finally {
    Pop-Location
}
