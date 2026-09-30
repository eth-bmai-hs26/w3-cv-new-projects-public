# Start the tile inspection platform (Windows PowerShell):
#   .\run.ps1                                              opens http://127.0.0.1:8000 in your browser
#   .\run.ps1 --model $HOME\Downloads\platform_model.pt    ... with the model you downloaded from Colab
#   .\run.ps1 --host 0.0.0.0                               ... reachable from a phone on the same Wi-Fi
# Or double-click start-windows.bat, which runs this script.
# First run: creates .venv and installs the Python packages (a few minutes, needs internet).
# The web interface is shipped pre-built in frontend\dist, so Node.js is not needed.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Run a native command and stop if it fails ($ErrorActionPreference does not cover exe exit codes).
function Invoke-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "'$exe $($arguments -join ' ')' failed (exit code $LASTEXITCODE)" }
}

# 1. A Python 3.10+ interpreter: $env:PYTHON, else the "py -3" launcher, else python.
#    (The "python" that opens the Microsoft Store is a stub and fails the version check.)
function Find-Python {
    $candidates = @()
    if ($env:PYTHON) { $candidates += , @($env:PYTHON) }
    $candidates += , @("py", "-3")
    $candidates += , @("python")
    foreach ($cand in $candidates) {
        $exe = $cand[0]
        $pre = @($cand | Select-Object -Skip 1)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            & $exe @pre -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return , $cand }
        } catch { }
    }
    return $null
}

$venvPy = ".venv\Scripts\python.exe"
$marker = ".venv\.installed"

# 2. The virtual environment. The marker file records which requirements.txt was
#    installed, so an interrupted or outdated install is repeated automatically.
$upToDate = (Test-Path $venvPy) -and (Test-Path $marker) -and
            ((Get-FileHash requirements.txt).Hash -eq (Get-FileHash $marker).Hash)
if (-not $upToDate) {
    $py = Find-Python
    if (-not $py) {
        Write-Host "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/"
        Write-Host "and tick 'Add python.exe to PATH' in the installer. Then run this again."
        exit 1
    }
    $pyExe = $py[0]
    $pyPre = @($py | Select-Object -Skip 1)
    Write-Host "Setting up .venv with $(& $pyExe @pyPre --version) - one time, a few minutes..."
    if (-not (Test-Path $venvPy)) { Invoke-Checked $pyExe ($pyPre + @("-m", "venv", ".venv")) }
    Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip", "--quiet")
    Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "requirements.txt")
    Copy-Item requirements.txt $marker -Force
}

# 3. The web interface (only rebuilt if frontend\dist was deleted).
if (-not (Test-Path "frontend\dist\index.html")) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "Building the frontend..."
        Push-Location frontend
        try {
            Invoke-Checked "npm" @("install", "--no-audit", "--no-fund")
            Invoke-Checked "npm" @("run", "build")
        } finally { Pop-Location }
    } else {
        Write-Host "frontend\dist is missing and Node.js is not installed: restore it with 'git checkout frontend/dist'."
        Write-Host "The API still starts; open http://127.0.0.1:8000/docs"
    }
}

# 4. Start. Serving (no sub-command) also opens the browser.
$cliArgs = @($args)
if ($cliArgs.Count -eq 0 -or "$($cliArgs[0])".StartsWith("-")) { $cliArgs += "--open" }
& $venvPy -m backend @cliArgs
exit $LASTEXITCODE
