[CmdletBinding()]
param([switch]$SkipData, [switch]$Test)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    $uvPath = $uv.Source
} else {
    $runtime = Join-Path $PWD '.runtime'
    New-Item -ItemType Directory -Force -Path $runtime | Out-Null
    $uvPath = Join-Path $runtime 'uv.exe'
    if (-not (Test-Path -LiteralPath $uvPath)) {
        $zip = Join-Path $runtime 'uv.zip'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
        Expand-Archive -LiteralPath $zip -DestinationPath $runtime -Force
    }
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    $settings = New-Object System.Diagnostics.ProcessStartInfo
    $settings.FileName = $Executable
    $settings.Arguments = ($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $settings.WorkingDirectory = (Get-Location).Path
    $settings.UseShellExecute = $false
    $settings.CreateNoWindow = $true
    $settings.RedirectStandardOutput = $true
    $settings.RedirectStandardError = $true
    $settings.StandardOutputEncoding = [Text.Encoding]::UTF8
    $settings.StandardErrorEncoding = [Text.Encoding]::UTF8
    $process = [Diagnostics.Process]::Start($settings)
    try {
        $errors = $process.StandardError.ReadToEndAsync()
        while (($line = $process.StandardOutput.ReadLine()) -ne $null) { Write-Host $line }
        $process.WaitForExit()
        if ($errors.Result) { Write-Host $errors.Result }
        if ($process.ExitCode -ne 0) { throw "Error de $Executable (codigo $($process.ExitCode))" }
    } finally {
        $process.Dispose()
    }
}
$env:PYTHONUTF8 = '1'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.runtime/python'
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    Invoke-Checked $uvPath @('python', 'install', '3.12.11', '--no-bin')
    Invoke-Checked $uvPath @('venv', '--python', '3.12.11', '--managed-python')
}
$py = Join-Path $PWD '.venv/Scripts/python.exe'
Invoke-Checked $uvPath @('pip', 'install', '--python', $py, '-r', 'scripts/requirements-windows.txt')
Invoke-Checked $uvPath @('pip', 'install', '--python', $py, '--no-deps', '--no-build-isolation', '-e', '.')
Invoke-Checked $py @('-m', 'flybrain_snake', 'doctor')
if (-not $SkipData) { Invoke-Checked $py @('-m', 'flybrain_snake', 'prepare') }
if ($Test) { Invoke-Checked $py @('-m', 'pytest', '-q') }
