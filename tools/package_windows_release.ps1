param(
    [string] $DistDir,
    [string] $Commit
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
if ([string]::IsNullOrWhiteSpace($DistDir)) {
    $DistDir = Join-Path $RepoRoot "dist"
}
if ([string]::IsNullOrWhiteSpace($Commit)) {
    $Commit = (& git -C $RepoRoot rev-parse --short HEAD).Trim()
}

$CudaBuild = Join-Path $RepoRoot "build-win\Release"
$VulkanBuild = Join-Path $RepoRoot "build-win-vulkan\Release"
$CudaPackage = Join-Path $DistDir "trellis2-local-windows-x64-cuda"
$VulkanPackage = Join-Path $DistDir "trellis2-local-windows-x64-vulkan"

function Ensure-Dir($Path) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Reset-Dir($Path) {
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    Ensure-Dir $Path
}

function Copy-Required($Source, $DestinationDir) {
    if (!(Test-Path -LiteralPath $Source)) {
        throw "Missing required file: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $DestinationDir -Force
}

function Find-Required($Name, [string[]] $Dirs) {
    foreach ($Dir in $Dirs) {
        if ([string]::IsNullOrWhiteSpace($Dir)) {
            continue
        }
        $Candidate = Join-Path $Dir $Name
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }
    throw "Could not find required runtime DLL: $Name"
}

function Find-Optional($Name, [string[]] $Dirs) {
    foreach ($Dir in $Dirs) {
        if ([string]::IsNullOrWhiteSpace($Dir)) {
            continue
        }
        $Candidate = Join-Path $Dir $Name
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }
    return $null
}

function Write-Launcher($DestinationDir) {
    $Content = @'
@echo off
set "HERE=%~dp0"
cd /d "%HERE%"
start "" "%HERE%trellis-gui.exe" --weights "%HERE%TRELLIS.2"
'@
    Set-Content -LiteralPath (Join-Path $DestinationDir "Launch trellis2 local.bat") -Value $Content -Encoding ASCII
}

function Write-DownloadScripts($DestinationDir) {
    Copy-Required (Join-Path $RepoRoot "tools\download_weights.py") $DestinationDir

    $Batch = @'
@echo off
set "HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%download_weights.ps1" %*
if errorlevel 1 pause
'@
    Set-Content -LiteralPath (Join-Path $DestinationDir "Download weights.bat") -Value $Batch -Encoding ASCII

    $PowerShell = @'
param(
    [ValidateSet("huggingface", "hf", "modelscope", "ms")]
    [string] $Source = "huggingface",

    [ValidateSet("all", "trellis", "dino", "birefnet", "background")]
    [string] $Only = "all",

    [switch] $Full,
    [string] $Revision,
    [int] $MaxWorkers = 8
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Here "TRELLIS.2"
$Downloader = Join-Path $Here "download_weights.py"

function Find-Python {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @($py.Source, "-3")
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @($python.Source)
    }
    $python3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python3) {
        return @($python3.Source)
    }
    throw "Python was not found. Install Python 3, then run this script again."
}

$Python = Find-Python
$Exe = $Python[0]
$Prefix = @()
if ($Python.Length -gt 1) {
    $Prefix = $Python[1..($Python.Length - 1)]
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $Exe @Prefix -m pip install -U "huggingface_hub[cli]"
if ($Source -eq "modelscope" -or $Source -eq "ms") {
    & $Exe @Prefix -m pip install -U modelscope
}

$Args = @($Downloader, "--source", $Source, "--output-dir", $OutDir, "--only", $Only, "--max-workers", "$MaxWorkers")
if ($Full) {
    $Args += "--full"
}
if (![string]::IsNullOrWhiteSpace($Revision)) {
    $Args += @("--revision", $Revision)
}

& $Exe @Prefix @Args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Weights are ready in: $OutDir"
Write-Host "You can now launch trellis2 local."
'@
    Set-Content -LiteralPath (Join-Path $DestinationDir "download_weights.ps1") -Value $PowerShell -Encoding ASCII
}

function Copy-ExampleImage($DestinationDir) {
    $ExampleSource = Join-Path $RepoRoot "example_image\images.jpg"
    if (Test-Path -LiteralPath $ExampleSource) {
        Copy-Item -LiteralPath $ExampleSource -Destination (Join-Path $DestinationDir "images.jpg") -Force
        $ExampleDir = Join-Path $DestinationDir "example_image"
        Ensure-Dir $ExampleDir
        Copy-Item -LiteralPath $ExampleSource -Destination $ExampleDir -Force
    }
}

function Write-Readme($DestinationDir, $Backend) {
    $TitleBackend = if ($Backend -eq "cuda") { "CUDA" } else { "Vulkan" }
    $Extra = if ($Backend -eq "cuda") {
        "This package includes CUDA runtime DLLs and requires a compatible NVIDIA driver."
    } else {
        "This package includes the Vulkan backend and does not include CUDA runtime DLLs."
    }
    $Content = @"
trellis2 local Windows x64 $TitleBackend package
Commit: $Commit

Run:
  1. Double-click "Download weights.bat" once.
  2. Double-click "Launch trellis2 local.bat".

Weights:
  No weights are included in this folder.
  The downloader writes to .\TRELLIS.2\, which the GUI auto-detects.
  You can also run:
    powershell -ExecutionPolicy Bypass -File .\download_weights.ps1
    powershell -ExecutionPolicy Bypass -File .\download_weights.ps1 -Source modelscope

Included:
  - trellis-gui.exe
  - trellis2-image-to-gltf.exe (TRELLIS.2-only CLI)
  - pixal3d-image-to-gltf.exe (Pixal3D-only CLI; weights supplied separately)
  - vkmesh.exe for mesh postprocess
  - runtime DLLs for this backend
  - download_weights.py plus Windows wrapper scripts
  - images.jpg for the first preview image

Notes:
  - GUI title: trellis2 local
  - Image picking uses the native Windows file dialog.
  - Image preview supports UTF-8/Chinese paths.
  - $Extra

Outputs are written to viewer_outputs\ next to the app.
"@
    Set-Content -LiteralPath (Join-Path $DestinationDir "README.txt") -Value $Content -Encoding UTF8
}

function Copy-Common($DestinationDir, $BuildDir) {
    foreach ($Name in @(
        "trellis-gui.exe",
        "trellis2-image-to-gltf.exe",
        "pixal3d-image-to-gltf.exe",
        "trellis2_c.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "raylib.dll")) {
        Copy-Required (Join-Path $BuildDir $Name) $DestinationDir
    }
    Copy-Required (Join-Path $BuildDir "vkmesh.exe") $DestinationDir
    Write-Launcher $DestinationDir
    Write-DownloadScripts $DestinationDir
    Copy-ExampleImage $DestinationDir
}

Ensure-Dir $DistDir
Reset-Dir $CudaPackage
Reset-Dir $VulkanPackage

$CudaPathBin = $null
if (![string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
    $CudaPathBin = Join-Path $env:CUDA_PATH "bin"
}

$ExistingRuntimeDirs = @(
    (Join-Path $RepoRoot "dist\trellis2-gui-windows-x64-self-contained-e8eeecf"),
    (Join-Path $RepoRoot "dist\trellis2-gui-windows-x64-vulkan-no-weights-ed2485c"),
    (Join-Path $RepoRoot "dist\trellis2-local-windows-x64-cuda"),
    (Join-Path $RepoRoot "dist\trellis2-local-windows-x64-vulkan"),
    (Join-Path $env:WINDIR "System32"),
    $CudaPathBin
)

Copy-Common $CudaPackage $CudaBuild
Copy-Required (Join-Path $CudaBuild "ggml-cuda.dll") $CudaPackage
foreach ($Name in @("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")) {
    Copy-Required (Find-Required $Name $ExistingRuntimeDirs) $CudaPackage
}
foreach ($Name in @("msvcp140.dll", "ucrtbase.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vulkan-1.dll")) {
    Copy-Required (Find-Required $Name $ExistingRuntimeDirs) $CudaPackage
}
Write-Readme $CudaPackage "cuda"

Copy-Common $VulkanPackage $VulkanBuild
Copy-Required (Join-Path $VulkanBuild "ggml-vulkan.dll") $VulkanPackage
foreach ($Name in @("msvcp140.dll", "ucrtbase.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vulkan-1.dll")) {
    Copy-Required (Find-Required $Name $ExistingRuntimeDirs) $VulkanPackage
}
$Vcomp = Find-Optional "vcomp140.dll" $ExistingRuntimeDirs
if ($Vcomp) {
    Copy-Required $Vcomp $VulkanPackage
}
Write-Readme $VulkanPackage "vulkan"

Write-Host "Created:"
Write-Host "  $CudaPackage"
Write-Host "  $VulkanPackage"
