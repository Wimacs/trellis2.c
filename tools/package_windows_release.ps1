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

function Copy-Required-As($Source, $Destination) {
    if (!(Test-Path -LiteralPath $Source)) {
        throw "Missing required file: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
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
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
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
    $candidates = @()
    if (![string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
        $venvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path -LiteralPath $venvPython) {
            $candidates += [pscustomobject]@{ Exe = $venvPython; Prefix = @() }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $candidates += [pscustomobject]@{ Exe = $python.Source; Prefix = @() }
    }
    $python3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python3) {
        $candidates += [pscustomobject]@{ Exe = $python3.Source; Prefix = @() }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $candidates += [pscustomobject]@{ Exe = $py.Source; Prefix = @("-3") }
    }

    foreach ($candidate in $candidates) {
        try {
            & $candidate.Exe @($candidate.Prefix) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
            continue
        }
    }
    throw "A working Python 3.9+ was not found. Install Python 3, then run this script again."
}

$Python = Find-Python
$Exe = $Python.Exe
$Prefix = @($Python.Prefix)

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $Exe @Prefix -m pip install -U huggingface_hub
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
if ($Source -eq "modelscope" -or $Source -eq "ms") {
    & $Exe @Prefix -m pip install -U modelscope
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
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

function Copy-LicenseFiles($DestinationDir) {
    Copy-Required (Join-Path $RepoRoot "LICENSE") $DestinationDir
    Copy-Required (Join-Path $RepoRoot "THIRD_PARTY_NOTICES.md") $DestinationDir

    $LicenseDir = Join-Path $DestinationDir "licenses"
    Ensure-Dir $LicenseDir
    $Files = [ordered]@{
        "cgltf-LICENSE.txt" = "3rd\cgltf\LICENSE"
        "eigen-LICENSE.txt" = "3rd\eigen\LICENSE"
        "eigen-COPYING.README.txt" = "3rd\eigen\COPYING.README"
        "eigen-COPYING.MPL2.txt" = "3rd\eigen\COPYING.MPL2"
        "eigen-COPYING.APACHE.txt" = "3rd\eigen\COPYING.APACHE"
        "eigen-COPYING.BSD.txt" = "3rd\eigen\COPYING.BSD"
        "eigen-COPYING.MINPACK.txt" = "3rd\eigen\COPYING.MINPACK"
        "ggml-LICENSE.txt" = "3rd\ggml\LICENSE"
        "meshoptimizer-LICENSE.txt" = "3rd\meshoptimizer\LICENSE.md"
        "raylib-LICENSE.txt" = "3rd\raylib\LICENSE"
        "raylib-glfw-LICENSE.txt" = "3rd\raylib\src\external\glfw\LICENSE.md"
        "stb-LICENSE.txt" = "3rd\stb\LICENSE"
        "xatlas-LICENSE.txt" = "3rd\xatlas\LICENSE"
        "O-Voxel-LICENSE.txt" = "licenses\O-Voxel-LICENSE.txt"
        "xatlas-embedded-LICENSES.txt" = "licenses\xatlas-embedded-LICENSES.txt"
    }
    foreach ($Name in $Files.Keys) {
        Copy-Required-As `
            (Join-Path $RepoRoot $Files[$Name]) `
            (Join-Path $LicenseDir $Name)
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
  DINOv3 is downloaded from a public mirror but remains subject to Meta's
  DINOv3 License; it is not covered by the trellis2.c MIT License.
  Review and preserve the model-card and license files downloaded with each model.
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
  - LICENSE, THIRD_PARTY_NOTICES.md, and licenses\

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
    Copy-LicenseFiles $DestinationDir
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
