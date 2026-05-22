# Windows build env for Handy (fork: thinkpload/handy-video-transcribe).
# Dot-source from a PowerShell session, then run `bun run tauri dev` / `cargo build`.
#
#   . .\scripts\dev-env.ps1
#   bun run tauri dev
#
# Or run a one-shot command:
#   .\scripts\dev-env.ps1 -Run 'bun run tauri dev'
#
# Why this exists:
#  - whisper-rs-sys builds whisper.cpp via CMake; ggml-vulkan spawns a nested
#    cmake for `vulkan-shaders-gen` that needs MSVC in PATH -> we load
#    VS Dev Shell.
#  - whisper.cpp nested build paths blow past Windows MAX_PATH (260) when
#    target/ lives deep under the source tree -> we point CARGO_TARGET_DIR
#    at a short absolute path.
#  - Ninja generator avoids the MSBuild-vs-ExternalProject toolchain
#    inheritance bug; ninja is installed via `winget install Ninja-build.Ninja`.

[CmdletBinding()]
param(
    [string]$Run,
    [string]$TargetDir = 'D:\t\handy',
    [string]$VsDevShell = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $VsDevShell)) {
    throw "VS Dev Shell not found at: $VsDevShell. Install 'Build Tools for Visual Studio 2022' with the 'Desktop development with C++' workload."
}

# Load MSVC x64 toolchain into the current session (cl.exe, link.exe, vcvars).
& $VsDevShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null

# Short CARGO_TARGET_DIR so whisper.cpp nested CMake paths fit under MAX_PATH.
if (-not (Test-Path $TargetDir)) { New-Item -ItemType Directory -Path $TargetDir | Out-Null }
$env:CARGO_TARGET_DIR = $TargetDir

# Ninja generator for the cmake crate (whisper-rs-sys, ort-sys, etc).
$env:CMAKE_GENERATOR = 'Ninja'

# Make ninja from winget visible.
$ninjaLinks = 'C:\Users\gllex\AppData\Local\Microsoft\WinGet\Links'
if ((Test-Path "$ninjaLinks\ninja.exe") -and ($env:Path -notlike "*$ninjaLinks*")) {
    $env:Path = "$ninjaLinks;$env:Path"
}

Write-Host "Handy dev env ready:" -ForegroundColor Green
Write-Host "  CARGO_TARGET_DIR = $env:CARGO_TARGET_DIR"
Write-Host "  CMAKE_GENERATOR  = $env:CMAKE_GENERATOR"
Write-Host "  cl.exe           = $((Get-Command cl.exe -ErrorAction SilentlyContinue).Source)"
Write-Host "  ninja            = $((Get-Command ninja.exe -ErrorAction SilentlyContinue).Source)"

if ($Run) {
    Write-Host "`n> $Run" -ForegroundColor Cyan
    Invoke-Expression $Run
}
