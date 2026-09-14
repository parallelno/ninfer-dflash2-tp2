param(
    [string]$VcpkgRoot = (Join-Path $PSScriptRoot 'third_party\vcpkg'),
    [int]$Parallel = 8
)

$ErrorActionPreference = 'Stop'

$repo = Join-Path $PSScriptRoot 'ninfer-dflash2-tp2-port'
$vcpkg = Join-Path $VcpkgRoot 'vcpkg.exe'
$pkgConfig = Join-Path $VcpkgRoot 'installed\x64-windows\tools\pkgconf\pkgconf.exe'
$cudaCompiler = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe'

if (-not (Test-Path -LiteralPath $repo -PathType Container)) {
    throw "NInfer source checkout was not found: $repo"
}

if (-not (Test-Path -LiteralPath $VcpkgRoot -PathType Container)) {
    git clone --depth 1 https://github.com/microsoft/vcpkg.git $VcpkgRoot
}

if (-not (Test-Path -LiteralPath $vcpkg -PathType Leaf)) {
    & (Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
    if ($LASTEXITCODE -ne 0) {
        throw 'vcpkg bootstrap failed.'
    }
}

& $vcpkg install 'ffmpeg[avcodec,avformat,swscale]:x64-windows' 'curl:x64-windows' 'pkgconf:x64-windows'
if ($LASTEXITCODE -ne 0) {
    throw 'vcpkg dependency installation failed.'
}

if (-not (Test-Path -LiteralPath $pkgConfig -PathType Leaf)) {
    throw "pkgconf was not installed where expected: $pkgConfig"
}

if (-not (Test-Path -LiteralPath $cudaCompiler -PathType Leaf)) {
    throw "CUDA 13.4 compiler was not found: $cudaCompiler"
}

$env:PKG_CONFIG_PATH = (Join-Path $VcpkgRoot 'installed\x64-windows\lib\pkgconfig')
$build = Join-Path $repo 'build'

cmake -S $repo -B $build -G 'Visual Studio 17 2022' -A x64 `
    "-DCMAKE_CUDA_COMPILER=$cudaCompiler" `
    '-DCMAKE_CUDA_ARCHITECTURES=120a' `
    "-DCMAKE_TOOLCHAIN_FILE=$(Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake')" `
    "-DPKG_CONFIG_EXECUTABLE=$pkgConfig"
if ($LASTEXITCODE -ne 0) {
    throw 'CMake configuration failed.'
}

cmake --build $build --config Release --target ninfer-serve --parallel $Parallel
if ($LASTEXITCODE -ne 0) {
    throw 'NInfer server build failed.'
}

$server = Join-Path $build 'apps\Release\ninfer-serve.exe'
if (-not (Test-Path -LiteralPath $server -PathType Leaf)) {
    throw "NInfer server was not found after build: $server"
}

Write-Host "TP2 DFlash2-capable server built: $server"