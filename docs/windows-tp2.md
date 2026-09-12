# Windows 11 TP2 — сборка и запуск (форк ninfer-tp2-1m, ветка windows-tp2)

Нативная Windows 11 x64 сборка NInfer TP2 для RTX 5060 Ti / RTX 5090 (`sm_120a`) — порт
совместимости из [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows),
наложенный на TP2-базу [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m)
без изменения TP2-поведения.

## Требования

- Windows 11 x64;
- драйвер NVIDIA с поддержкой CUDA 12.8+ (проверено на 581.57, CUDA 13.0);
- Visual Studio 2022 (MSVC 14.4x) — нагрузка «Desktop development with C++»;
- CMake 3.28+, Ninja;
- **CUDA 13.1** (Toolkit 12.8 не собирает `gqa_attention_decode_i8_tiled_kernel`: ptxas 12.8
  кладёт статическую shared-память на 72 байта сверх лимита 48 КиБ). Рабочая копия тулчейна
  собирается в workspace без системной установки — см. «Локальная CUDA 13.1» ниже;
- vcpkg (клонируется в `third_party/vcpkg`).

## Сборка (один раз всё, из workspace)

```powershell
# 1. vcpkg (в workspace)
git clone https://github.com/microsoft/vcpkg third_party/vcpkg
third_party\vcpkg\bootstrap-vcpkg.bat -disableMetrics

# 2. CUDA 13.1 из официальных redist-архивов (без установщика)
#    см. research/PROGRESS.md, раздел «Что делать после перезагрузки» — полный список zips
#    (cuda_nvcc, cuda_cudart, cuda_cccl, cuda_crt, cuda_nvtx, cuda_nvml_dev,
#     cuda_profiler_api, libnvvm, cuda_cuobjdump, cuda_nvdisasm, cuda_nvprune, cuda_cuxxfilt)
#    → распаковать и слить в third_party/cuda-13.1 (bin/, include/, lib/x64, nvvm/)

# 3. Конфигурация + сборка (скрипт сам вызывает vcvars64)
cmd /c scripts\build-windows.cmd all
```

Результат:

```text
build-windows-131\apps\ninfer.exe
build-windows-131\apps\ninfer-serve.exe   (+ avcodec/avformat/avutil/swresample/swscale/libcurl/z DLL рядом)
```

(CUDA runtime статический — `cudart*.dll` не нужен.)

## Запуск

```powershell
# MTP0, TP2
build-windows-131\apps\ninfer.exe models\qwen3_8_27b_nvfp4.ninfer `
  --tp 2 --devices 0,1 --max-context 32768 --kv-capacity auto `
  --prompt "..." --max-new 512 --no-thinking

# MTP3
build-windows-131\apps\ninfer.exe models\qwen3_8_27b_nvfp4.ninfer `
  --tp 2 --devices 0,1 --max-context 32768 --kv-capacity auto `
  --spec mtp --draft-tokens 3 --lm-head-draft --prompt "..." --max-new 512 --no-thinking

# HTTP-сервер
build-windows-131\apps\ninfer-serve.exe models\qwen3_8_27b_nvfp4.ninfer `
  --tp 2 --devices 0,1 --max-context 32768 --kv-capacity auto `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

## Диагностика и бенчмарки TP2 (workspace)

```powershell
# сборка (nvcc 13.1 из workspace)
build\diagnostics\p2p_probe.exe 0 1          # P2P доступность + стейджинг-полоса/латентность
build\diagnostics\transport_probe.exe 0 1     # eager/graph-легальность транспорта
build\diagnostics\reduce_bench.exe 0 1        # реальный allreduce_sum: 10 KiB, 1 и 128 редукций, graph
```

## Отличия от Linux-сборки

- Зависимости (FFmpeg/libcurl/zlib) приходят из vcpkg-манифеста, не pkg-config; CUDA runtime
  линкуется статически (`CUDA::cudart_static`).
- Читатель артефактов использует `CreateFileW`/`MapViewOfFile` + overlapped `ReadFile` с
  `FILE_FLAG_NO_BUFFERING` — контракт выравнивания 4096 байт идентичен POSIX `O_DIRECT`/`pread`.
- MSVC-специфика: TMA-дескрипторы передаются в ядро через device-указатель (by-value
  `alignas(128)` параметр невозможен — C2719); plan-классы имеют явные тела move-операторов
  (MSVC 19.44 не эмитит out-of-line `= default` явные специализации); `const dim3` вместо
  `constexpr dim3` (3 файла); `<stdlib.h>` перед `nvtx3` для старых тулчейнов.
- `--tp 1` и все Linux-маршруты не затронуты: все Windows-ветки под `#ifdef _WIN32`.

## Известные особенности на этой машине (2× 5060 Ti, Z690)

- P2P недоступен (WDDM/GeForce): `cudaDeviceCanAccessPeer` = 0 в обе стороны → TP2 идёт по
  штатному host-staged пути (это нормальный режим форка, не деградация).
- GPU1 сидит на PCIe 3.0 x4 через чипсет (GPU0 — PCIe 5.0 x16 от CPU) — асимметрия полосы.
- Декод communication-bound: 128 коллективов ≈ 35 мс/ток из ~61 мс (MTP0); утилизация GPU при
  этом низкая (8–15 %), чипы ~500–700 МГц, питание 13–14 W. Ускорение возможно только
  сокращением/сращением коллективов (Phase 2), не «разгоном» карт.
- WDDM-утечка: после кросс-девайсных стейджинг-прогонов GPU1 может показать ~14.5 ГиБ занятыми
  без процесса-владельца — лечится перезагрузкой; перед длинными прогонами проверять
  `nvidia-smi --query-gpu=memory.used --format=csv`.
