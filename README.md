# HYPIR Upscaler v0.4.0

A simplified Windows desktop front end for local HYPIR-SD2 image restoration/upscaling.

## End-user goal

The person using the finished application should not install Python, Git, Conda, ComfyUI, PyTorch, CUDA toolkits, or HYPIR manually.

They install `HYPIR-Upscaler-Setup-0.4.0.exe`, launch HYPIR Upscaler, and the application downloads the required model files on first launch. Inference then runs locally on a CUDA-compatible NVIDIA GPU.

## Building the Windows installer

The canonical build is GitHub Actions. You do **not** need Python installed on your own Windows PC.

1. Create an empty GitHub repository.
2. Upload/push the contents of this folder to it.
3. Open the repository's **Actions** tab.
4. Choose **Build Windows Installer**.
5. Click **Run workflow**.
6. When it finishes, download the `HYPIR-Upscaler-Setup-0.4.0` artifact.
7. Inside that artifact is `HYPIR-Upscaler-Setup-0.4.0.exe`.

The workflow also creates a portable ZIP build.

## Reproducibility

The build uses:

- Windows Server 2022 GitHub runner
- Python 3.10.11
- HYPIR commit `b61d107c6cef38f01a93c7833558869731cfa8c1`
- PyTorch 2.6.0 / torchvision 0.21.0 from the CUDA 12.4 wheel index
- PyInstaller 6.22.0
- pinned runtime Python dependencies in `requirements-runtime.txt`

The workflow verifies the HYPIR checkout SHA and performs an import smoke test before packaging.

## Why this is an installer rather than one giant EXE

PyInstaller's one-folder mode is used internally and Inno Setup turns that folder into a normal Windows installer. This avoids making a huge CUDA/PyTorch one-file executable unpack itself to a temporary directory on every launch.

End users still receive one installer EXE.

## First launch

Model files are stored under:

`%LOCALAPPDATA%\HYPIR-Upscaler\models`

The app downloads:

- `lxq007/HYPIR` → `HYPIR_sd2.pth`
- `stabilityai/stable-diffusion-2-1-base`

The download is only required once unless the model directory is removed.

## Hardware

This build targets 64-bit Windows systems with a CUDA-compatible NVIDIA GPU. The official HYPIR implementation uses CUDA and bfloat16.

## Licensing

HYPIR's upstream repository declares the software non-commercial-only unless written permission is obtained from the authors. Review upstream HYPIR and Stable Diffusion 2.1 licensing before redistributing this application.
