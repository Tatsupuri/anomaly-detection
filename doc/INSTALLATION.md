# Installation

## Supported environment

The recommended environment is:

```text
OS:       Windows 10/11 x86-64 or Linux x86-64
Python:   3.11 or later
GPU:      NVIDIA CUDA-capable GPU
Driver:   CUDA 12.8-compatible NVIDIA driver
Package:  PyTorch 2.11.0 + cu128
```

CPU execution is available with:

```powershell
--device cpu
```

However, the current package configuration still installs the CUDA-enabled PyTorch package.

## 1. Install uv

Official documentation:

https://docs.astral.sh/uv/getting-started/installation/

### Windows PowerShell

Use the official standalone installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell.

Verify the installation:

```powershell
uv --version
```

### Windows with WinGet

```powershell
winget install --id=astral-sh.uv -e
```

Close and reopen PowerShell, then verify:

```powershell
uv --version
```

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell and verify:

```bash
uv --version
```

The current CUDA-enabled project configuration is intended for Windows or Linux with an NVIDIA GPU. macOS does not support NVIDIA CUDA execution.

## 2. Obtain the project

Extract the distributed ZIP file.

Example:

```powershell
Expand-Archive .\mp4-anomaly.zip
cd .\mp4-anomaly
```

The directory must contain:

```text
pyproject.toml
main.py
src/
```

## 3. Check the NVIDIA driver

Run:

```powershell
nvidia-smi
```

If the command is not found, or no GPU is displayed, install or update the NVIDIA driver.

Official driver download:

https://www.nvidia.com/ja-jp/drivers/

Detailed CUDA requirements:

```text
docs/CUDA.md
```

## 4. Install with uv tool

Run this command in the directory containing `pyproject.toml`:

```powershell
uv tool install . --force
```

`uv tool install` creates an isolated Python environment and installs the command registered by the package.

The command installed by this project is:

```text
mp4-anomaly
```

If uv reports that its executable directory is not on `PATH`, run:

```powershell
uv tool update-shell
```

Close and reopen PowerShell.

Official uv tool documentation:

https://docs.astral.sh/uv/guides/tools/

## 5. Verify the installation

```powershell
mp4-anomaly --help
```

The help output should show:

```text
input
output
--model
--conf
--imgsz
--device
--batch
--half
--real-seconds-per-frame
--window-seconds
--threshold
--pre-seconds
--post-seconds
--review-fps
```

## 6. Verify PyTorch and CUDA

Because a uv tool is installed in an isolated environment, first locate the environment:

```powershell
uv tool dir
```

For a simple application-level check, run a small input file with:

```powershell
mp4-anomaly "D:\sample.mp4" ".\out_test" --device 0
```

For development installations, use:

```powershell
uv run python -c "import torch; print('torch:', torch.__version__); print('runtime CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output for GPU execution:

```text
torch: 2.11.0+cu128
runtime CUDA: 12.8
CUDA available: True
GPU: NVIDIA ...
```

## 7. Run the tool

Basic command:

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out"
```

Recommended initial command:

```powershell
mp4-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --device 0 `
  --batch 1
```

Full example:

```powershell
mp4-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --model yolo11m.pt `
  --conf 0.25 `
  --imgsz 960 `
  --device 0 `
  --batch 1 `
  --real-seconds-per-frame 2 `
  --window-seconds 60 `
  --threshold 10 `
  --pre-seconds 5 `
  --post-seconds 5 `
  --review-fps 10
```

Do not enable FP16 during the first test.

After confirming stable operation:

```powershell
mp4-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out_half" `
  --device 0 `
  --batch 1 `
  --half
```

If many invalid detections are skipped, remove `--half`.

## Updating the tool

After receiving a new source version:

```powershell
uv tool install . --force
```

To upgrade uv itself when installed by the standalone installer:

```powershell
uv self update
```

## Uninstalling the tool

```powershell
uv tool uninstall mp4-anomaly
```

List installed tools:

```powershell
uv tool list
```

## Development installation

Create the project environment:

```powershell
uv sync
```

Run without installing as a tool:

```powershell
uv run python .\main.py "D:\LPSE0001.MP4" ".\out"
```

Run the package entry point:

```powershell
uv run python -m mp4_anomaly "D:\LPSE0001.MP4" ".\out"
```

## Build and install a wheel

Build:

```powershell
uv build
```

The build products are written to:

```text
dist/
```

For this project, distributing the complete source directory and using:

```powershell
uv tool install .
```

is preferred because the source `pyproject.toml` contains the custom PyTorch CUDA package index.

## Official links

* uv installation
  https://docs.astral.sh/uv/getting-started/installation/

* uv tool guide
  https://docs.astral.sh/uv/guides/tools/

* uv command reference
  https://docs.astral.sh/uv/reference/cli/

* uv Python installation
  https://docs.astral.sh/uv/guides/install-python/

* PyTorch local installation
  https://pytorch.org/get-started/locally/

* PyTorch previous versions
  https://pytorch.org/get-started/previous-versions/
