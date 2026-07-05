# CUDA and GPU requirements

## Current package configuration

This project currently installs:

```text
torch==2.11.0
torchvision==0.26.0
CUDA wheel: cu128
```

The relevant section of `pyproject.toml` is:

```toml
[project]
dependencies = [
    "torch==2.11.0",
    "torchvision==0.26.0",
]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

The package does not automatically select a CUDA build based on the CUDA Toolkit installed on the recipient's PC.

It always requests the PyTorch CUDA 12.8 build.

## Required GPU environment

GPU execution requires:

1. A CUDA-capable NVIDIA GPU
2. A supported Windows or Linux environment
3. A sufficiently recent NVIDIA driver
4. A successful CUDA-enabled PyTorch installation
5. `torch.cuda.is_available()` returning `True`

Official CUDA-capable GPU information:

https://developer.nvidia.com/cuda-gpus

## NVIDIA driver requirements

### Recommended versions

For the current CUDA 12.8 build, use at least:

| OS             | Recommended minimum driver |
| -------------- | -------------------------: |
| Windows x86-64 |                   `570.65` |
| Linux x86-64   |                   `570.26` |

Using the latest available production or Game Ready/Studio driver is recommended.

Official driver download:

https://www.nvidia.com/ja-jp/drivers/

### CUDA 12.x compatibility floor

NVIDIA documents the following minimum versions for CUDA 12.x minor-version compatibility:

| OS             | Compatibility minimum |
| -------------- | --------------------: |
| Windows x86-64 |              `528.33` |
| Linux x86-64   |           `525.60.13` |

These lower versions may operate through CUDA minor-version compatibility, but some newer features can be restricted.

For predictable deployment, use the CUDA 12.8 recommended driver or a newer driver instead of relying on the compatibility floor.

Official references:

* CUDA 12.8 release notes
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/

* CUDA compatibility guide
  https://docs.nvidia.com/deploy/cuda-compatibility/

* CUDA minor-version compatibility
  https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## Local CUDA Toolkit

The application uses the prebuilt PyTorch `cu128` wheel.

Normal use of this application does not compile CUDA source code.

Therefore, the main deployment requirement is a compatible NVIDIA GPU driver and a working CUDA-enabled PyTorch installation.

A separately installed local CUDA Toolkit becomes relevant when:

* building PyTorch from source
* compiling a custom CUDA extension
* using `nvcc` directly
* developing native CUDA code

The version shown by:

```powershell
nvcc --version
```

describes a locally installed CUDA Toolkit.

The version reported by PyTorch is checked with:

```powershell
uv run python -c "import torch; print(torch.version.cuda)"
```

For this project, the expected PyTorch runtime value is:

```text
12.8
```

## Check the NVIDIA driver

Run:

```powershell
nvidia-smi
```

Confirm that:

* an NVIDIA GPU is listed
* the driver is loaded
* the driver version is sufficiently recent

Example fields:

```text
Driver Version: 581.29
CUDA Version: 13.0
```

The important compatibility check for this package is that the installed driver can run the CUDA 12.8 PyTorch build.

A driver that supports a newer CUDA generation can generally run applications built for an older CUDA generation because NVIDIA drivers are backward compatible.

## Check PyTorch CUDA support

In a development checkout:

```powershell
uv run python -c "import torch; print('torch:', torch.__version__); print('runtime CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected result:

```text
torch: 2.11.0+cu128
runtime CUDA: 12.8
available: True
device count: 1
device: NVIDIA ...
```

## Check actual GPU usage

Start the application:

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out" --device 0
```

In another PowerShell window:

```powershell
nvidia-smi -l 1
```

During inference, the Python process should appear and GPU utilization should increase.

## FP16

FP16 can be enabled with:

```powershell
--half
```

Example:

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out" --device 0 --half
```

FP16 can reduce memory consumption and improve speed on some GPUs.

However, some GPU and model combinations can produce invalid values such as `NaN`.

Run the first test without `--half`.

Enable it only after the normal FP32 execution succeeds.

## Older NVIDIA drivers

Preferred solution:

1. Update the NVIDIA driver
2. Reboot the PC
3. Run `nvidia-smi`
4. Reinstall the tool
5. Test `torch.cuda.is_available()`

Driver download:

https://www.nvidia.com/ja-jp/drivers/

## Alternative CUDA wheel

PyTorch 2.11.0 also provides official builds for:

```text
cu126
cu128
cu130
cpu
```

Official version matrix:

https://pytorch.org/get-started/previous-versions/

To use CUDA 12.6 instead of CUDA 12.8, change the uv source definitions before installing.

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

Then reinstall:

```powershell
uv tool install . --force
```

Changing CUDA wheels should be treated as a separate distribution configuration. Do not mix packages from different CUDA indexes in the same environment.

## CPU fallback

Run with:

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out_cpu" --device cpu
```

CPU processing is significantly slower than NVIDIA GPU processing.

The current package still downloads the CUDA-enabled PyTorch build even when `--device cpu` is selected.

A lightweight CPU-only distribution requires changing the PyTorch source to:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then reinstall:

```powershell
uv tool install . --force
```

## Common problems

### `nvidia-smi` is not found

Cause:

* NVIDIA driver is not installed
* NVIDIA driver installation is damaged
* the PC has no NVIDIA GPU

Action:

* install the latest NVIDIA driver
* reboot Windows

### `torch.cuda.is_available()` is `False`

Possible causes:

* CPU-only PyTorch was installed
* NVIDIA driver is missing or too old
* mixed PyTorch package sources
* the tool environment was created before the CUDA source was configured

Action:

```powershell
uv tool uninstall mp4-anomaly
uv tool install . --force
```

Then test again.

### `CUDA out of memory`

Reduce:

```powershell
--batch 1
```

and then:

```powershell
--imgsz 640
```

Example:

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out" `
  --device 0 `
  --batch 1 `
  --imgsz 640
```

### Invalid detections or `NaN`

Remove:

```powershell
--half
```

Run in FP32:

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out" --device 0
```

### Unsupported GPU architecture

Update:

* NVIDIA driver
* PyTorch version
* CUDA wheel selection

Check the current official PyTorch installation matrix:

https://pytorch.org/get-started/locally/

## Official references

* PyTorch local installation
  https://pytorch.org/get-started/locally/

* PyTorch version matrix
  https://pytorch.org/get-started/previous-versions/

* NVIDIA CUDA compatibility
  https://docs.nvidia.com/deploy/cuda-compatibility/

* CUDA 12.8 release notes
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/

* NVIDIA driver downloads
  https://www.nvidia.com/ja-jp/drivers/

* CUDA-capable GPUs
  https://developer.nvidia.com/cuda-gpus
