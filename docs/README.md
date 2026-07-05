# detec-anomaly

タイムラプスMP4をUltralytics YOLOで解析し、対象カテゴリーの検出数について移動和を計算するコマンドラインツールです。

移動和が閾値を超えた区間について、検出枠付きの確認動画、CSVログ、移動和グラフを出力します。

## Features

- MP4の全フレームをYOLOで解析
- 人、車両などのカテゴリー別検出数を集計
- 指定時間幅で移動和を計算
- 閾値超過区間を異常として検出
- 異常発生前後を含む確認動画を出力
- 検出枠、信頼度、実時間、動画内時間を確認動画へ表示
- 全フレーム統計をCSVへ出力
- 異常区間をCSVへ出力
- 移動和グラフをPNGへ出力
- NVIDIA GPUまたはCPUで実行可能

## Detected classes

デフォルトでは、COCOデータセットの次のクラスを検出します。

| Class ID | Class |
|---:|---|
| `0` | person |
| `1` | bicycle |
| `2` | car |
| `3` | motorcycle |
| `5` | bus |
| `7` | truck |

## Output

```text
OUTPUT/
├── clips/
│   ├── anomaly_0001.mp4
│   ├── anomaly_0002.mp4
│   └── ...
├── anomalies.csv
├── moving_sum.csv
└── moving_sum.png
```

| File | Description |
|---|---|
| `clips/anomaly_XXXX.mp4` | 検出枠付きの異常区間確認動画 |
| `anomalies.csv` | 異常区間、ピーク値、時刻、出力動画パス |
| `moving_sum.csv` | フレームごとの検出数と移動和 |
| `moving_sum.png` | カテゴリー別移動和グラフ |

# Installation

## System requirements

推奨環境は次のとおりです。

```text
OS:       Windows 10/11 x86-64
Python:   3.11以上
GPU:      CUDA対応NVIDIA GPU
PyTorch:  2.11.0
CUDA:     cu128版PyTorch
```

GPUを使用する場合は、CUDA 12.8ランタイムと互換性のあるNVIDIAドライバーが必要です。

次のコマンドでNVIDIA GPUとドライバーを確認できます。

```powershell
nvidia-smi
```

`nvidia-smi`に表示される`CUDA Version`は、NVIDIAドライバーが対応可能なCUDAバージョンです。

PCへCUDA Toolkitを別途インストールしていなくても、CUDA対応PyTorch wheelと互換性のあるNVIDIAドライバーがあれば通常は実行できます。

CPUでも実行できますが、GPU実行より大幅に時間がかかります。

CUDAの詳細は次を参照してください。

- [CUDA and GPU requirements](docs/CUDA.md)

## 1. Install uv

uv公式インストーラーを使用します。

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、PowerShellを開き直します。

```powershell
uv --version
```

PATHに関する警告が表示された場合は、次を実行します。

```powershell
uv tool update-shell
```

実行後、PowerShellを開き直してください。

### Official uv documentation

- uv installation  
  https://docs.astral.sh/uv/getting-started/installation/

- uv tools  
  https://docs.astral.sh/uv/guides/tools/

- uv command reference  
  https://docs.astral.sh/uv/reference/cli/

## 2. Install the CUDA version

配布されたwheelを`dist`ディレクトリへ配置します。

wheelのファイル名では、プロジェクト名のハイフンがアンダースコアへ変換されます。

```text
detec-anomaly
    ↓
detec_anomaly-VERSION-py3-none-any.whl
```

プロジェクトディレクトリへ移動します。

```powershell
cd C:\path\to\detec-anomaly
```

最新のwheelを取得します。

```powershell
$wheel = Get-ChildItem .\dist\detec_anomaly-*.whl |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
```

CUDA 12.8版PyTorchを使用してインストールします。

```powershell
uv tool install --force `
  --index https://download.pytorch.org/whl/cu128 `
  $wheel.FullName
```

1行で指定する場合は、実際のwheel名を指定します。

```powershell
uv tool install --force --index https://download.pytorch.org/whl/cu128 .\dist\detec_anomaly-VERSION-py3-none-any.whl
```

インストール後、コマンドを確認します。

```powershell
detec-anomaly --help
```

## 3. Install from the source directory

`pyproject.toml`が存在するプロジェクトディレクトリから直接インストールすることもできます。

```powershell
cd C:\path\to\detec-anomaly

uv tool install --force `
  --index https://download.pytorch.org/whl/cu128 `
  .
```

インストール後に確認します。

```powershell
detec-anomaly --help
```

## 4. CPU-only installation

NVIDIA GPUを使用しない場合は、CPU版PyTorchのインデックスを指定します。

```powershell
$wheel = Get-ChildItem .\dist\detec_anomaly-*.whl |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv tool install --force `
  --index https://download.pytorch.org/whl/cpu `
  $wheel.FullName
```

CPUで実行する場合は、必ず`--device cpu`を指定します。

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out_cpu" `
  --device cpu `
  --threshold 15
```

## 5. Reinstall or update

新しいwheelを受け取った場合は、`--force`を付けて再インストールします。

```powershell
$wheel = Get-ChildItem .\dist\detec_anomaly-*.whl |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv tool install --force `
  --index https://download.pytorch.org/whl/cu128 `
  $wheel.FullName
```

キャッシュの影響を除外したい場合は、`--no-cache`を追加します。

```powershell
uv tool install --force --no-cache `
  --index https://download.pytorch.org/whl/cu128 `
  $wheel.FullName
```

## 6. Uninstall

```powershell
uv tool uninstall detec-anomaly
```

インストール済みのツールを確認します。

```powershell
uv tool list
```

# YOLO model weights

## Automatic download

デフォルトモデルは次のとおりです。

```text
yolo11m.pt
```

`yolo11m.pt`がローカルに存在しない場合、Ultralyticsは通常、初回実行時に公式の重みファイルを自動ダウンロードします。

そのため、初回実行時にはインターネット接続が必要です。

初回ダウンロード後は、取得済みの重みファイルが再利用されます。

## Offline installation

インターネットへ接続できないPCでは、`yolo11m.pt`を事前にダウンロードし、配布物へ含めてください。

配置例:

```text
detec-anomaly/
├── models/
│   └── yolo11m.pt
├── dist/
├── docs/
├── README.md
└── pyproject.toml
```

実行時に重みファイルのパスを指定します。

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --model ".\models\yolo11m.pt" `
  --threshold 15
```

独自に学習した重みも同様に指定できます。

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --model "D:\models\best.pt" `
  --threshold 15
```

重みファイルはPythonパッケージの依存関係としてインストールされるものではありません。

公式モデル名を指定した場合はUltralyticsが初回利用時に取得し、オフライン環境では手動配置が必要です。

# Usage

## Basic usage

```powershell
detec-anomaly INPUT OUTPUT [OPTIONS]
```

例:

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --threshold 15
```

## Recommended GPU command

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --model yolo11m.pt `
  --conf 0.25 `
  --imgsz 960 `
  --device 0 `
  --batch 1 `
  --real-seconds-per-frame 2 `
  --window-seconds 60 `
  --threshold 15 `
  --pre-seconds 5 `
  --post-seconds 5 `
  --review-fps 10
```

## Batch size 2

GPUメモリに余裕がある場合は、`--batch 2`を試せます。

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out_batch2" `
  --device 0 `
  --batch 2 `
  --threshold 15
```

`CUDA out of memory`が発生した場合は、`--batch 1`へ戻してください。

## Lower-memory configuration

GPUメモリ不足が発生する場合は、画像サイズを小さくします。

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out_low_memory" `
  --device 0 `
  --batch 1 `
  --imgsz 640 `
  --threshold 15
```

## CPU execution

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out_cpu" `
  --device cpu `
  --batch 1 `
  --threshold 15
```

## Show help

```powershell
detec-anomaly --help
```

# Command-line options

| Option | Default | Description |
|---|---:|---|
| `INPUT` | required | 入力MP4ファイル |
| `OUTPUT` | required | 出力ディレクトリ |
| `--model` | `yolo11m.pt` | YOLO重みファイルまたは公式モデル名 |
| `--conf` | `0.25` | 検出結果として採用する最低信頼度 |
| `--imgsz` | `960` | 推論時の画像サイズ |
| `--device` | `0` | CUDAデバイス番号または`cpu` |
| `--batch` | `1` | 推論バッチサイズ |
| `--half` | disabled | FP16推論を有効化 |
| `--real-seconds-per-frame` | `2.0` | 1フレームが表す実時間 |
| `--window-seconds` | `60.0` | 移動和の時間幅 |
| `--threshold` | `10` | 異常判定の閾値 |
| `--pre-seconds` | `5.0` | 異常開始前に動画へ含める時間 |
| `--post-seconds` | `5.0` | 異常終了後に動画へ含める時間 |
| `--review-fps` | `10.0` | 確認動画のフレームレート |
| `-h`, `--help` | — | ヘルプを表示 |

実行例では`--threshold 15`を使用しています。

プログラム上のデフォルト値は`10`です。デフォルト自体を`15`へ変更する場合は、`src/mp4_anomaly/cli.py`の設定も変更してください。

```python
parser.add_argument(
    "--threshold",
    type=int,
    default=15,
    help="Anomaly threshold; the moving sum must exceed it",
)
```

異常判定条件は次のとおりです。

```text
moving sum > threshold
```

`--threshold 15`の場合、移動和が`16`以上になった時点で異常と判定されます。

# Checking CUDA

## Check the NVIDIA driver

```powershell
nvidia-smi
```

推論実行中に次の状態であれば、GPUが使用されています。

- `python.exe`がProcessesへ表示される
- `GPU-Util`が上昇する
- GPUメモリ使用量が増加する
- GPUのPerformance Stateが`P0`などになる

## Check the installed tool environment

uv toolのインストール先を確認します。

```powershell
uv tool dir
```

Windows環境では、通常は次のような場所にツール環境が作成されます。

```text
%APPDATA%\uv\tools\detec-anomaly\
```

PyTorchの状態を確認します。

```powershell
$toolPython = Join-Path (uv tool dir) "detec-anomaly\Scripts\python.exe"

& $toolPython -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

CUDA版が正常にインストールされている場合の例:

```text
torch: 2.11.0+cu128
CUDA runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce ...
```

次のように表示される場合はCPU版PyTorchです。

```text
torch: 2.11.0+cpu
CUDA runtime: None
CUDA available: False
```

CPU版になっていた場合は、CUDAインデックスを明示して再インストールします。

```powershell
uv tool uninstall detec-anomaly

$wheel = Get-ChildItem .\dist\detec_anomaly-*.whl |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv tool install --force --no-cache `
  --index https://download.pytorch.org/whl/cu128 `
  $wheel.FullName
```

# Troubleshooting

## `Invalid CUDA 'device=0' requested`

表示例:

```text
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

主な原因:

- CPU版PyTorchがインストールされている
- NVIDIAドライバーがインストールされていない
- NVIDIAドライバーが古い
- GPUを認識できていない

CUDA版PyTorchを指定して再インストールします。

```powershell
uv tool uninstall detec-anomaly

$wheel = Get-ChildItem .\dist\detec_anomaly-*.whl |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv tool install --force --no-cache `
  --index https://download.pytorch.org/whl/cu128 `
  $wheel.FullName
```

## `CUDA out of memory`

バッチサイズを下げます。

```powershell
--batch 1
```

それでも不足する場合は、画像サイズを下げます。

```powershell
--imgsz 640
```

実行例:

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --device 0 `
  --batch 1 `
  --imgsz 640 `
  --threshold 15
```

## Invalid detections or `NaN`

`--half`を付けている場合は外してください。

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --device 0 `
  --threshold 15
```

## YOLO model download fails

考えられる原因:

- インターネットへ接続できない
- プロキシやファイアウォールで遮断されている
- GitHubのダウンロード先へ接続できない
- モデル名が間違っている

重みファイルを別のPCで取得し、ローカルパスを指定してください。

```powershell
detec-anomaly `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --model ".\models\yolo11m.pt" `
  --threshold 15
```

# Development

## Project structure

```text
detec-anomaly/
├── README.md
├── main.py
├── pyproject.toml
├── docs/
│   ├── INSTALLATION.md
│   ├── CUDA.md
│   ├── CLI.md
│   └── ARCHITECTURE.md
└── src/
    └── mp4_anomaly/
        ├── __init__.py
        ├── __main__.py
        ├── main.py
        ├── cli.py
        ├── settings.py
        ├── models.py
        ├── analysis.py
        ├── video.py
        ├── events.py
        ├── exporters.py
        └── app.py
```

## Create the development environment

```powershell
uv sync
```

## Run from source

```powershell
uv run python .\main.py `
  "D:\LPSE0001.MP4" `
  ".\out" `
  --device 0 `
  --batch 1 `
  --threshold 15
```

## Build

Gitタグを使用してバージョンを決定した後、ビルドします。

```powershell
git tag -a v0.1.0 -m "Release v0.1.0"
uv build
```

出力例:

```text
dist/
├── detec_anomaly-0.1.0-py3-none-any.whl
└── detec_anomaly-0.1.0.tar.gz
```

タグが現在のコミットより前にある場合、開発バージョン形式になることがあります。

```text
detec_anomaly-0.1.1.dev1+gXXXXXXXX.dYYYYMMDD-py3-none-any.whl
```

## Install the locally built wheel

```powershell
$wheel = Get-ChildItem .\dist\detec_anomaly-*.whl |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv tool install --force `
  --index https://download.pytorch.org/whl/cu128 `
  $wheel.FullName
```

# Documentation

- [Installation](docs/INSTALLATION.md)
- [CUDA and GPU requirements](docs/CUDA.md)
- [Command-line options](docs/CLI.md)
- [Source structure](docs/ARCHITECTURE.md)

# Official links

## uv

- Installation  
  https://docs.astral.sh/uv/getting-started/installation/

- Tool installation  
  https://docs.astral.sh/uv/guides/tools/

- Package indexes  
  https://docs.astral.sh/uv/concepts/indexes/

- Command reference  
  https://docs.astral.sh/uv/reference/cli/

## PyTorch

- Installation  
  https://pytorch.org/get-started/locally/

- Previous versions  
  https://pytorch.org/get-started/previous-versions/

## Ultralytics

- Documentation  
  https://docs.ultralytics.com/

- YOLO11  
  https://docs.ultralytics.com/models/yolo11/

- Object detection  
  https://docs.ultralytics.com/tasks/detect/

## NVIDIA

- Driver download  
  https://www.nvidia.com/ja-jp/drivers/

- CUDA compatibility  
  https://docs.nvidia.com/deploy/cuda-compatibility/