# mp4-anomaly

YOLOでタイムラプスMP4を解析し、カテゴリー別の検出数について移動和を計算します。

移動和が閾値を超えた区間について、次のファイルを出力します。

* 検出枠付き確認動画
* 移動和グラフ
* 異常区間ログ
* 全フレーム統計

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

## Quick start

### 1. Install uv

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

PowerShellを開き直し、インストールを確認します。

```powershell
uv --version
```

### 2. Install the tool

プロジェクトを展開したディレクトリで実行します。

```powershell
uv tool install . --force
```

PATHに関する警告が表示された場合:

```powershell
uv tool update-shell
```

PowerShellを開き直してください。

### 3. Run

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out"
```

NVIDIA GPUを明示する場合:

```powershell
mp4-anomaly "D:\LPSE0001.MP4" ".\out" --device 0
```

最初は`--half`を指定せずに実行してください。

## System requirements

* Python 3.11以上
* Windows x86-64またはLinux x86-64
* GPU実行時はCUDA対応NVIDIA GPU
* GPU実行時はCUDA 12.8と互換性のあるNVIDIAドライバー
* 入力MP4と出力ファイルを保存できる十分なディスク容量

現在のパッケージは、次のPyTorch構成を使用します。

```text
PyTorch:    2.11.0
TorchVision: 0.26.0
CUDA wheel: cu128
```

配布先PCにインストールされているCUDA Toolkitのバージョンではなく、このプロジェクトが指定しているPyTorchの`cu128` wheelを使用します。

GPU利用時は、NVIDIAドライバーがCUDA 12.8と互換性を持つ必要があります。

詳細:

* [Installation guide](docs/INSTALLATION.md)
* [CUDA and GPU requirements](docs/CUDA.md)
* [Command-line options](docs/CLI.md)
* [Source structure](docs/ARCHITECTURE.md)

## Documentation

### Installation

uvのインストール、`uv tool`による導入、更新、削除:

```text
docs/INSTALLATION.md
```

### CUDA requirements

GPU、NVIDIAドライバー、CUDA互換性、動作確認:

```text
docs/CUDA.md
```

### Command-line options

コマンドライン引数とデフォルト値:

```text
docs/CLI.md
```

### Source structure

各モジュールと関数の説明:

```text
docs/ARCHITECTURE.md
```

## Development

依存関係を作成します。

```powershell
uv sync
```

ソースから実行します。

```powershell
uv run python .\main.py "D:\LPSE0001.MP4" ".\out"
```

ヘルプ:

```powershell
uv run python .\main.py --help
```

## Build

wheelとsource distributionを作成します。

```powershell
uv build
```

出力:

```text
dist/
├── mp4_anomaly-0.2.0-py3-none-any.whl
└── mp4_anomaly-0.2.0.tar.gz
```

配布時は、`pyproject.toml`を含むプロジェクト一式をZIPで渡し、展開先で次を実行する方法を推奨します。

```powershell
uv tool install . --force
```

これにより、`pyproject.toml`に定義されたPyTorch CUDA 12.8用インデックスが使用されます。

## Official documentation

* uv installation
  https://docs.astral.sh/uv/getting-started/installation/

* uv tools
  https://docs.astral.sh/uv/guides/tools/

* uv documentation
  https://docs.astral.sh/uv/

* PyTorch installation
  https://pytorch.org/get-started/locally/

* PyTorch version matrix
  https://pytorch.org/get-started/previous-versions/

* NVIDIA CUDA compatibility
  https://docs.nvidia.com/deploy/cuda-compatibility/

* CUDA 12.8 release notes
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/

* NVIDIA drivers
  https://www.nvidia.com/ja-jp/drivers/
