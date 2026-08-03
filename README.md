# Quantifying internal variability in large-ensemble climate data with the Wasserstein distance

## 環境構築

1. `apt-get` などで以下のライブラリがインストールされていることを確認する (`build-essential`, `libgeos-dev`, `libproj-dev`, `proj-data`)
2. python のパッケージマネージャ `uv` がインストールされていることを確認する ([uv 公式レポジトリ](https://github.com/astral-sh/uv))
3. レポジトリのルートディレクトリに移動してから行うこと

    ```sh
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.11
    uv venv .venv-wasserstein_internal_variability_metric --python 3.11
    uv tool install poetry

    source .venv-wasserstein_internal_variability_metric/bin/activate
    poetry config virtualenvs.create false --local
    poetry config virtualenvs.in-project false --local
    poetry install

    poetry env info  # Path が .venv-wasserstein_internal_variability_metric であること
    python -m ipykernel install --user --name wasserstein_internal_variability_metric --display-name "wasserstein_internal_variability_metric (.venv)"
    ```

## 実験方法

### Synthetic Climate Data

### Community Earth System Model Large Ensemble (CESM-LE)
