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

1. データの生成: [generate_all_synthetic_data.ipynb](./notebook/synthetic_data/generate_all_synthetic_data.ipynb)
2. 図の作成: [plot_figures.ipynb](./notebook/synthetic_data/plot_figures.ipynb)

### Community Earth System Model Large Ensemble (CESM-LE)

1. データのダウンロード: [download_cesm_le_data.ipynb](./notebook/cesm_le/download_cesm_le_data.ipynb)
2. metrics の計算: [compute_all_metrics.ipynb](./notebook/cesm_le/compute_all_metrics.ipynb)
3. 図の作成: [plot_figures.ipynb](./notebook/cesm_le/plot_figures.ipynb)
