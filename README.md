# Quantifying internal variability in large-ensemble climate data with the Wasserstein distance

## Environment setup

1. Ensure that the following system libraries are installed (e.g., via `apt-get`): `build-essential`, `libgeos-dev`, `libproj-dev`, and `proj-data`.
2. Ensure that the Python package manager `uv` is installed ([official uv repository](https://github.com/astral-sh/uv)).
3. Run the following commands from the repository root directory:

    ```sh
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.11
    uv venv .venv-wasserstein_internal_variability_metric --python 3.11
    uv tool install poetry

    source .venv-wasserstein_internal_variability_metric/bin/activate
    poetry config virtualenvs.create false --local
    poetry config virtualenvs.in-project false --local
    poetry install

    python -m ipykernel install --user --name wasserstein_internal_variability_metric --display-name "wasserstein_internal_variability_metric (.venv)"
    ```

## Running experiments

- Run the following notebooks in `wasserstein_internal_variability_metric (.venv)`.
- The total data size is about 54 GB.

### Synthetic Climate Data

1. Generate data: [generate_all_synthetic_data.ipynb](./notebook/synthetic_data/generate_all_synthetic_data.ipynb)
2. Create figures: [plot_figures.ipynb](./notebook/synthetic_data/plot_figures.ipynb)

### Community Earth System Model Large Ensemble (CESM-LE)

1. Download data: [download_cesm_le_data.ipynb](./notebook/cesm_le/download_cesm_le_data.ipynb)
2. Compute metrics: [compute_all_metrics.ipynb](./notebook/cesm_le/compute_all_metrics.ipynb)
3. Create figures: [plot_figures.ipynb](./notebook/cesm_le/plot_figures.ipynb)

## License

This repository is distributed under the [GNU General Public License v3.0](./LICENSE) (GPL-3.0).

Copyright (C) 2026 Yuki Yasuda and Shoichiro Kido.

It includes code derived from Sane, Fox-Kemper, and Ullman (JGR Oceans, [DOI: 10.1029/2023JC020101](https://doi.org/10.1029/2023JC020101); Zenodo archive [DOI: 10.5281/zenodo.15849583](https://doi.org/10.5281/zenodo.15849583)), primarily in `src/metrics/sane2024.py`. See [LICENSE](./LICENSE) for attribution details.
