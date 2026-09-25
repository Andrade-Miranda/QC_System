# Installation

## Supported environment

The repository is pure Python. The current development and validation work used Linux with Python 3. No GPU is required for the public reproduction scripts. Real-data QC depends on CPU-readable medical-imaging files and the packages in `requirements.txt`.

## Create an environment

```bash
git clone https://github.com/Andrade-Miranda/QC_System.git
cd QC_System
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional external LLM support is configured through `configs/llm_config.yaml`. External providers require explicit runtime opt-in where implemented and environment variables such as `OPENAI_API_KEY`; no key is stored in the repository.

## Configuration

Tracked configs avoid workstation-specific raw-data paths. Configure local data using either:

```bash
cp configs/paths.local.example.yaml configs/paths.local.yaml
# edit RAW_DATASET_ROOT in configs/paths.local.yaml
```

or:

```bash
export AGENTQC_RAW_DATASET_ROOT="/path/to/dataset"
```

`configs/paths.local.yaml` is gitignored and must not be committed.

## Troubleshooting

- Missing raw dataset root: configure `RAW_DATASET_ROOT` or `AGENTQC_RAW_DATASET_ROOT`.
- Missing imaging dependency: reinstall with `python -m pip install -r requirements.txt`.
- LLM unavailable: deterministic QC and public reproduction scripts do not require LLM access.
- Plotting unavailable: install `matplotlib` from `requirements.txt`.

## Lightweight validation

Run the public unit-test suite with the standard library test runner:

```bash
python -m unittest discover -s tests
```

The current public requirements do not require `pytest`; if you install it separately, it is optional rather than the documented test runner.
