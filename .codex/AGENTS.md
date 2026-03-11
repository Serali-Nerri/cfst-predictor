# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains pipeline modules: `data_loader.py`, `preprocessor.py`, `model_trainer.py`, `evaluator.py`, `predictor.py`, `visualizer.py`, plus `src/utils/` for logging and model I/O helpers.
- Entry points are `train.py` (training/evaluation) and `predict.py` (inference).
- Configuration lives in `config/config.yaml`; prefer config-driven changes over hardcoded values. The current default mainline is `target_mode: psi_over_npl` + `target_transform.type: log` + `n_trials: 200`.
- Data is organized under `data/raw/` and `data/processed/`.
- Generated model artifacts go to `output/`; `logs/` is tracked and stores reusable params, experiment summaries, and the Optuna SQLite study.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python train.py --config config/config.yaml`: run the default `log(psi)` mainline and export artifacts to `output/psi_over_npl_log_original_200`.
- `python predict.py --model output/psi_over_npl_log_original_200 --input data/raw/all.csv --output output/predictions.csv`: run batch prediction with the current default saved model.
- `pyright`: run static type checking (configured by `pyrightconfig.json`).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear docstrings for public classes/functions.
- Use `snake_case` for functions/variables/files, `PascalCase` for classes (for example, `ModelTrainer`, `DataLoader`).
- Keep modules focused: data handling in `data_loader/preprocessor`, training logic in `model_trainer`, metrics in `evaluator`.
- Prefer type hints on new or modified public APIs.

## Testing Guidelines
- Add tests in `tests/` using `test_*.py` naming.
- Use `pytest` style assertions for new coverage; run with `pytest -q`.
- For pipeline changes, include a smoke check by running `train.py --config config/config.yaml` and one `predict.py` command against `output/psi_over_npl_log_original_200`.
- For changes that touch default configs or experiment routing, run `pytest -q tests/test_experiment_configs.py tests/test_train.py`.

## Commit & Pull Request Guidelines
- Current history mixes styles; standardize on Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- Keep commits scoped and imperative (for example, `fix: guard missing features in predictor`).
- PRs should include: purpose, key files changed, config/data assumptions, and validation evidence (metrics such as RMSE/R²/COV, or command output).
- Link related issues and include updated plots/report paths when model behavior changes.

## Documentation Sync
- If you change the default training mainline, update `README.md` in the same change.
- If you change reported baseline metrics or default best-params files, update the relevant report under `doc/`, currently `doc/raw_psi_vs_log_psi_full_run_20260311.md`.
- Do not describe `logs/` as disposable output unless `.gitignore` is updated accordingly.
