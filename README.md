# CFST柱极限承载力预测 - 可插拔Backbone流水线（默认 XGBoost）

## 项目概述

本项目构建了一个面向混凝土填充钢管（CFST）柱极限承载力预测的机器学习流水线，当前默认 `backbone` 为 `XGBoost`，并保持既有 XGBoost 训练/推理路径兼容，覆盖以下当前已实现的能力：

- 从已处理的特征 CSV 加载数据并提取报告目标列
- 由 `scripts/compute_feature_parameters.py` 离线生成 `Npl (kN)`、`eta_u = Nexp / Npl`、`r = (Nexp - Npl) / Npl`、`b/h`、`L/h`、`axial_flag`、`section_family` 等派生列
- 先划分训练/测试集，再进行预处理，避免预处理数据泄漏
- 按配置剔除指定特征列
- 训练默认 `XGBoost` 回归器（当前主线实现）
- 可选使用 `Optuna` 做超参数搜索
- 在交叉验证各折内部切验证集支持 `early_stopping_rounds`
- 按 `config.cv` 中的 `n_splits` / `shuffle` / `random_state` 执行交叉验证
- 以 `CV` 复合目标选择模型，并在 `train_full` 上重训最终模型
- 输出训练/测试指标、交叉验证结果、可比较的 `regime_analysis`、图表与模型产物
- 从已保存模型目录加载模型并进行 CSV 批量预测

当前默认主线已经切换为：

- `target_mode: eta_u_over_npl`
- `target_transform.enabled: true`
- `target_transform.type: log`
- `model.keml.enabled: true`
- `model.n_trials: 100`
- 默认输出目录：`output/eta_u_over_npl_log_original_default_optuna100`

## 当前实现的核心特性

- **模块化结构**：`src/` 下按数据加载、预处理、训练、评估、预测、可视化拆分
- **严格参数入口**：默认 XGBoost 参数从 `config.model.params` 读取
- **上下文隔离的最优参数复用**：通过 `context_hash` 约束 `logs/best_params_*.json` 的复用范围
- **目标空间分离**：训练可在 `eta_u` / `r` 空间进行，最终统一回到 `Nexp` 空间报告
- **多指标评估**：当前实现 `RMSE`、`MAE`、`R²`、`MAPE`、`COV`
- **可比较的 regime analysis**：先在训练集拟合 regime schema，再对 train/test 共用同一套区间
- **训练产物保存**：保存模型、预处理器、特征名、训练元数据与评估报告

数据集字段、几何统一口径与原始数据说明见：`doc/DATA_README.md`。

## 环境要求

- Python 3.8+
- 推荐在虚拟环境中运行

依赖见 `requirements.txt`，主要包括：

- `pandas`
- `numpy`
- `xgboost`（当前默认 backbone）
- `scikit-learn`
- `optuna`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `joblib`

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 项目结构

```text
cfst_predictitor/
├── config/
│   ├── config.yaml
│   └── experiments/
│       ├── compact_group_B_iterations/
│       └── ...
├── data/
│   ├── raw/
│   └── processed/
├── doc/
├── logs/
│   ├── experiments/
│   │   └── compact_group_B_iterations/
│   └── ...
├── output/
│   ├── experiments/
│   │   └── compact_group_B_iterations/
│   └── ...
├── plan/
├── scripts/
├── src/
│   ├── data_loader.py
│   ├── domain_features.py
│   ├── evaluator.py
│   ├── model_trainer.py
│   ├── predictor.py
│   ├── preprocessor.py
│   ├── splitting.py
│   ├── visualizer.py
│   └── utils/
├── tests/
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## 训练流程

当前训练脚本 `train.py` 的真实流程为：

1. 读取 `config/config.yaml`
2. 从已处理的特征 CSV 加载报告目标 `Nexp (kN)`，并按 `target_mode` 构造训练目标
3. 训练前假定 `scripts/compute_feature_parameters.py` 已离线生成 `Npl / eta_u / r / b/h / L/h / axial_flag / section_family`
4. 先划分 `train/test`
5. 在 `train_full` 上执行 `CV` / `Optuna`；各 fold 内部按 `validation_size` 切验证集用于早停
6. 使用 `CV` 复合目标选择参数，并根据各 fold `best_iteration` 选取最终 `n_estimators`
7. 仅在完整 `train_full` 上 `fit` 预处理器与最终模型，不再永久留出最终验证集
8. 统一在 `Nexp` 空间计算训练 / 测试 / CV 指标
9. 在训练集拟合 regime schema，并对 train/test 应用同一套 schema
10. 保存模型、元数据、评估报告与图表

运行训练：

```bash
python train.py --config config/config.yaml
```

以上命令会直接运行当前默认主线：

- `target_mode: eta_u_over_npl`
- `target_transform.enabled: true`
- `target_transform.type: log`
- `model.keml.enabled: true`
- `n_trials: 100`
- 默认输出目录：`output/eta_u_over_npl_log_original_default_optuna100`

指定输出目录：

```bash
python train.py --config config/config.yaml --output output/model_run
```

## 配置说明

主配置文件为 `config/config.yaml`。

为了说明当前代码实际支持的配置项，仓库还提供了一份带注释的示例文件：`config/config.example.yaml`。

相关补充文档：
- CFST 字段、特征与参数计算说明：`doc/CFST字段与特征说明.md`
- 数据集字段、几何统一口径与原始数据说明：`doc/DATA_README.md`

关键配置项：

```yaml
data:
  file_path: "data/processed/final_feature_parameters_raw.csv"
  target_column: "Nexp (kN)"
  target_mode: "eta_u_over_npl"
  target_transform:
    enabled: true
    type: "log"  # 默认主线；实验支持还包括 boxcox_<lambda>，例如 boxcox_0.50
  columns_to_drop:
    - "b (mm)"
    - "h (mm)"
    - "r0 (mm)"
    - "t (mm)"
    - "L (mm)"
    - "lambda"
  test_size: 0.2
  random_state: 42
  sample_weight:
    enabled: false
    strategy: "e_over_h_threshold"
    column: "e/h"
    threshold: 0.1
    base_weight: 1.0
    high_weight: 1.5

model:
  params:
    objective: "reg:squarederror"
    max_depth: 5
    learning_rate: 0.0451669400461485
    n_estimators: 3970
    min_child_weight: 4
    subsample: 0.7238136617365515
    colsample_bytree: 0.4212006920305448
    reg_alpha: 0.012529841627285286
    reg_lambda: 0.44957639473634237
    gamma: 0.00013613870646182803
    random_state: 42
    tree_method: "hist"
    device: "cpu"
    n_jobs: -1
  use_optuna: true
  n_trials: 100
  optuna_timeout: 14400
  optuna_storage_url: "sqlite:///logs/optuna_study.db"
  best_params_path: "logs/best_params.json"
  early_stopping_rounds: 100
  eval_metric: "rmse"
  validation_size: 0.15
  selection_objective:
    metric_space: "original_nexp"
    rmse_normalizer: "mean_actual"
    cov_threshold: 0.10
    r2_threshold: 0.99
    cov_weight: 2.0
    r2_weight: 2.0

cv:
  n_splits: 5
  random_state: 42
  shuffle: true
```

说明：

- 当前默认主线通过 `config.model.params` 配置 XGBoost 参数（保持与既有配置兼容）。
- `target_mode: raw` 表示直接预测 `Nexp (kN)`；`target_mode: eta_u_over_npl` 和 `r_over_npl` 表示先在无量纲目标空间训练，最终仍回到 `Nexp` 空间汇报指标。
- `target_transform` 作用于训练目标，而不是直接作用于报告目标；当前默认主线使用 `log(eta_u)` 训练，但最终仍回到 `Nexp` 空间汇报。
- 当前默认主线是 `target_mode: eta_u_over_npl + target_transform.enabled: true + target_transform.type: log + model.keml.enabled: true`。
- 当前代码还支持 `boxcox_<lambda>` 形式的目标变换作为**实验支持**，用于协议探索和对照实验；它不是当前默认主线，也不代表最终 target 方案已经定稿。
- `data.sample_weight.enabled` 用于开启/关闭样本加权；关闭时保持原始无权重训练路径。
- 当前样本加权仅支持 `data.sample_weight.strategy: e_over_h_threshold`：当指定列（默认 `e/h`）大于 `threshold` 时使用 `high_weight`，其余样本使用 `base_weight`。
- 样本加权会同时作用于 `Optuna`、训练阶段 `CV`、fold 内验证集拆分以及最终 `train_full` 重训。
- `config.cv` 会同时控制 `Optuna` 调参和训练阶段输出的交叉验证报告；`cv.n_splits` 控制折数，`cv.shuffle` 与 `cv.random_state` 控制折分复现性。
- `selection_objective` 会在原始 `Nexp` 空间综合考虑 `RMSE / R² / COV`，而不是只按单一 RMSE 选模。
- 当 `use_optuna: true` 时，训练会先用 `CV` 复合目标调参，再在 `train_full` 上重训最终模型。
- 当 `use_optuna: false` 时，如果 `best_params_path` 指向的 `logs/best_params_*.json` 与当前 `context_hash` 匹配，则会自动加载最优参数。

## 实验配置与结果目录约定

- 默认主线配置仍放在 `config/config.yaml`。
- 一次性对照实验或长期迭代实验，统一放在 `config/experiments/` 下的对应子目录中。
- 当前 `compact_group_B` 的迭代优化配置统一放在：`config/experiments/compact_group_B_iterations/`
- 所有非默认主线的实验结果统一放在：`output/experiments/`
- 所有非默认主线的实验日志、Optuna 数据库与最优参数统一放在：`logs/experiments/`
- 其中 `compact_group_B` 的迭代优化结果和日志分别放在：`output/experiments/compact_group_B_iterations/` 与 `logs/experiments/compact_group_B_iterations/`
- 长期实验结论与每轮迭代记录统一写入：`doc/cfst_experiment_log.md`

这样可以把：
- 可复用主线结果
- 一次性实验结果
- 迭代调优实验结果

分开管理，避免 `config/`、`logs/`、`output/` 根目录继续堆积。

## 训练输出

如果直接运行默认主线：

```bash
python train.py --config config/config.yaml
```

则当前代码会在 `output/eta_u_over_npl_log_original_default_optuna100/` 下生成类似产物：

```text
output/eta_u_over_npl_log_original_default_optuna100/
├── xgboost_model.pkl
├── preprocessor.pkl
├── feature_names.json
├── training_metadata.json
├── evaluation_report.json
└── plots/
    ├── xgboost_model_train_predictions_scatter.png
    ├── xgboost_model_train_residuals.png
    ├── xgboost_model_train_error_distribution.png
    ├── xgboost_model_train_feature_importance.png
    ├── xgboost_model_train_feature_ranking.csv
    ├── xgboost_model_train_feature_ranking.txt
    ├── xgboost_model_test_predictions_scatter.png
    ├── xgboost_model_test_residuals.png
    ├── xgboost_model_test_error_distribution.png
    ├── xgboost_model_test_feature_importance.png
    ├── xgboost_model_test_feature_ranking.csv
    └── xgboost_model_test_feature_ranking.txt
```

如果显式传入 `--output output/model_run`，则会改写到你指定的目录。

非默认主线的对照实验、特征实验和迭代优化实验，统一写入：

- `output/experiments/`
- `logs/experiments/`

例如当前 `compact_group_B` 的迭代优化结果统一放在：

- `output/experiments/compact_group_B_iterations/`
- `logs/experiments/compact_group_B_iterations/`

当前仓库默认不会追踪新产生的 `output/` 结果目录；`output/experiments/` 是阶段性实验结果的聚合位置。默认主线的代表性输出如 `output/eta_u_over_npl_log_original_default_optuna100/` 仍可单独保留在根目录下。

## 评估指标解释

当前 `Evaluator` 实现了以下指标：

- `RMSE`
- `MAE`
- `R²`
- `MAPE`
- `MSE`
- `max_error`
- `COV`

其中 `COV` 基于预测值与真实值之比的均值和样本标准差计算，用于描述预测离散性。

## 预测使用手册

当前 `predict.py` 的真实行为是：

- 从模型目录加载模型工件（当前默认文件名为 `xgboost_model.pkl`），以及 `preprocessor.pkl`、`feature_names.json`、`training_metadata.json`
- 读取输入 CSV
- 假定输入是已经由 `scripts/compute_feature_parameters.py` 生成的 processed 特征表
- 当模型主线为 `eta_u_over_npl` 或 `r_over_npl` 时，输入需要包含 `Npl (kN)`，但不需要提供 `eta_u`、`r` 或 `Nexp (kN)`
- 在需要时将模型输出从 `eta_u` / `r` 空间恢复到 `Nexp`
- 进行单条或批量预测
- 可选导出 CSV

### 批量预测

```bash
python predict.py --model output/eta_u_over_npl_log_original_default_optuna100 --input data/processed/final_feature_parameters_raw.csv --output output/predictions.csv
```

参数说明：

- `--model`：模型目录路径
- `--input`：输入 CSV 路径
- `--output`：预测结果输出路径，可选
- `--single`：单条预测模式；仍然需要提供 CSV，脚本会只使用第一行
- `--verbose`：输出更详细日志

### 单条预测

当前实现不是交互式输入。建议复用现有 processed 数据文件并启用 `--single`：

```bash
python predict.py --model output/eta_u_over_npl_log_original_default_optuna100 --input data/processed/final_feature_parameters_raw.csv --single
```

在 `--single` 模式下：

- 输入文件仍然必须是 CSV
- 如果 CSV 超过 1 行，脚本会只使用第一行
- 返回结果为单个数值预测

### 预测输出

当前导出的 CSV 默认包含：

- 原始输入特征列
- 新增的 `prediction` 列

也就是说，输出通常类似：

```csv
fc (MPa),fy (MPa),Ac (mm^2),...,prediction
40.5,350.2,10000,...,2850.7
```

## Optuna 最优参数持久化

当前项目支持：

- 将 Optuna study 持久化到 SQLite
- 将最优参数保存到 `best_params_path` 指向的 JSON 文件
- 在上下文哈希匹配时自动复用最优参数

典型流程：

```bash
python train.py --config config/config.yaml
```

当 `use_optuna: true` 时，会：

1. 在训练数据上进行调参
2. 将最优参数保存到 `best_params_path` 指向的 JSON 文件
3. 在同一轮训练中使用最优参数重训最终模型

## 已知限制

以下内容是当前代码的已知限制，写论文或报告时需要明确区分：

### 1. `eta_u_over_npl` / `r_over_npl` 推理仍依赖足够的原始特征

当训练主线是 `eta_u = Nexp / Npl` 或 `r = (Nexp - Npl) / Npl` 时，推理脚本会自动把模型输出恢复回 `Nexp`，但前提是输入 CSV 已经是 processed 特征表，并包含 `Npl (kN)`。

当前推荐流程是：

1. 先对原始数据运行 `scripts/compute_feature_parameters.py`
2. 再将生成的 processed 特征表输入 `predict.py`

如果输入里缺少 `Npl (kN)`，推理脚本无法完成 `eta_u/r -> Nexp` 的恢复。

### 2. 非默认目标变换路径未作为当前主线验证

当前默认主线仍采用 `log` 目标变换，但代码中已加入 Box-Cox 目标变换的**实验支持**。

因此：

- 默认实验当前使用 `target_mode: eta_u_over_npl`
- 默认训练变换当前使用 `target_transform.enabled: true` 与 `target_transform.type: log`
- 当前代码还支持 `target_transform.type: boxcox_<lambda>` 形式的实验性配置（例如 `boxcox_0.25`、`boxcox_0.50`）
- 这些 Box-Cox 路径当前仅用于实验扫描与协议探索，不代表仓库主线已经切换
- 如需将 Box-Cox 或其他目标变换纳入正式主线，仍应补充完整验证并统一研究结论

### 3. 当前 CV 结果更适合作为调参参考，不宜直接作为无偏论文结论

当前实现中：

- `Optuna` 在训练集上使用交叉验证做调参
- 然后又在同一训练集上计算交叉验证结果并输出
- 这两条路径现在都会遵守 `config.cv` 中配置的 `n_splits` / `shuffle` / `random_state`

这会导致 CV 结果偏乐观。对论文写作而言，更建议：

- 以独立测试集指标作为主要泛化结果
- 如果需要更严格的泛化估计，后续采用 `nested CV`

### 4. 当前版本仅加入了基础的目标定义域校验，尚未加入工程边界校验

当前代码已经对目标变换做了基础定义域检查：

- `log` 变换要求目标值全部大于 0
- `boxcox_<lambda>` 变换也要求输入目标值全部大于 0

但当前代码仍未额外检查：

- 数据是否存在超出经验适用范围的边界工况
- 样本是否超出你后续论文打算采用的工程筛选范围

这部分更适合结合你的人工数据筛选规则、工程经验和后续经验公式一起处理。

## 论文使用建议

如果你打算将本项目用于论文实验，当前更稳妥的建议是：

- 优先把数据集筛选规则写清楚，尤其是是否排除了非经典工况
- 优先报告独立测试集上的 `RMSE`、`MAE`、`R²`、`COV`
- 把交叉验证结果表述为“训练阶段模型选择参考”，避免当作完全无偏的最终泛化结论
- 如果关闭目标变换并重新训练，请确保论文中的推理结果、评估指标和工程解释全部在同一物理量空间下进行

## 测试与验证

文档变更的最小验证（smoke）建议：

1. 验证 README 中引用的关键路径与文件存在：

```bash
ls config/config.yaml data/processed/final_feature_parameters_raw.csv train.py predict.py
```

2. 运行测试套件：

```bash
pytest -q
```

3. 可选语法检查：

```bash
python -m compileall train.py predict.py src tests
```

## 后续建议（本轮未实现）

以下改动很值得作为下一轮工作：

- 将训练/验证/测试三套指标完全拆开报告
- 将当前 CV/Optuna 方案升级为更严格的 `nested CV`
- 按你的数据筛选规则增加目标定义域与边界工况校验
- 如果未来要恢复特征选择功能，建议在代码落地后再补回对应文档
