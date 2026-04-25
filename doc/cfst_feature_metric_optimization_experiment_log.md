# CFST 下一阶段特征与指标优化计划

## 目标

当前目标不是继续扩大特征数，而是在“论文可解释性”和“真实泛化指标”之间建立一条更稳的主线：

- 主指标：原始 `Nexp (kN)` 空间的 `R² / COV / a20-index / RMSE`
- 目标线：优先逼近 `R² >= 0.99` 与 `COV <= 0.10`
- 默认建模协议：`eta_u_over_npl + log + KeML`
- 默认评估协议：固定 holdout test + 5-fold CV + regime analysis

## 当前判断

上一轮 12 特征模型：

- Test `R² = 0.9796`
- Test `COV = 0.1400`
- Test `RMSE = 513.30 kN`

相对历史 `B4 + log + KeML` 主线退化。新增的曲率三特征还没有证明收益，且 `reverse_curvature_flag` 只覆盖约 1% 样本，存在稀有 regime 被树模型过度切分的风险。

因此下一阶段不把 12 特征作为主线，而是回到紧凑机理特征与受控消融。

## 代码改造

### 1. 显式特征白名单

新增 `config.data.feature_columns` 支持。实验配置可以显式声明模型只使用哪些特征，训练脚本自动把其他列加入 drop list。

目的：

- 避免新增诊断列污染旧实验配置
- 保证 9/B4/B6/模型对照的特征集完全可复现
- 让 `feature_names.json` 与配置语义一致

### 2. 配置组织

新建目录：

```text
config/experiments/next_stage_feature_metric/
```

单 seed Phase 1 配置命名：

```text
phase1_G1_core9_xgboost.yaml
phase1_G2_b4_xgboost.yaml
phase1_G3_core9_lightgbm.yaml
phase1_G4_core9_catboost.yaml
```

对应输出：

```text
output/experiments/next_stage_feature_metric/<experiment_name>
logs/experiments/next_stage_feature_metric/<experiment_name>_best_params.json
logs/experiments/next_stage_feature_metric/<experiment_name>_optuna.db
```

## Phase 1：单 seed 100-trial 筛选

固定：

- `random_state = 42`
- `test_size = 0.2`
- `split.strategy = regression_stratified`
- `cv.n_splits = 5`
- `n_trials = 100`
- `target_mode = eta_u_over_npl`
- `target_transform = log`
- `KeML = enabled`

实验矩阵：

| ID | 特征组 | Backbone | 用途 |
|---|---|---|---|
| G1 | core9 | XGBoost | 机理紧凑主线 |
| G2 | B4 | XGBoost | 历史强基线复现 |
| G3 | core9 | LightGBM | COV 竞争模型 |
| G4 | core9 | CatBoost | a20-index 竞争模型 |

### core9 特征

```text
fy (MPa)
fc (MPa)
Re (mm)
te (mm)
ke
lambda_bar
e_bar
b/h
e/h
```

### B4 特征

```text
fy (MPa)
fc (MPa)
Re (mm)
te (mm)
ke
lambda_bar
e/h
e_bar
e1/e2
b/h
```

B4 保留旧 `e1/e2` 是为了复现实验日志中的历史强基线。它不作为最终论文主线的默认结论，后续需要用更稳的单一 bounded moment-gradient 替代再做消融。

### Phase 1 决策规则

- 若 G1 与 G2 接近，优先选择 G1 作为论文主线。
- 若 G2 明显优于 G1，保留端弯矩梯度信息，但进入 Phase 2 时改为更稳定的单一 bounded 表达。
- 若 LightGBM 明显降低 COV，进入 Phase 2 主池。
- 若 CatBoost 明显提高 a20-index，进入 Phase 2 主池。

## Phase 2：多 seed 稳健性验证

只对 Phase 1 前两名或指标互补的模型继续跑：

```text
random_state = 0, 7, 21, 42, 99
```

每个 seed 仍使用 100 trials。报告均值和标准差，而不是只报告单次最好结果。

## Phase 3：误差结构优化

基于 Phase 1/2 的最佳特征组，针对当前最差 regime 做三类优化：

1. regime sample weighting
   - `scale_npl_q5`
   - `low_conf`
   - `very_high_conf`
   - axial high-capacity samples
2. regime residual calibration
   - global model + axial/eccentric correction
   - global model + section_family correction
3. section/source aware split
   - random stratified split：论文可比口径
   - `Ref.No.` group split：跨来源泛化口径

## Phase 4：集成模型

当至少两个 backbone 在 Phase 2 中表现互补时，做简单平均或 stacking：

```text
ensemble = mean(XGBoost, LightGBM, CatBoost)
```

优先观察：

- `COV`
- `a20-index`
- `scale_npl_q5` 的 regime COV

## 暂不优先做的事

- 不继续扩大到 12+ 特征作为默认主线
- 不继续细扫 Box-Cox lambda
- 不把 `log + smearing` 作为主突破方向
- 不在数据审计前追求过强模型复杂度

## 当前执行顺序

1. 加入 `feature_columns` 白名单支持。
2. 生成 Phase 1 四个配置。
3. 跑 G1/G2/G3/G4 单 seed 100-trial。
4. 汇总 `evaluation_report.json` 与 regime metrics。
5. 决定 Phase 2 多 seed 主池。

## Phase 1 执行记录

执行时间：2026-04-25

| ID | Backbone | 特征数 | Test RMSE | Test R² | Test COV | Test a20 | CV J | CV COV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 core9 | XGBoost | 9 | 552.8705 | 0.9763 | 0.1427 | 0.9364 | 6.6228 | 0.1564 |
| G2 B4 | XGBoost | 10 | 454.8201 | 0.9840 | 0.1358 | 0.9434 | 6.8349 | 0.1508 |
| G3 core9 | LightGBM | 9 | 380.9289 | 0.9888 | 0.1375 | 0.9294 | 7.4198 | 0.1671 |
| G4 core9 | CatBoost | 9 | 404.7662 | 0.9873 | 0.1431 | 0.9387 | 7.7542 | 0.1781 |

### Phase 1 判断

- `core9 + LightGBM` 当前取得最低 holdout RMSE 和最高 holdout R²，但 holdout COV 仍为 `0.1375`，距离 `0.10` 还有明显差距。
- `B4 + XGBoost` 仍是更稳的树模型基线，Test COV 和 a20-index 在四组中最好，但 RMSE/R² 不如 `core9 + LightGBM`。
- 只保留 9 个核心机理特征并不自动优于历史 B4；端弯矩梯度信息仍然有价值。
- CatBoost 未在本轮超过 LightGBM 或 B4 XGBoost，暂不作为 Phase 2 主模型。

### Phase 2 候选

下一步不建议直接把 G3 设为默认主线，而是先做三组补充：

1. `B4 + LightGBM`：验证 LightGBM 的 RMSE 优势是否也能作用于含端弯矩信息的 B4 特征组。
2. `bounded_moment_gradient + XGBoost/LightGBM`：用单一有界端弯矩梯度表达替代旧 `e1/e2`，比较是否兼顾可解释性和稳定性。
3. 对 `G2 B4 XGBoost`、`G3 core9 LightGBM`、补充的 `B4 LightGBM` 做 5-seed 稳健性验证。

## Phase 2 补充单 seed 执行记录

执行时间：2026-04-25

| ID | Backbone | 特征数 | Test RMSE | Test R² | Test COV | Test a20 | CV J | CV COV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G5 B4 | LightGBM | 10 | 456.5097 | 0.9839 | 0.1308 | 0.9370 | 8.0600 | 0.1635 |
| G6 bounded MG | XGBoost | 10 | 577.5668 | 0.9742 | 0.1377 | 0.9393 | 6.6317 | 0.1600 |
| G7 bounded MG | LightGBM | 10 | 565.0304 | 0.9753 | 0.1390 | 0.9411 | 7.2851 | 0.1702 |

### Phase 2 补充判断

- `B4 + LightGBM` 没有继承 `core9 + LightGBM` 的低 RMSE 优势，但把 holdout COV 降到了 `0.1308`，是当前所有单 seed 单模型里最低的 COV。
- `bounded_moment_gradient` 两组在 CV 上表现不错，尤其 XGBoost 的 CV J 达到 `6.6317`，但 holdout RMSE/R² 明显退化，说明该特征替换在当前 split 下存在 CV/holdout 不一致，不能直接作为默认主线。
- 当前单 seed 排名应拆成两个目标看：
  - 低 RMSE/R² 主线：`G3 core9 LightGBM`
  - 低 COV 主线：`G5 B4 LightGBM`
- 下一步优先做同一 holdout 上的 `G3 + G5` 简单平均 ensemble；如果 ensemble 同时降低 RMSE 和 COV，再进入 5-seed 验证。

## Phase 2.5 同一 holdout 集成评估记录

执行时间：2026-04-25

脚本：

```text
scripts/evaluate_next_stage_ensembles.py
```

输出：

```text
output/experiments/next_stage_feature_metric/ensembles/evaluation_report.json
output/experiments/next_stage_feature_metric/ensembles/predictions.csv
```

固定复现 `phase1_G3_core9_lightgbm` 的 train/test 切分：

- Train: 6852
- Test: 1714
- Split: regression_stratified
- Strata: 30 train / 30 test

### 集成结果

| 模型/集成 | Test RMSE | Test R² | Test COV | Test a20 | 备注 |
|---|---:|---:|---:|---:|---|
| G3 core9 LightGBM | 380.9289 | 0.9888 | 0.1375 | 0.9294 | 当前最低 RMSE / 最高 R² |
| G5 B4 LightGBM | 456.5097 | 0.9839 | 0.1308 | 0.9370 | 当前最低单模型 COV |
| E1 mean(G3, G5) | 407.0741 | 0.9872 | 0.1324 | 0.9329 | COV 改善但 RMSE/R² 退化 |
| E2 mean(G2, G3, G5) | 397.6472 | 0.9877 | 0.1308 | 0.9382 | 当前最均衡集成 |
| E3 mean(G1,G2,G3,G4,G5) | 409.1936 | 0.9870 | 0.1331 | 0.9399 | 加入弱模型后 RMSE 继续退化 |
| E5 mean(G1-G7) | 418.8248 | 0.9864 | 0.1332 | 0.9411 | bounded MG 模型没有带来整体收益 |

### 集成判断

- 简单平均没有超过 `G3 core9 LightGBM` 的 RMSE/R²；最低 RMSE 仍是 `380.9289`。
- `mean(G2, G3, G5)` 是当前最均衡的后处理结果：COV 基本等于 G5 的最低值 `0.1308`，RMSE 从 G5 的 `456.5097` 降到 `397.6472`，但仍高于 G3。
- `G3` 与 `G5` 的 holdout 残差相关性为 `0.8896`，互补性不够强，所以二者平均不能带来明显误差抵消。
- holdout 权重扫描显示：
  - 最低 RMSE 权重仍是 100% G3。
  - 最低 COV 权重约为 5% G3 + 95% G5，COV `0.130827`，但 RMSE `450.7082`。
  - 该扫描使用 holdout 选权重，只能作为诊断，不能作为论文可报告的无偏测试结果。

### 下一步调整

不建议直接把 ensemble 作为默认主线。下一阶段应转向误差结构优化：

1. 对 `G3 core9 LightGBM` 做 worst-regime 诊断，重点是 `scale_npl_q5`、`very_stocky`、`axial`、`mid_conf`。
2. 在不扩大特征集的前提下做 regime sample weighting / residual calibration。
3. 若校准方案能同时改善 G3 的 COV 和最差 regime RMSE，再做 5-seed 验证。

## Phase 3 误差结构优化执行记录

执行时间：2026-04-25

### 代码改造

新增：

```text
scripts/evaluate_next_stage_regime_calibration.py
scripts/run_next_stage_phase3.py
```

扩展：

```text
train.py::build_sample_weights
```

新增 `sample_weight.strategy = multiplicative_rules`，允许用训练数据中的诊断列定义乘法权重规则。模型输入特征仍可保持 `core9`，权重规则可以引用不进入模型的列，例如 `Npl (kN)`、`xi`、`axial_flag`。

### Phase 3A：Residual Calibration 诊断

输出：

```text
output/experiments/next_stage_feature_metric/phase3_regime_calibration/evaluation_report.json
output/experiments/next_stage_feature_metric/phase3_regime_calibration/predictions.csv
output/experiments/next_stage_feature_metric/phase3_regime_calibration/calibration_factors.csv
```

校准对象：`G3 core9 LightGBM`

校准只在训练 split 上拟合，再应用到 holdout。当前实现仍是诊断版，因为校准因子来自最终模型的训练集内预测；若要作为论文最终指标，需要改成 OOF calibration。

| 校准方案 | Test RMSE | Test R² | Test COV | Test a20 | ΔRMSE | ΔCOV |
|---|---:|---:|---:|---:|---:|---:|
| baseline G3 | 380.9289 | 0.9888 | 0.137519 | 0.9294 | - | - |
| axiality × scale_npl ratio | 380.0828 | 0.9888 | 0.137453 | 0.9294 | -0.8461 | -0.000066 |
| slenderness × scale_npl ratio | 380.1126 | 0.9888 | 0.137436 | 0.9294 | -0.8163 | -0.000082 |
| confinement_level ratio | 380.1390 | 0.9888 | 0.137470 | 0.9294 | -0.7899 | -0.000048 |

判断：

- 简单 regime 均值/比例校准只有约 `0.2%` 的 RMSE 改善，COV 几乎不变。
- 这说明 G3 的主要问题不是“分组均值偏差”，而更可能是尾部样本方差、跨来源分布差异或极值样本影响。
- 不建议把 residual calibration 作为当前主突破口。

### Phase 3B：Regime Sample Weighting 100-trial 重训

固定：

- `core9` 特征
- LightGBM
- `eta_u_over_npl + log + KeML`
- `random_state = 42`
- `n_trials = 100`

实验：

| ID | 权重策略 | Test RMSE | Test R² | Test COV | Test a20 | CV J | CV COV |
|---|---|---:|---:|---:|---:|---:|---:|
| G3 baseline | 无加权 | 380.9289 | 0.9888 | 0.1375 | 0.9294 | 7.4198 | 0.1671 |
| W1 | `Npl` top 20% × 1.6 | 468.0754 | 0.9830 | 0.1416 | 0.9282 | 8.0459 | 0.2024 |
| W2 | `Npl` top 20% × 1.4 + `lambda_bar <= 0.25` × 1.15 + axial × 1.1 + `0.5 <= xi <= 1.5` × 1.15 | 397.8484 | 0.9877 | 0.1411 | 0.9329 | 8.1722 | 0.1857 |

输出目录：

```text
output/experiments/next_stage_feature_metric/phase3_W1_core9_lightgbm_high_npl_weight/
output/experiments/next_stage_feature_metric/phase3_W2_core9_lightgbm_regime_weight/
```

Worst-regime RMSE 对比：

| 模型 | axial | section | very_stocky | scale_npl_q5 | confinement |
|---|---:|---:|---:|---:|---:|
| G3 baseline | 419.5105 | circular 417.8466 | 441.4376 | 796.9527 | mid_conf 398.3955 |
| W1 | 522.6758 | square 573.9064 | 553.2293 | 998.5407 | low_conf 656.3622 |
| W2 | 432.1860 | circular 423.3161 | 464.7407 | 837.6239 | low_conf 447.3024 |

判断：

- 两个加权重训都没有超过 G3；W1 明显退化，W2 虽然 a20 略升，但 RMSE/R²/COV 和最差 regime 都不如 G3。
- 加权后训练集 RMSE 下降到约 `97-109 kN`，但 holdout RMSE 仍为 `398-468 kN`，训练/测试 RMSE 比超过 4，过拟合更强。
- 当前的“尾部/短柱/轴压/约束加权”不是有效方向，至少不能用作默认主线。

### Phase 3 结论

Phase 3 目前排除了两个直觉方向：

1. 简单 residual calibration 不能显著降低 COV。
2. 直接 regime sample weighting 会加重过拟合，不能改善 holdout。

下一步不建议继续加权。更合理的后续方向是：

- 做 `Ref.No.` / 文献来源级 group split，判断当前高指标是否受同源样本泄漏或来源重复影响。
- 审计 `scale_npl_q5` 中的极端样本和重复/近重复记录，确认是否有单位、构件类别、公式计算或来源标注问题。
- 若要继续提升指标，优先改为 OOF stacking / OOF calibration，而不是在最终训练集内做校准。

## Phase 3C 来源审计与 Source Group Holdout

执行时间：2026-04-25

用户确认原始数据：

```text
data/raw/final.csv
```

新增脚本：

```text
scripts/audit_source_and_extreme_regimes.py
scripts/run_next_stage_source_group_holdout.py
```

### 3C-1：当前随机 split 的来源重叠审计

输出：

```text
output/experiments/next_stage_feature_metric/source_audit/audit_report.json
output/experiments/next_stage_feature_metric/source_audit/source_split_summary.csv
output/experiments/next_stage_feature_metric/source_audit/test_source_metrics.csv
output/experiments/next_stage_feature_metric/source_audit/scale_npl_q5_test_samples.csv
output/experiments/next_stage_feature_metric/source_audit/scale_npl_q5_source_metrics.csv
output/experiments/next_stage_feature_metric/source_audit/worst_test_errors.csv
```

审计对象：`phase1_G3_core9_lightgbm`

| 项目 | 数值 |
|---|---:|
| Raw rows | 8566 |
| Processed rows | 8566 |
| Source count (`Ref.No.`) | 619 |
| Train sources | 614 |
| Test sources | 487 |
| Train/Test 同时出现的 sources | 482 |
| Test sources 已在 train 中出现的比例 | 98.97% |

随机 split 下 G3 holdout 指标：

| Split | RMSE | R² | COV | a20 |
|---|---:|---:|---:|---:|
| random stratified test | 380.9289 | 0.9888 | 0.1375 | 0.9294 |
| random stratified `scale_npl_q5` | 796.9527 | 0.9857 | 0.1842 | 0.8955 |

判断：

- 当前常规 holdout 不是严格跨来源外推。测试集中绝大多数文献来源也出现在训练集里，模型可能学习到同一论文/同一实验系列的分布特征。
- 这不一定是“数据泄漏 bug”，因为同一来源中不同试件确实是不同样本；但如果论文目标是报告面向未知文献/未知实验系列的泛化能力，必须增加 `Ref.No.` 级 group split 口径。
- `scale_npl_q5` 仍是最难的尺度 regime，随机 split 下 COV 已达到 `0.1842`，说明高承载力/大尺度样本误差不是单纯由来源重叠解释。

### 3C-2：G3 Source Group Holdout 100-trial 重训

命令：

```text
python scripts/run_next_stage_source_group_holdout.py --raw data/raw/final.csv --config config/experiments/next_stage_feature_metric/phase1_G3_core9_lightgbm.yaml --experiment-name phase3_G3_core9_lightgbm_source_group_holdout --n-trials 100
```

输出：

```text
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout/predictions.csv
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout/source_split_summary.csv
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout/source_test_metrics.csv
logs/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout_best_params.json
logs/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout_optuna.db
```

Split：

| 项目 | 数值 |
|---|---:|
| Train samples | 6867 |
| Test samples | 1699 |
| Train sources | 495 |
| Test sources | 124 |
| Source overlap | 0 |
| Actual test row share | 19.83% |

Best Optuna params：

```text
n_estimators = 565
learning_rate = 0.1022162485
num_leaves = 81
subsample = 0.7926750805
colsample_bytree = 0.6292058641
```

指标：

| Protocol | RMSE | R² | COV | a20 | 备注 |
|---|---:|---:|---:|---:|---|
| random stratified G3 test | 380.9289 | 0.9888 | 0.1375 | 0.9294 | 来源高度重叠 |
| source group CV | 761.1373 | 0.9317 | 0.3747 | 0.7759 | 训练来源内 group-CV |
| source group train apparent | 135.5508 | 0.9979 | 0.0644 | 0.9847 | 表观训练指标 |
| source group holdout test | 595.2878 | 0.9576 | 0.1874 | 0.8034 | 完全留出 124 个来源 |

主要 worst regimes：

| Regime | Worst bucket | RMSE | R² | COV | a20 |
|---|---|---:|---:|---:|---:|
| axiality | axial | 672.9993 | 0.9568 | 0.1954 | 0.7955 |
| section_family | square | 651.0742 | 0.9032 | 0.2298 | 0.7856 |
| slenderness_state | very_stocky | 647.2448 | 0.9595 | 0.1801 | 0.8154 |
| confinement_level | mid_conf | 601.7618 | 0.9506 | 0.1954 | 0.8279 |
| confinement_level | very_high_conf | 786.1985 | 0.9815 | 0.1892 | 0.7206 |

判断：

- `Ref.No.` 完全留出后，G3 的 test R² 从 `0.9888` 降到 `0.9576`，COV 从 `0.1375` 升到 `0.1874`，a20 从 `0.9294` 降到 `0.8034`。这说明当前离论文中常见 `R²=0.99+ / COV<0.10` 的差距，核心不只是模型调参，也包含评估口径与来源泛化难度。
- 训练表观指标非常好，但 source group CV 和 source group holdout 都明显变差，说明模型对同源/同系列样本拟合强，对未知来源外推弱。
- 后续论文报告不应只给随机 holdout。建议同时报告两个口径：
  - `random stratified split`：用于和多数 ML/DL 论文指标对齐。
  - `Ref.No. group split`：用于证明跨文献来源泛化能力。
- 当前 source group split 是单 seed，不能单独作为最终结论。下一步应做多 seed source group holdout，至少比较 `G3 core9 LightGBM` 与 `G5 B4 LightGBM`，再决定是否需要 source-aware stacking / OOF calibration。

### 3C-3：G5 B4 Source Group Holdout 100-trial 重训

命令：

```text
python scripts/run_next_stage_source_group_holdout.py --raw data/raw/final.csv --config config/experiments/next_stage_feature_metric/phase2_G5_b4_lightgbm.yaml --experiment-name phase3_G5_b4_lightgbm_source_group_holdout --n-trials 100
```

输出：

```text
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout/predictions.csv
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout/source_split_summary.csv
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout/source_test_metrics.csv
logs/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout_best_params.json
logs/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout_optuna.db
```

Best Optuna params：

```text
n_estimators = 702
learning_rate = 0.0307095459
num_leaves = 51
subsample = 0.8202492182
colsample_bytree = 0.9123977928
```

同一 `Ref.No.` source group split 下的 G3/G5 对比：

| Model | 特征数 | Best CV J | CV RMSE | CV R² | CV COV | Test RMSE | Test R² | Test COV | Test a20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G3 core9 LightGBM | 9 | 17.5350 | 761.1373 | 0.9317 | 0.3747 | 595.2878 | 0.9576 | 0.1874 | 0.8034 |
| G5 B4 LightGBM | 10 | 17.2731 | 744.8846 | 0.9330 | 0.3747 | 578.6748 | 0.9599 | 0.1768 | 0.8252 |

判断：

- `B4 + LightGBM` 在 source group split 下全面略优于 `core9 + LightGBM`：RMSE 降低约 `16.61 kN`，R² 提高约 `0.0023`，COV 降低约 `0.0106`，a20 提高约 `0.0218`。
- 端弯矩梯度信息 `e1/e2` 在跨来源外推中仍然有价值。此前 bounded moment-gradient 替代方案没有复现这个收益，因此当前不能简单把 B4 压缩回 core9。
- 但两者 source group CV 的 COV 都约 `0.375`，远高于随机 split 下的 holdout COV，说明跨来源泛化仍是主瓶颈。
- 当前更合理的候选主线应暂时从 `G3 core9 LightGBM` 调整为 `G5 B4 LightGBM`，但需要多 seed source group holdout 复核，而不是只凭 seed=42 决策。

### 3C-4：Source Group Holdout 多 seed 快速复核（已停止作为主线）

在用户追问“文献来源划分是否把原先分层 split 改掉”之后，确认：

- 默认主线配置仍为 `regression_stratified`。
- source group holdout 只由独立脚本 `scripts/run_next_stage_source_group_holdout.py` 执行。
- source group 路线用于诊断跨文献来源外推，不再作为当前冲击论文常见 `R² 0.99+ / COV < 0.10` 指标的主线。

已完成的固定参数快速复核如下；这些 run 使用已调好的参数，不再重新 Optuna：

| Model | Seed | Protocol | Test RMSE | Test R² | Test COV | Test a20 |
|---|---:|---|---:|---:|---:|---:|
| G3 core9 LightGBM | 0 | source group holdout, fixed params | 889.2389 | 0.9145 | 0.1997 | 0.7841 |
| G5 B4 LightGBM | 0 | source group holdout, fixed params | 790.4679 | 0.9325 | 0.1966 | 0.8025 |
| G3 core9 LightGBM | 7 | source group holdout, fixed params | 941.0228 | 0.9329 | 0.2004 | 0.7565 |
| G5 B4 LightGBM | 7 | source group holdout, fixed params | 965.4834 | 0.9294 | 0.1836 | 0.7829 |

输出目录：

```text
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout_seed0_fixed/
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout_seed0_fixed/
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout_seed7_fixed/
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout_seed7_fixed/
```

阶段判断：

- source group holdout 是更严格、更接近“未知文献来源外推”的口径，但它和大量论文常用的 random/stratified holdout 指标不可直接混为一个目标。
- 该口径下 G5 B4 通常不差于 G3 core9，但绝对指标明显低于随机分层 holdout。继续沿 source group 路线优化，会把任务变成“跨文献泛化”而不是“对齐主流论文评估口径”。
- 当前主线恢复为 `regression_stratified`，以 `G5 B4 LightGBM` 作为 baseline；source group 结果只在论文中作为补充泛化诊断。

## Phase 4 标准 CFST 标记、标准子集与 B4 消融

执行时间：2026-04-25

目标：

1. 保持主线为 `regression_stratified` 分层 split。
2. 以 `G5 B4 LightGBM` 作为当前 baseline。
3. 不删除样本，先为全量 processed 数据生成 `standard_cfst` 标签：
   - `standard`
   - `nonstandard_material`
   - `stiffened_or_reinforced`
   - `recycled_or_fiber`
   - `defect_or_gap`
   - `hollow_or_special_section`
   - `unclear`
4. 在 `standard_cfst == standard` 子集上跑 `B4/G5`，观察指标是否自然接近 `R² 0.99 / COV 0.10`。
5. 在 full data 上跑 `B4_plus_xi` / `B4_plus_scale` / `B4_plus_local` 消融。
6. 若仍主要差在 COV，再做 OOF correction 或分区模型，专门压 COV。

新增脚本计划：

```text
scripts/tag_standard_cfst_dataset.py
scripts/create_phase4_standard_cfst_configs.py
scripts/run_next_stage_phase4.py
```

新增数据文件计划：

```text
data/processed/final_feature_parameters_tagged.csv
data/processed/final_feature_parameters_standard_cfst.csv
output/experiments/next_stage_feature_metric/dataset_scope_audit/standard_cfst_summary.json
```

消融定义：

| Experiment | 数据范围 | 特征定义 | 目的 |
|---|---|---|---|
| `phase4_standard_G5_b4_lightgbm` | `standard_cfst == standard` | B4 | 看标准 CFST 子集能否自然逼近文献常见高指标 |
| `phase4_full_B4_plus_xi_lightgbm` | full data | B4 + `xi` | 检查显式约束比是否补足机理信息 |
| `phase4_full_B4_plus_scale_lightgbm` | full data | B4 + `log_Npl` | 检查尺度效应是否影响 `eta_u_over_npl` |
| `phase4_full_B4_plus_local_lightgbm` | full data | B4 + `b/t` | 检查局部宽厚比是否解释残差/COV |

注意：

- 标签是审计/筛选字段，不作为模型输入。
- full-data 消融继续使用 `regression_stratified`，不使用 source group split。
- `B4_plus_scale` 中的尺度代理先取 `log_Npl`，避免 `Npl` 原值过大且与目标逆变换尺度强耦合。

### Phase 4 路线修正：不再剔除“非标准 CFST”

用户澄清：原始数据库纳入阶段已经处理过非标准 CFST 工况，当前数据主要是碳钢 + 普通/高强/超高强/再生/UHPC 等材料类别的 CFST 柱。因此，上面的 `standard_cfst` 标签只能作为一次过宽关键词启发式的废弃草稿，后续不再按该标签删除或筛选样本。

废弃草稿的实际产物记录如下，保留在这里是为了避免后续误判为已完成主线实验：

```text
data/processed/final_feature_parameters_tagged.csv
data/processed/final_feature_parameters_standard_cfst.csv
output/experiments/next_stage_feature_metric/dataset_scope_audit/standard_cfst_summary.json
output/experiments/next_stage_feature_metric/phase4_standard_cfst_ablation_summary.json
logs/experiments/next_stage_feature_metric/phase4_standard_G5_b4_lightgbm_optuna.db
```

标签草稿统计：

| Label | Rows |
|---|---:|
| `standard` | 6307 |
| `recycled_or_fiber` | 1494 |
| `nonstandard_material` | 387 |
| `stiffened_or_reinforced` | 146 |
| `hollow_or_special_section` | 94 |
| `defect_or_gap` | 72 |
| `unclear` | 66 |

草稿训练状态：

| Experiment/config | 状态 | 说明 |
|---|---|---|
| `phase4_standard_G5_b4_lightgbm` | interrupted / no final report | Optuna DB 中有 83 个 trials，其中 82 个 complete、1 个 running；best observed CV J 为 `8.1017`，但没有完成 CV、final train、holdout evaluation，不可作为结果引用 |
| `phase4_full_B4_plus_xi_lightgbm` | not run | 仅生成 config，未开始训练 |
| `phase4_full_B4_plus_scale_lightgbm` | not run | 仅生成 config，未开始训练 |
| `phase4_full_B4_plus_local_lightgbm` | not run | 仅生成 config，未开始训练 |

处理结论：

- 不删除这些草稿产物，便于复核历史操作。
- 后续不得按 `standard_cfst` 标签筛除样本，也不得引用 `phase4_standard_G5_b4_lightgbm` 的 partial Optuna 数值作为实验结果。
- Phase 4 正式路线改为高尺度/高承载力样本诊断。

当前 Phase 4 改为诊断“高尺度/高承载力样本是否拖低随机分层 holdout 指标”：

- 筛选字段：`Npl (kN)`
- 筛选口径：保留 `Npl (kN) <= q80`，剔除最高 20% 的承载力尺度样本。
- 原因：`Npl` 由几何和材料参数计算得到，不直接使用实验标签 `Nexp`；同时它正对应此前 worst regime `scale_npl_q5`。
- 目的：判断指标不足是否主要由大尺寸/大承载力区间主导，而不是由“非标准试件”主导。

新增脚本：

```text
scripts/create_scale_trimmed_dataset.py
scripts/create_phase4_scale_trim_configs.py
scripts/run_next_stage_phase4_scale_trim.py
```

新增文件：

```text
data/processed/final_feature_parameters_npl_le_q80.csv
data/processed/final_feature_parameters_npl_gt_q80.csv
output/experiments/next_stage_feature_metric/scale_trim_audit/npl_q80_summary.json
config/experiments/next_stage_feature_metric/phase4_npl_le_q80_G5_b4_lightgbm.yaml
```

待运行实验：

| Experiment | 数据范围 | Split | 模型 | 特征 |
|---|---|---|---|---|
| `phase4_npl_le_q80_G5_b4_lightgbm` | 去除 `Npl` 最高 20% 后的 80% 样本 | `regression_stratified` | LightGBM + KeML + 100-trial Optuna | B4 |

### Phase 4B：剔除 `Npl` 最高 20% 后的 G5 B4 训练结果

执行命令：

```text
python scripts/create_scale_trimmed_dataset.py
python scripts/create_phase4_scale_trim_configs.py
python scripts/run_next_stage_phase4_scale_trim.py
```

筛选摘要：

| 项目 | 数值 |
|---|---:|
| `Npl` q80 threshold | 3147.5408 kN |
| Full rows | 8566 |
| Kept rows (`Npl <= q80`) | 6853 |
| Excluded rows (`Npl > q80`) | 1713 |
| Kept share | 80.00% |
| Excluded share | 20.00% |
| Kept median `Nexp` | 1150.0 kN |
| Excluded median `Nexp` | 3659.0 kN |
| Kept median `b` / `h` | 138.87 / 126.0 mm |
| Excluded median `b` / `h` | 219.0 / 219.0 mm |

训练设置：

- 数据：`data/processed/final_feature_parameters_npl_le_q80.csv`
- Split：`regression_stratified`
- 模型：`G5 B4 LightGBM`
- 目标：`eta_u_over_npl + log`
- KeML：enabled
- Optuna：100 trials

指标对比：

| Experiment | Rows | Test RMSE | Test R² | Test COV | Test MAPE | Test a20 | CV J | CV RMSE | CV COV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-data G5 B4 baseline | 8566 | 456.5097 | 0.9839 | 0.1308 | 6.8170% | 0.9370 | 8.0600 | 561.0609 | 0.1635 |
| `Npl <= q80` G5 B4 | 6853 | 189.7342 | 0.9383 | 0.0969 | 5.6725% | 0.9613 | 6.7116 | 151.3911 | 0.1157 |

Best Optuna params：

```text
n_estimators = 561
learning_rate = 0.0802803291
num_leaves = 109
subsample = 0.6922607996
colsample_bytree = 0.8788455081
```

主要观察：

- 剔除最高 20% `Npl` 后，test COV 从 `0.1308` 降到 `0.0969`，已经低于 `0.10`。
- test MAPE 从 `6.8170%` 降到 `5.6725%`，a20 从 `0.9370` 升到 `0.9613`，说明相对误差口径明显改善。
- test RMSE 从 `456.51 kN` 降到 `189.73 kN`，但这部分不能单独作为模型能力提升，因为测试集承载力尺度被主动压低了。
- test R² 从 `0.9839` 降到 `0.9383`，不应简单解释为模型变差；剔除高尺度样本后 `Nexp` 方差显著缩小，R² 的分母变小，对同等相对误差更敏感。
- CV 指标同步改善：`J 8.0600 -> 6.7116`，`CV COV 0.1635 -> 0.1157`。这说明改善不是单次 holdout 偶然结果。

阶段判断：

- 当前 COV 不达标的主因之一确实是高承载力/大尺寸区间，而不是“非标准 CFST 工况”。
- 若论文目标是报告一个接近 `COV < 0.10` 的主流随机分层指标，直接剔除最高 20% `Npl` 可以做到，但这会缩小适用域，必须在论文中明确建模范围。
- 若要保留全尺寸适用域，就不应删除这 20% 样本，而应对高 `Npl` regime 单独建模、分区校准或做 OOF correction；否则全量 COV 很难自然降到 `0.10` 以下。

### Phase 4C：`Npl <= q80` 后取消 log 的目标形式对比

用户假设：剔除大尺寸/大承载力样本后，可能不再需要 log 目标变换。

追加两轮 100-trial Optuna：

| Experiment | Target mode | Target transform | 还原评估空间 |
|---|---|---|---|
| `phase4_npl_le_q80_G5_b4_lightgbm_r_no_log` | `r_over_npl` | none | `Nexp (kN)` |
| `phase4_npl_le_q80_G5_b4_lightgbm_raw_nexp_no_log` | `raw` | none | `Nexp (kN)` |

配置保持一致：

- 数据：`data/processed/final_feature_parameters_npl_le_q80.csv`
- Split：`regression_stratified`
- 模型：`G5 B4 LightGBM`
- KeML：enabled
- Optuna：100 trials
- 特征：B4

指标对比：

| Experiment | Target | Transform | Test RMSE | Test R² | Test COV | Test MAPE | Test a20 | CV J | CV RMSE | CV COV |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Npl <= q80` G5 B4 | `eta_u_over_npl` | `log` | 189.7342 | 0.9383 | 0.0969 | 5.6725% | 0.9613 | 6.7116 | 151.3911 | 0.1157 |
| `Npl <= q80` G5 B4 | `r_over_npl` | none | 194.1209 | 0.9354 | 0.1217 | 6.9109% | 0.9351 | 8.7889 | 160.1478 | 0.1694 |
| `Npl <= q80` G5 B4 | `raw Nexp` | none | 203.0340 | 0.9296 | 0.1736 | 8.7455% | 0.9081 | 9.3981 | 159.5695 | 0.2041 |

Best Optuna params：

```text
eta_u_over_npl + log:
  n_estimators = 561
  learning_rate = 0.0802803291
  num_leaves = 109
  subsample = 0.6922607996
  colsample_bytree = 0.8788455081

r_over_npl + no log:
  n_estimators = 296
  learning_rate = 0.1524049258
  num_leaves = 65
  subsample = 0.6669200246
  colsample_bytree = 0.7929161524

raw Nexp + no log:
  n_estimators = 780
  learning_rate = 0.0712416767
  num_leaves = 59
  subsample = 0.5537658420
  colsample_bytree = 0.8446323364
```

阶段判断：

- 剔除 `Npl` 最高 20% 后，`log(eta_u)` 仍然是最优目标形式。
- `r` 目标虽然从力学表达上合理，但直接拟合 `r = eta_u - 1` 后，test COV 从 `0.0969` 升到 `0.1217`，a20 从 `0.9613` 降到 `0.9351`。
- 直接预测原始 `Nexp` 最差，test COV 升到 `0.1736`，MAPE 升到 `8.7455%`，说明即使去掉高承载力 20%，原始力值空间仍然不适合作为当前主线目标。
- 当前主线应保留 `eta_u_over_npl + log`。后续若要进一步压 COV，应优先做高 `Npl` regime 的分区/校准，而不是取消 log 或改成 raw target。

## Log Coverage Audit

审计时间：2026-04-25

已对照以下位置：

```text
config/experiments/next_stage_feature_metric/
output/experiments/next_stage_feature_metric/
logs/experiments/next_stage_feature_metric/
```

当前已完成并拥有 `evaluation_report.json` 的实验，均已在本 plan 中记录指标或结论：

```text
output/experiments/next_stage_feature_metric/phase1_G1_core9_xgboost/evaluation_report.json
output/experiments/next_stage_feature_metric/phase1_G2_b4_xgboost/evaluation_report.json
output/experiments/next_stage_feature_metric/phase1_G3_core9_lightgbm/evaluation_report.json
output/experiments/next_stage_feature_metric/phase1_G4_core9_catboost/evaluation_report.json
output/experiments/next_stage_feature_metric/phase2_G5_b4_lightgbm/evaluation_report.json
output/experiments/next_stage_feature_metric/phase2_G6_bounded_mg_xgboost/evaluation_report.json
output/experiments/next_stage_feature_metric/phase2_G7_bounded_mg_lightgbm/evaluation_report.json
output/experiments/next_stage_feature_metric/ensembles/evaluation_report.json
output/experiments/next_stage_feature_metric/phase3_regime_calibration/evaluation_report.json
output/experiments/next_stage_feature_metric/phase3_W1_core9_lightgbm_high_npl_weight/evaluation_report.json
output/experiments/next_stage_feature_metric/phase3_W2_core9_lightgbm_regime_weight/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout_seed0_fixed/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout_seed0_fixed/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G3_core9_lightgbm_source_group_holdout_seed7_fixed/evaluation_report.json
output/experiments/next_stage_feature_metric/source_group_holdout/phase3_G5_b4_lightgbm_source_group_holdout_seed7_fixed/evaluation_report.json
output/experiments/next_stage_feature_metric/phase4_npl_le_q80_G5_b4_lightgbm/evaluation_report.json
output/experiments/next_stage_feature_metric/phase4_npl_le_q80_G5_b4_lightgbm_r_no_log/evaluation_report.json
output/experiments/next_stage_feature_metric/phase4_npl_le_q80_G5_b4_lightgbm_raw_nexp_no_log/evaluation_report.json
```

当前仅有 partial / not-run 状态的实验：

```text
phase4_standard_G5_b4_lightgbm: partial Optuna only, no evaluation_report.json
phase4_full_B4_plus_xi_lightgbm: config only, not run
phase4_full_B4_plus_scale_lightgbm: config only, not run
phase4_full_B4_plus_local_lightgbm: config only, not run
```

因此截至本次审计，没有发现已完成但未写入 plan 的 `evaluation_report.json`。补充写入的遗漏项为：

- `standard_cfst` 标签草稿统计与停用说明。
- `phase4_standard_G5_b4_lightgbm` partial Optuna 状态。
- `phase4_full_B4_plus_*` 三个 config-only / not-run 状态。
- source group fixed seed 输出目录。
- Phase 3 W1/W2 sample weighting 输出目录。
