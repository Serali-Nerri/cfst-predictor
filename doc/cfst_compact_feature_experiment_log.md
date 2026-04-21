# 多截面 CFST 柱统一预测框架：紧凑特征组 100 轮调优对比实验

## 1. 背景

本轮实验基于当前仓库的统一主线：

- 报告目标：`Nexp (kN)`
- 训练主线：`eta_u_over_npl + log + KeML`
- 评估口径：原始 `Nexp` 空间下的 `RMSE / MAE / R² / MAPE / μ / COV / a20-index`

实验目的不是继续堆叠原始几何变量，而是在“多截面 CFST 柱统一预测框架”背景下，围绕以下四类核心机理筛选紧凑参数组：

1. 材料强度基础
2. 统一截面等效与形状效应
3. 整体稳定性
4. 偏心受压效应

所有候选组均单独执行 **100 轮 Optuna 超参数调优**，并保留独立输出、独立参数文件与独立 Optuna study。

---

## 2. 候选特征组

### Group A：统一机理紧凑主线组（8个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e/h`
- `e_bar`

### Group B：偏心增强组（9个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e/h`
- `e_bar`
- `e1/e2`

### Group C：形状解释增强组（9个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `r0/h`
- `b/h`
- `e_bar`

### Group D：超紧凑组（7个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e_bar`

---

## 3. 实验输出位置

基线紧凑组输出：

- Group A：`output/compact_group_A_optuna100/`
- Group B：`output/compact_group_B_optuna100/`
- Group C：`output/compact_group_C_optuna100/`
- Group D：`output/compact_group_D_optuna100/`

对应最优参数文件：

- Group A：`logs/compact_group_A_best_params.json`
- Group B：`logs/compact_group_B_best_params.json`
- Group C：`logs/compact_group_C_best_params.json`
- Group D：`logs/compact_group_D_best_params.json`

后续 `compact_group_B` 迭代优化实验统一采用以下目录约定：

- 实验配置目录：`config/experiments/compact_group_B_iterations/`
- 实验结果目录：`output/experiments/compact_group_B_iterations/`
- 实验日志/Optuna/最优参数目录：`logs/experiments/compact_group_B_iterations/`

---

## 4. 结果汇总

### 4.1 Test 集结果

| Group | RMSE | MAE | R² | MAPE | μ | COV | a20-index |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 502.67 | 134.49 | 0.9759 | 6.59% | 1.0062 | 0.1131 | 0.9431 |
| B | 527.66 | 133.34 | 0.9734 | 6.24% | 1.0051 | 0.1092 | 0.9526 |
| C | 552.62 | 141.69 | 0.9708 | 6.81% | 1.0072 | 0.1142 | 0.9431 |
| D | 603.59 | 146.96 | 0.9652 | 7.00% | 1.0065 | 0.1161 | 0.9386 |

### 4.2 CV 结果

| Group | J | RMSE | MAE | R² | MAPE | μ | COV | a20-index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1.6812 | 349.28 | 135.55 | 0.9849 | 7.00% | 1.0056 | 0.1158 | 0.9402 |
| B | 1.6124 | 350.28 | 134.78 | 0.9848 | 6.84% | 1.0055 | 0.1137 | 0.9427 |
| C | 1.6866 | 354.84 | 139.50 | 0.9847 | 7.26% | 1.0062 | 0.1188 | 0.9343 |
| D | 1.6910 | 354.99 | 141.11 | 0.9846 | 7.35% | 1.0063 | 0.1195 | 0.9337 |

### 4.3 Train 结果

| Group | RMSE | MAE | R² | MAPE | μ | COV | a20-index |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 154.76 | 65.58 | 0.9973 | 3.50% | 1.0017 | 0.0610 | 0.9888 |
| B | 145.45 | 58.74 | 0.9976 | 3.03% | 1.0014 | 0.0549 | 0.9913 |
| C | 164.47 | 69.03 | 0.9970 | 3.70% | 1.0019 | 0.0638 | 0.9877 |
| D | 177.32 | 75.77 | 0.9965 | 4.01% | 1.0020 | 0.0664 | 0.9872 |

---

## 5. 结果分析

### 5.1 Group A 与 Group B 是当前最有价值的两组

- **Group A** 在 Test 集上取得了最低的 `RMSE` 和最高的 `R²`
- **Group B** 在 Test 集上取得了最低的 `MAE`、最低的 `MAPE`、最低的 `COV`、最高的 `a20-index`，并且 `μ` 也最接近 1

这表明：

- 如果更看重平方误差和拟合优度，A 更占优
- 如果更看重论文/工程口径下的 `μ-COV-a20`，B 更占优

### 5.2 `e1/e2` 确实提供了偏心模式信息收益

Group B 相比 Group A 的关键变化只有：增加 `e1/e2`。

结果显示：

- Test `RMSE` 略变差（527.66 vs 502.67）
- Test `R²` 略变差（0.9734 vs 0.9759）
- 但 Test `MAE / MAPE / μ / COV / a20` 全部更优

这说明 `e1/e2` 没有提升全局平方误差意义下的最优性，但改善了比值分布和偏心模式识别，对工程口径更有利。

### 5.3 显式形状参数组（Group C）解释性更强，但当前预测性能不占优

Group C 显式加入了：

- `r0/h`
- `b/h`

理论上这更符合多截面统一解释框架，但在当前数据集与当前主线下，Group C 的 Test 指标整体不如 A/B。

因此，Group C 更适合作为：

- 形状机理解释组
- 论文中的机理对照组

而不是当前的最佳预测主线。

### 5.4 超紧凑组（Group D）损失明显

Group D 去掉了 `e/h`，仅保留 `e_bar` 作为偏心描述。结果显示：

- Test `RMSE` 最差
- Test `R²` 最差
- Test `a20-index` 最低

说明仅保留 `e_bar` 不足以完整表达偏心受压信息，偏心幅值信息本身仍然需要保留。

---

## 6. 排序建议

### 若按论文/工程综合口径排序
**B > A > C > D**

### 若按 `RMSE / R²` 排序
**A > B > C > D**

---

## 7. 是否还有必要继续增加超参数调优轮数？

### 7.1 当前结论

**不建议优先继续增加超参数调优轮数。**

原因如下：

1. 四组都已经完成 **100 轮独立 Optuna 调优**
2. A/B 的 CV 复合目标已经较接近，说明当前搜索已经把这两组推到了一个较稳定区域
3. 当前瓶颈更像是：
   - 特征组本身的信息边界
   - 对偏心模式的表达方式
   - `R²` 与 `COV` 目标之间的结构性权衡

### 7.2 与目标的差距

用户目标是：

- `R²` 尽可能贴近 `0.99`
- 同时保证 `COV <= 0.10`

当前最接近目标的是 Group B：

- Test `R² = 0.9734`
- Test `COV = 0.1092`

当前最接近高 `R²` 的是 Group A：

- Test `R² = 0.9759`
- Test `COV = 0.1131`

也就是说，当前差距主要不是“再多跑 100 轮就能自然补齐”的量级，而更像是：

- 需要继续改善特征表达
- 或者进一步调整主线目标定义/样本分层/误差结构

### 7.3 判断

如果继续从 100 轮加到 200 或 300 轮，**可能会有局部小幅改善**，但大概率只能带来：

- `RMSE / R²` 的边际优化
- 很难把 `COV` 从 `0.109~0.113` 稳定压到 `<= 0.10`

因此，后续优先级建议不是“先加轮数”，而是：

1. 在 A/B 两组基础上继续做小规模特征修正
2. 针对偏心与高误差 regime 继续补充机理变量
3. 必要时再对最终 1~2 个候选组追加更高轮数调优

---

## 8. 当前最推荐的后续路线

### 主推荐组：Group B
- `fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2`

理由：
- 更符合当前对 `μ / COV / a20-index` 的关注
- 偏心模式信息带来了工程口径收益
- 参数数量仍控制在 10 个以内

### 次推荐组：Group A
- `fy, fc, Re, te, ke, lambda_bar, e/h, e_bar`

理由：
- 更紧凑
- Test `RMSE / R²` 最好
- 更适合作为默认紧凑主线 baseline

---

## 9. 当前建议结论

在当前这一轮对比实验完成后，可以形成如下判断：

1. **A/B 明显优于 C/D**
2. **B 更适合作为论文/工程口径下的优先方案**
3. **A 更适合作为更稳健、更紧凑的 baseline**
4. **当前阶段没有必要优先靠“继续增加调参轮数”来解决问题**
5. 下一步更值得投入的是：在 A/B 的基础上继续做针对性机理修正，而不是盲目增加 Optuna 轮数

---

## 10. 修复 665 行样本后的迭代优化日志

> 说明：以下结果基于最新数据底座（已恢复此前跳过的 665 行样本），与上文旧版 A/B/C/D 对比结果不直接同口径。

### Iteration 1：分层切分升级（Group B + `e/h` 辅助分层）

- 基线特征组：`fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2`
- 本轮只改 1 个变量：在原有 `lambda_bar` 辅助分层基础上，新增 `e/h` 作为第二辅助分层特征
- 配置文件：`config/experiments/compact_group_B_iterations/compact_group_B_iter1_split_eh.yaml`
- 输出目录：`output/experiments/compact_group_B_iterations/compact_group_B_iter1_split_eh_optuna100/`

实际分层结果：
- `lambda_bar`：`used_bins = 3`
- `e/h`：`used_bins = 1`

说明 `e/h` 在当前稳定分层约束下没有形成有效附加分层，实际切分基本退化为原有方案。

Test 指标：
- `R² = 0.9735`
- `COV = 0.1422`
- `μ = 1.0094`
- `a20-index = 0.9358`

结论：
- 相比历史 Group B 记录未显示出收益
- 且 `COV` 明显高于目标线 `0.10`
- 单独增加 `e/h` 辅助分层不值得继续保留

### Iteration 2：分层切分升级（Group B + `e_bar` 辅助分层）

- 基线特征组：`fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2`
- 本轮只改 1 个变量：在原有 `lambda_bar` 辅助分层基础上，新增 `e_bar` 作为第二辅助分层特征
- 配置文件：`config/experiments/compact_group_B_iterations/compact_group_B_iter2_split_ebar.yaml`
- 输出目录：`output/experiments/compact_group_B_iterations/compact_group_B_iter2_split_ebar_optuna100/`

实际分层结果：
- `lambda_bar`：`used_bins = 3`
- `e_bar`：`used_bins = 1`

说明 `e_bar` 也没有形成有效附加分层，最终切分与 Iteration 1 的有效结构一致。

Test 指标：
- `R² = 0.9735`
- `COV = 0.1422`
- `μ = 1.0094`
- `a20-index = 0.9358`

结论：
- 与 Iteration 1 完全一致，无改进
- 说明在当前配置下，继续沿“额外偏心辅助分层”这一方向推进，短期内大概率不会带来收益
- 下一轮更值得尝试的单变量方向应转向：样本加权，或在 Group B 基础上进一步精简/替换特征

### Iteration 3：精简特征组（Group B 去掉 `e1/e2`）

- 基线特征组：`fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2`
- 本轮只改 1 个变量：删除 `e1/e2`
- 配置文件：`config/experiments/compact_group_B_iterations/compact_group_B_iter3_drop_e1e2.yaml`
- 输出目录：`output/experiments/compact_group_B_iterations/compact_group_B_iter3_drop_e1e2_optuna100/`

Test 指标：
- `R² = 0.9827`
- `COV = 0.1428`
- `μ = 1.0105`
- `a20-index = 0.9317`

结论：
- 相比 Iteration 1/2，`R²` 明显提升
- 但 `COV` 进一步变差，`a20-index` 也下降
- 说明在修复 665 行样本后的数据上，`e1/e2` 依然更有助于论文/工程口径指标，而删掉它更偏向于换取更高的拟合优度

### Iteration 4：加权训练（高偏心样本加权，`e/h > 0.1` 权重 1.5）

- 基线特征组：`fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2`
- 本轮只改 1 个变量：启用样本加权，对 `e/h > 0.1` 的样本赋予更高权重
- 配置文件：`config/experiments/compact_group_B_iterations/compact_group_B_iter4_weight_eh.yaml`
- 输出目录：`output/experiments/compact_group_B_iterations/compact_group_B_iter4_weight_eh_optuna100/`
- 加权设置：`threshold = 0.1`, `high_weight = 1.5`, `n_high_weight = 1955`

Test 指标：
- `R² = 0.9863`
- `COV = 0.1383`
- `μ = 1.0097`
- `a20-index = 0.9408`

结论：
- 相比 Iteration 3，`R²` 继续提升，`COV` 也有所下降
- 但距目标 `R² >= 0.99` 且 `COV <= 0.10` 仍有明显差距
- 样本加权是目前已验证有效的方向，值得继续只调一个变量做强度扫描

### Iteration 5：加权训练（高偏心样本加权，`e/h > 0.1` 权重 2.0）

- 基线特征组：`fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2`
- 本轮只改 1 个变量：将高偏心样本权重从 `1.5` 提高到 `2.0`
- 配置文件：`config/experiments/compact_group_B_iterations/compact_group_B_iter5_weight_eh2.yaml`
- 输出目录：`output/experiments/compact_group_B_iterations/compact_group_B_iter5_weight_eh2_optuna100/`
- 加权设置：`threshold = 0.1`, `high_weight = 2.0`, `n_high_weight = 1955`

Test 指标：
- `R² = 0.9861`
- `COV = 0.1370`
- `μ = 1.0090`
- `a20-index = 0.9370`

结论：
- 相比 Iteration 4，`COV` 继续小幅下降，`μ` 也更接近 1
- 但 `R²` 略有回落，`a20-index` 也略降
- 权重强度继续增大并不是单调增益，更值得继续只改“权重作用区间”而不是一味加大权重

### Iteration 6：加权训练（高偏心样本加权，`e/h > 0.2` 权重 2.0）

- 基线特征组：`fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2`
- 本轮只改 1 个变量：将高权重作用阈值从 `e/h > 0.1` 改为 `e/h > 0.2`
- 配置文件：`config/experiments/compact_group_B_iterations/compact_group_B_iter6_weight_eh_threshold02.yaml`
- 输出目录：`output/experiments/compact_group_B_iterations/compact_group_B_iter6_weight_eh_threshold02_optuna100/`
- 加权设置：`threshold = 0.2`, `high_weight = 2.0`, `n_high_weight = 1259`

Test 指标：
- `R² = 0.9716`
- `COV = 0.1367`
- `μ = 1.0095`
- `a20-index = 0.9382`

结论：
- `COV` 比 Iteration 5 略降，但 `R²` 明显回落
- 说明把高权重只集中到更极端偏心区间，会牺牲整体拟合稳定性
- 当前样本加权方向里，已验证的相对更优折中点仍是 Iteration 4/5，而不是进一步抬高阈值
