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

- Group A：`output/experiments/compact_group_A_optuna100/`
- Group B：`output/experiments/compact_group_B_optuna100/`
- Group C：`output/experiments/compact_group_C_optuna100/`
- Group D：`output/experiments/compact_group_D_optuna100/`

对应最优参数文件：

- Group A：`logs/experiments/compact_group_A_best_params.json`
- Group B：`logs/experiments/compact_group_B_best_params.json`
- Group C：`logs/experiments/compact_group_C_best_params.json`
- Group D：`logs/experiments/compact_group_D_best_params.json`

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

---

## 11. `compact_group_B` 特征参数重构筛选计划

### 11.1 本阶段目标

在不考虑 18 特征主线的前提下，以 `config/experiments/compact_group_B.yaml` 为统一母体，对当前紧凑特征组做新一轮结构性筛选。

本阶段目标不是立刻切换到 Box-Cox 目标族，而是先在 **不超过 11 个特征** 的约束下，筛选出 1~2 条最有潜力的紧凑特征主线，再在这些主线上继续做目标族优化。

### 11.2 统一实验约束

- 统一基线配置：`config/experiments/compact_group_B.yaml`
- 统一训练主线：`eta_u_over_npl + log + KeML`
- 统一评估口径：原始 `Nexp` 空间下的 `RMSE / MAE / R² / MAPE / μ / COV / a20-index`
- 每组均执行：**100 轮 Optuna 超参数调优**
- 本阶段默认：**不开启样本加权**
- 特征参数个数上限：**11 个**
- 本阶段只做特征组重构，不引入 Box-Cox 目标族

### 11.3 候选重构组

#### Group B0：当前基线组（9个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e/h`
- `e_bar`
- `e1/e2`

作用：
- 作为本阶段所有重构实验的共同对照基线
- 与后续重构组统一比较 Test `R² / COV / μ / a20-index`

#### Group B3：去掉 `e1/e2` 的偏心简化组（8个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e/h`
- `e_bar`

目的：
- 复验 `e1/e2` 在修复 665 行样本后的数据底座上是否仍值得长期保留
- 判断它是否只是工程口径增强项，还是已成为不可替代的偏心模式变量

#### Group B4：加入 `b/h` 的形状增强组（10个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e/h`
- `e_bar`
- `e1/e2`
- `b/h`

目的：
- 在已保留 `Re / te / ke` 的前提下，补充一个显式形状比例变量
- 验证 `b/h` 是否能帮助当前多截面统一框架进一步压低 `COV`

#### Group B5：加入 `r0/h` 的圆角形状增强组（10个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e/h`
- `e_bar`
- `e1/e2`
- `r0/h`

目的：
- 验证显式圆角/圆端比例信息是否仍能为统一截面表达提供额外收益
- 与 `B4` 形成同口径对照，比较 `b/h` 和 `r0/h` 哪个更有价值

#### Group B6：加入 `b/h + r0/h` 的 11 特征上限增强组（11个）
- `fy (MPa)`
- `fc (MPa)`
- `Re (mm)`
- `te (mm)`
- `ke`
- `lambda_bar`
- `e/h`
- `e_bar`
- `e1/e2`
- `b/h`
- `r0/h`

目的：
- 在 11 个特征的上限内，同时保留偏心增强与形状解释增强
- 判断是否存在一条比当前 B 组更强、同时仍保持紧凑的候选主线

### 11.4 执行顺序

本阶段按以下顺序串行执行：

1. `B0`：当前 `compact_group_B` 基线复验
2. `B3`：去掉 `e1/e2`
3. `B4`：加入 `b/h`
4. `B5`：加入 `r0/h`
5. `B6`：加入 `b/h + r0/h`

### 11.5 结果筛选目标

本阶段结束后，需要从以上候选组中筛出：

1. **最有潜力的主候选组**
   - 作为后续 Box-Cox 目标族实验的首选特征组
2. **一个清晰的对照组**
   - 用于判断 Box-Cox 改善究竟主要来自目标变换，还是来自特征结构变化

### 11.6 本阶段输出目录约定

- 实验配置目录：`config/experiments/compact_group_B_rebuild/`
- 实验结果目录：`output/experiments/compact_group_B_rebuild/`
- 实验日志/Optuna/最优参数目录：`logs/experiments/compact_group_B_rebuild/`

### 11.7 实验结果汇总

| Group | 特征组说明 | RMSE | MAE | R² | MAPE | μ | COV | a20-index |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 当前 `compact_group_B` 基线 | 585.19 | 167.99 | 0.9735 | 7.58% | 1.0094 | 0.1422 | 0.9358 |
| B3 | 去掉 `e1/e2` | 473.01 | 157.63 | 0.9827 | 7.56% | 1.0105 | 0.1428 | 0.9317 |
| B4 | `B + b/h` | 454.82 | 156.02 | 0.9840 | 7.20% | 1.0099 | 0.1358 | 0.9434 |
| B5 | `B + r0/h` | 503.38 | 161.11 | 0.9804 | 7.34% | 1.0095 | 0.1393 | 0.9399 |
| B6 | `B + b/h + r0/h` | 416.43 | 149.63 | 0.9866 | 7.18% | 1.0111 | 0.1431 | 0.9446 |
| B7 | `B + b/h - e1/e2` | 552.87 | 162.68 | 0.9763 | 7.42% | 1.0111 | 0.1427 | 0.9364 |

### 11.8 结果分析

#### 11.8.1 `b/h` 是当前最有价值的新增变量

从 `B4` 相比 `B0` 的结果看：

- `R²` 明显提升（0.9735 → 0.9840）
- `COV` 明显下降（0.1422 → 0.1358）
- `a20-index` 也提升（0.9358 → 0.9434）

说明在当前多截面统一预测框架下，`b/h` 提供了当前 `Re / te / ke` 尚未完全吸收的显式形状比例信息，是本阶段最明确有效的新增变量。

#### 11.8.2 单独加入 `r0/h` 有收益，但弱于 `b/h`

`B5` 相比 `B0` 也有改进：

- `R²` 提升到 0.9804
- `COV` 降到 0.1393

但提升幅度整体弱于 `B4`，说明在当前数据和统一表达方式下，`r0/h` 的独立收益存在，但优先级低于 `b/h`。

#### 11.8.3 去掉 `e1/e2` 仍然更偏向提升 `R²`，不利于工程口径

`B3` 的表现与此前迭代结论一致：

- `R²` 提升到 0.9827
- 但 `COV` 没有改善，反而略变差（0.1428）
- `a20-index` 也更低

这说明 `e1/e2` 仍然更偏向工程/论文口径下的偏心模式识别收益，不建议在当前阶段直接去掉。

补充对照：

- `B7 = B + b/h - e1/e2` 的结果为：`R² = 0.9763`，`COV = 0.1427`，`a20-index = 0.9364`
- 相比 `B4 = B + b/h`，删除 `e1/e2` 后：
  - `R²` 明显下降（0.9840 → 0.9763）
  - `COV` 明显变差（0.1358 → 0.1427）
  - `a20-index` 也明显下降（0.9434 → 0.9364）

这进一步说明：即便已经加入 `b/h`，`e1/e2` 仍然是当前最有价值的偏心模式信息变量之一。

#### 11.8.4 `B6` 是当前 `R²` 最高组，但不是最均衡组

`B6` 在本阶段取得了：

- 最低 `RMSE = 416.43`
- 最高 `R² = 0.9866`
- 最高 `a20-index = 0.9446`

但它的 `COV = 0.1431`，反而高于 `B4` 与 `B5`，且 `μ` 也略偏离 1。

这说明：

- `b/h + r0/h` 的双形状增强确实能进一步提高拟合精度
- 但当前它更偏向于提升 `RMSE / R² / a20`
- 还没有同步改善 `μ / COV`

### 11.9 本阶段筛选结论

基于本轮 `compact_group_B` 特征参数重构实验，可形成如下判断：

1. **`b/h` 是当前最值得保留的新增形状变量**
2. **`r0/h` 可以作为次优补充变量，但独立收益弱于 `b/h`**
3. **`e1/e2` 仍不建议直接删除**；`B3` 与 `B7` 都验证了，去掉它后更容易出现 `R² / COV / a20` 同时退化或至少失衡
4. **当前最有潜力的两条后续路线分别是：**
   - **主候选组：`B4 = B + b/h`**
     - 在 `R² / COV / a20-index` 三者之间更均衡
     - 更适合作为后续 Box-Cox 目标族实验的首选特征组
   - **对照组：`B6 = B + b/h + r0/h`**
     - 当前 `R²` 和 `RMSE` 最优
     - 适合作为“更强拟合、但 COV 尚未同步改善”的对照路线

### 11.10 下一步建议

后续 Box-Cox 目标族实验建议优先基于以下两组展开：

1. `B4 = fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2, b/h`
2. `B6 = fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2, b/h, r0/h`

其中：

- `B4` 用于优先追求更均衡的 `R²-COV-a20` 折中
- `B6` 用于测试“在更高拟合精度基础上，目标族变换能否继续把 `COV` 压下来”

---

## 12. B4 / B6 的 Box-Cox 目标族扫描计划

### 12.1 目标

在当前已经筛出的两条最有潜力的紧凑特征路线基础上，继续评估目标变换本身能否进一步改善 `R²` 与 `COV` 的折中关系。

本阶段不再继续改动特征组，只扫描 `eta_u_over_npl` 的目标变换形式。

### 12.2 扫描对象

- `B4 = fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2, b/h`
- `B6 = fy, fc, Re, te, ke, lambda_bar, e/h, e_bar, e1/e2, b/h, r0/h`

### 12.3 目标族定义

统一建模目标仍为 `eta_u_over_npl`，仅改变 target transform：

- `log`
- `boxcox_0.25`
- `boxcox_0.50`
- `boxcox_0.75`
- `boxcox_1.00`

其中：

- `boxcox_1.00` 与不做额外变换的线性 `eta_u` 同口径
- `log` 可视为 `λ -> 0` 的参考端点
- 中间的 `0.25 / 0.50 / 0.75` 用于寻找 `R²-COV` 折中更优的中间形态

### 12.4 统一实验约束

- 每组均执行：**100 轮 Optuna 超参数调优**
- 默认不开启样本加权
- 统一 split / cv / selection objective
- 不改 KeML 开关
- 不改基线特征组，只改 target transform

### 12.5 结果筛选标准

每组统一汇报：

- Test `R²`
- Test `COV`
- `μ`
- `a20-index`

最终希望回答两个问题：

1. `B4` 或 `B6` 上，是否存在比当前 `log` 更优的 Box-Cox 中间点
2. 若存在，该改善更偏向于：
   - 提升 `R²`
   - 降低 `COV`
   - 或两者同时改善

### 12.6 当前结果汇总（B4 已补完）

> 说明：`B4_boxcox_1.00` 已于 2026-04-21 补跑完成；`B6` 五组尚未开始，下一步按串行方式继续执行。

| Group | Target transform | R² | COV | μ | a20-index | RMSE | MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| B4 | `log` | 0.9849 | 0.1363 | 1.0103 | 0.9399 | 441.35 | 151.11 |
| B4 | `boxcox_0.25` | 0.9781 | 0.1407 | 1.0130 | 0.9393 | 531.81 | 158.89 |
| B4 | `boxcox_0.50` | 0.9866 | 0.1652 | 1.0151 | 0.9382 | 416.09 | 150.23 |
| B4 | `boxcox_0.75` | 0.9835 | 0.1804 | 1.0172 | 0.9376 | 461.86 | 149.55 |
| B4 | `boxcox_1.00` | 0.9811 | 0.1996 | 1.0196 | 0.9317 | 493.92 | 152.36 |

`B4_boxcox_1.00` 关键运行记录：

- 配置文件：`config/experiments/boxcox_scan/B4_boxcox_1_00.yaml`
- 输出目录：`output/experiments/boxcox_scan/B4_boxcox_1_00_optuna100/`
- 评估报告：`output/experiments/boxcox_scan/B4_boxcox_1_00_optuna100/evaluation_report.json`
- Optuna study：`logs/experiments/boxcox_scan/B4_boxcox_1_00_optuna.db`
- 最优参数：`logs/experiments/boxcox_scan/B4_boxcox_1_00_best_params.json`
- 关键训练日志：`/tmp/claude-1001/-home-thelya-Work-CFST-Alchemy-Lab-cfst-predictitor/5dc17af7-055c-4574-a69c-4a1ea48ca13f/tasks/baly78y3o.output`

`B4_boxcox_1.00` 补充指标：

- CV：`J = 6.2998`，`RMSE = 456.81`，`MAE = 148.52`，`R² = 0.9718`，`μ = 1.0187`，`COV = 0.2219`，`a20-index = 0.9248`
- Train：`RMSE = 125.24`，`MAE = 51.17`，`R² = 0.9979`，`μ = 1.0043`，`COV = 0.0699`，`a20-index = 0.9891`
- Test：`RMSE = 493.92`，`MAE = 152.36`，`R² = 0.9811`，`μ = 1.0196`，`COV = 0.1996`，`a20-index = 0.9317`
- Final `n_estimators = 1207`
- Optuna trial 进度：`79 -> 179`
- 过拟合检查：原始空间 `RMSE(train/test) ratio = 3.94`，为当前 B4 扫描中最明显的过拟合点之一

### 12.7 当前阶段性判断（B4 全扫描完成后）

基于当前已经补完的 `B4` 五个点结果，可形成如下判断：

1. **`log` 仍然是 B4 路线下最均衡的目标变换点**
   - 在五个点中，`log` 的 `COV` 最低（0.1363）
   - `μ` 也最接近 1（1.0103）
   - 虽然 `R²` 不是最高，但整体折中最好

2. **中等 λ 值仍然主要是在用更高 `R² / 更低 RMSE` 换更差的比例稳定性**
   - `boxcox_0.50` 取得了 B4 路线下最高 `R² = 0.9866` 与最低 `RMSE = 416.09`
   - 但其 `COV = 0.1652`，明显差于 `log`
   - `boxcox_0.75` 进一步把 `COV` 推高到 `0.1804`

3. **`boxcox_1.00` 明确不是 B4 路线下的可取折中点**
   - 相比 `log`，`boxcox_1.00` 的 `R²` 从 `0.9849` 降到 `0.9811`
   - `COV` 从 `0.1363` 恶化到 `0.1996`
   - `μ` 从 `1.0103` 偏移到 `1.0196`
   - `a20-index` 也从 `0.9399` 下降到 `0.9317`

4. **B4 路线已经给出足够清晰的证据：当前不值得继续围绕 Box-Cox λ 本身深挖**
   - `log` 是当前 B4 的稳健参考端点
   - `boxcox_0.50` 可保留为“更高拟合但更差 COV”的典型对照点
   - 其余更接近线性端的变换并未显示出进入下一阶段主线的价值

### 12.8 当前串行执行计划

按用户要求，后续实验改为严格串行执行。下一步建议顺序为：

1. `B6_log`
2. `B6_boxcox_0.25`
3. `B6_boxcox_0.50`
4. `B6_boxcox_0.75`
5. `B6_boxcox_1.00`

这样可以在不并行占用资源的前提下，尽快判断：`B6` 是否也会复现与 `B4` 相同的 `R²-COV` 结构性拉扯。

### 12.9 `B6_log` 串行补跑结果

`B6_log` 已于 2026-04-21 串行完成，结果如下：

- 配置文件：`config/experiments/boxcox_scan/B6_log.yaml`
- 输出目录：`output/experiments/boxcox_scan/B6_log_optuna100/`
- 评估报告：`output/experiments/boxcox_scan/B6_log_optuna100/evaluation_report.json`
- Optuna study：`logs/experiments/boxcox_scan/B6_log_optuna.db`
- 最优参数：`logs/experiments/boxcox_scan/B6_log_best_params.json`
- 关键训练日志：`/tmp/claude-1001/-home-thelya-Work-CFST-Alchemy-Lab-cfst-predictitor/5dc17af7-055c-4574-a69c-4a1ea48ca13f/tasks/bomgra0va.output`

核心指标：

- CV：`J = 6.8051`，`RMSE = 525.00`，`MAE = 156.96`，`R² = 0.9626`，`μ = 1.0077`，`COV = 0.1536`，`a20-index = 0.9318`
- Train：`RMSE = 152.33`，`MAE = 62.14`，`R² = 0.9969`，`μ = 1.0016`，`COV = 0.0601`，`a20-index = 0.9893`
- Test：`RMSE = 447.48`，`MAE = 154.94`，`R² = 0.9845`，`μ = 1.0102`，`COV = 0.1386`，`a20-index = 0.9417`
- Final `n_estimators = 1261`
- Optuna trial 进度：`24 -> 124`
- 过拟合检查：原始空间 `RMSE(train/test) ratio = 2.94`

与当前 `B4_log` 对照：

- `B6_log` 的 `R²` 略低于 `B4_log`（`0.9845 < 0.9849`）
- `B6_log` 的 `COV` 略高于 `B4_log`（`0.1386 > 0.1363`）
- `μ` 两者几乎相同（`1.0102` vs `1.0103`）
- `a20-index` 则是 `B6_log` 略优（`0.9417 > 0.9399`）

当前判断：

1. **`B6_log` 没有形成对 `B4_log` 的明确统治**
   - `r0/h` 的加入并没有让 `B6` 在 `R²` 与 `COV` 上同时超过 `B4`
   - 目前更像是用极小幅度的 `a20-index` 收益，换来略差的 `RMSE / R² / COV`

2. **`B6` 路线依然值得继续扫 Box-Cox，但当前 `log` 端点并未显示出结构性突破**
   - 这说明 `B6` 的意义仍然更接近“更强拟合上限组/对照组”
   - 是否真的优于 `B4`，还要看 `boxcox_0.25 / 0.50 / 0.75 / 1.00` 的完整曲线

3. **当前证据继续支持：仅靠 target transform 端点切换，不足以解决 `R²-COV` 的结构性拉扯**
   - `B4_boxcox_1.00` 已经给出明显反证
   - `B6_log` 也没有直接把 `COV` 压进更优区间

### 12.10 `B6_boxcox_0.25` 串行补跑结果

`B6_boxcox_0.25` 已于 2026-04-21 串行完成，结果如下：

- 配置文件：`config/experiments/boxcox_scan/B6_boxcox_0_25.yaml`
- 输出目录：`output/experiments/boxcox_scan/B6_boxcox_0_25_optuna100/`
- 评估报告：`output/experiments/boxcox_scan/B6_boxcox_0_25_optuna100/evaluation_report.json`
- Optuna study：`logs/experiments/boxcox_scan/B6_boxcox_0_25_optuna.db`
- 最优参数：`logs/experiments/boxcox_scan/B6_boxcox_0_25_best_params.json`
- 关键训练日志：`/tmp/claude-1001/-home-thelya-Work-CFST-Alchemy-Lab-cfst-predictitor/5dc17af7-055c-4574-a69c-4a1ea48ca13f/tasks/bskhkvqzc.output`

核心指标：

- CV：`J = 6.0652`，`RMSE = 491.51`，`MAE = 150.31`，`R² = 0.9671`，`μ = 1.0106`，`COV = 0.1622`，`a20-index = 0.9351`
- Train：`RMSE = 141.54`，`MAE = 59.46`，`R² = 0.9973`，`μ = 1.0025`，`COV = 0.0609`，`a20-index = 0.9892`
- Test：`RMSE = 424.70`，`MAE = 147.77`，`R² = 0.9860`，`μ = 1.0133`，`COV = 0.1441`，`a20-index = 0.9411`
- Final `n_estimators = 1258`
- Optuna trial 进度：`0 -> 100`
- 过拟合检查：原始空间 `RMSE(train/test) ratio = 3.00`

与 `B6_log` 对照：

- `R²` 从 `0.9845` 提升到 `0.9860`
- `RMSE` 从 `447.48` 降到 `424.70`
- 但 `COV` 从 `0.1386` 恶化到 `0.1441`
- `μ` 从 `1.0102` 偏移到 `1.0133`
- `a20-index` 基本持平并略降（`0.9417 -> 0.9411`）

当前判断：

1. **`B6_boxcox_0.25` 延续了与 B4 相同的结构性模式：更高 `R² / 更低 RMSE`，但更差 `COV`**
   - 这说明 `boxcox_0.25` 仍然没有改变当前主矛盾
   - 它只是把模型推向更强调拟合优度的方向

2. **到目前为止，`B6` 路线还没有出现比 `B6_log` 更优的均衡点**
   - `boxcox_0.25` 虽然在 `R²` 上超过了 `log`
   - 但 `COV / μ / a20-index` 并没有同步改善

3. **`B6` 正在复现 B4 的同类规律**
   - 如果后续 `boxcox_0.50 / 0.75 / 1.00` 继续沿这个方向发展
   - 则可进一步确认：当前 Box-Cox 扫描的主要作用更接近于生成“高 `R²` 对照点”，而不是找到真正更均衡的主线

### 12.11 `B6_boxcox_0.50` 串行补跑结果

`B6_boxcox_0.50` 已于 2026-04-21 串行完成，结果如下：

- 配置文件：`config/experiments/boxcox_scan/B6_boxcox_0_50.yaml`
- 输出目录：`output/experiments/boxcox_scan/B6_boxcox_0_50_optuna100/`
- 评估报告：`output/experiments/boxcox_scan/B6_boxcox_0_50_optuna100/evaluation_report.json`
- Optuna study：`logs/experiments/boxcox_scan/B6_boxcox_0_50_optuna.db`
- 最优参数：`logs/experiments/boxcox_scan/B6_boxcox_0_50_best_params.json`
- 关键训练日志：`/tmp/claude-1001/-home-thelya-Work-CFST-Alchemy-Lab-cfst-predictitor/5dc17af7-055c-4574-a69c-4a1ea48ca13f/tasks/b49uk8isp.output`

核心指标：

- CV：`J = 6.0533`，`RMSE = 477.85`，`MAE = 147.13`，`R² = 0.9690`，`μ = 1.0130`，`COV = 0.1810`，`a20-index = 0.9348`
- Train：`RMSE = 134.91`，`MAE = 54.86`，`R² = 0.9976`，`μ = 1.0032`，`COV = 0.0613`，`a20-index = 0.9893`
- Test：`RMSE = 434.00`，`MAE = 145.83`，`R² = 0.9854`，`μ = 1.0153`，`COV = 0.1597`，`a20-index = 0.9446`
- Final `n_estimators = 2350`
- Optuna trial 进度：`0 -> 100`
- 过拟合检查：原始空间 `RMSE(train/test) ratio = 3.22`

与 `B6_boxcox_0.25` 对照：

- `R²` 从 `0.9860` 略降到 `0.9854`
- `RMSE` 从 `424.70` 回升到 `434.00`
- `COV` 从 `0.1441` 进一步恶化到 `0.1597`
- `μ` 从 `1.0133` 继续偏移到 `1.0153`
- `a20-index` 则略升到当前已完成 `B6` 路线中的最高值（`0.9446`）

当前判断：

1. **`B6_boxcox_0.50` 没有带来更优折中，反而继续放大了 `COV / μ` 代价**
   - 这说明随着 λ 增大，`B6` 也在逐步走向“更差比例稳定性”的一侧
   - 与 B4 的扫描趋势保持一致

2. **当前已完成的 `B6` 三个点中，`log` 仍然是最均衡点**
   - `boxcox_0.25` 给出更高 `R²`
   - `boxcox_0.50` 给出更高 `a20-index`
   - 但两者都没有同时改善 `COV / μ / R²`

3. **`B6` 路线下，Box-Cox 中间点的主要价值正在变成“不同指标偏好的对照点”**
   - `log`：当前最均衡
   - `boxcox_0.25`：更高 `R² / 更低 RMSE`
   - `boxcox_0.50`：更高 `a20-index`
   - 但尚未出现真正能同时改善主要指标的点

下一步按串行继续：`B6_boxcox_0.75`

### 12.12 `B6_boxcox_0.75` 串行补跑结果

`B6_boxcox_0.75` 已于 2026-04-21 串行完成，结果如下：

- 配置文件：`config/experiments/boxcox_scan/B6_boxcox_0_75.yaml`
- 输出目录：`output/experiments/boxcox_scan/B6_boxcox_0_75_optuna100/`
- 评估报告：`output/experiments/boxcox_scan/B6_boxcox_0_75_optuna100/evaluation_report.json`
- Optuna study：`logs/experiments/boxcox_scan/B6_boxcox_0_75_optuna.db`
- 最优参数：`logs/experiments/boxcox_scan/B6_boxcox_0_75_best_params.json`
- 关键训练日志：`/tmp/claude-1001/-home-thelya-Work-CFST-Alchemy-Lab-cfst-predictitor/5dc17af7-055c-4574-a69c-4a1ea48ca13f/tasks/baywi88kf.output`

核心指标：

- CV：`J = 6.4143`，`RMSE = 477.09`，`MAE = 147.48`，`R² = 0.9693`，`μ = 1.0158`，`COV = 0.2018`，`a20-index = 0.9323`
- Train：`RMSE = 126.79`，`MAE = 53.28`，`R² = 0.9979`，`μ = 1.0039`，`COV = 0.0653`，`a20-index = 0.9892`
- Test：`RMSE = 428.39`，`MAE = 143.76`，`R² = 0.9858`，`μ = 1.0168`，`COV = 0.1754`，`a20-index = 0.9399`
- Final `n_estimators = 2255`
- Optuna trial 进度：`0 -> 100`
- 过拟合检查：原始空间 `RMSE(train/test) ratio = 3.38`

与 `B6_boxcox_0.50` 对照：

- `R²` 从 `0.9854` 小幅回升到 `0.9858`
- `RMSE` 从 `434.00` 降到 `428.39`
- 但 `COV` 从 `0.1597` 进一步恶化到 `0.1754`
- `μ` 从 `1.0153` 继续偏移到 `1.0168`
- `a20-index` 从 `0.9446` 回落到 `0.9399`

当前判断：

1. **`B6_boxcox_0.75` 仍然没有形成比 `B6_log` 更优的均衡点**
   - 虽然 `RMSE` 继续低于 `log`
   - 但 `COV / μ / a20-index` 并未同步改善

2. **随着 λ 从 `0.25 -> 0.75` 增大，`B6` 路线的 `COV / μ` 正持续恶化**
   - `boxcox_0.25`：`COV = 0.1441`
   - `boxcox_0.50`：`COV = 0.1597`
   - `boxcox_0.75`：`COV = 0.1754`
   - 而 `R²` 并没有同步形成持续上升趋势

3. **当前证据进一步支持：`B6_log` 仍是 B6 路线下最均衡的目标变换点**
   - 中间 λ 点更多是在提供不同指标偏好的对照点
   - 而不是给出真正同时改善 `R² / COV / μ` 的新主线

下一步按串行继续：`B6_boxcox_1.00`

### 12.13 `B6_boxcox_1.00` 串行补跑结果

`B6_boxcox_1.00` 已于 2026-04-21 串行完成，结果如下：

- 配置文件：`config/experiments/boxcox_scan/B6_boxcox_1_00.yaml`
- 输出目录：`output/experiments/boxcox_scan/B6_boxcox_1_00_optuna100/`
- 评估报告：`output/experiments/boxcox_scan/B6_boxcox_1_00_optuna100/evaluation_report.json`
- Optuna study：`logs/experiments/boxcox_scan/B6_boxcox_1_00_optuna.db`
- 最优参数：`logs/experiments/boxcox_scan/B6_boxcox_1_00_best_params.json`
- 关键训练日志：`/tmp/claude-1001/-home-thelya-Work-CFST-Alchemy-Lab-cfst-predictitor/5dc17af7-055c-4574-a69c-4a1ea48ca13f/tasks/br4qt5vss.output`

核心指标：

- CV：`J = 6.4610`，`RMSE = 459.16`，`MAE = 152.05`，`R² = 0.9716`，`μ = 1.0189`，`COV = 0.2271`，`a20-index = 0.9215`
- Train：`RMSE = 140.55`，`MAE = 60.47`，`R² = 0.9974`，`μ = 1.0054`，`COV = 0.0802`，`a20-index = 0.9864`
- Test：`RMSE = 460.89`，`MAE = 154.17`，`R² = 0.9835`，`μ = 1.0185`，`COV = 0.1989`，`a20-index = 0.9317`
- Final `n_estimators = 1493`
- Optuna trial 进度：`0 -> 100`
- 过拟合检查：原始空间 `RMSE(train/test) ratio = 3.28`

与 `B6_boxcox_0.75` 对照：

- `R²` 从 `0.9858` 下降到 `0.9835`
- `RMSE` 从 `428.39` 回升到 `460.89`
- `COV` 从 `0.1754` 进一步恶化到 `0.1989`
- `μ` 从 `1.0168` 继续偏移到 `1.0185`
- `a20-index` 从 `0.9399` 继续下降到 `0.9317`

当前判断：

1. **`B6_boxcox_1.00` 明确不是 B6 路线下可取的折中点**
   - 它几乎沿着与 `B4_boxcox_1.00` 相同的方向恶化
   - `R² / RMSE / COV / μ / a20-index` 没有任何一项形成对 `log` 的综合优势

2. **B6 全扫描已经给出清晰结论：`log` 仍然是最均衡点**
   - `log`：当前最均衡，`COV` 最低（0.1386），`μ` 也最接近 1（1.0102）
   - `boxcox_0.25`：更高 `R²`、更低 `RMSE`
   - `boxcox_0.50`：更高 `a20-index`
   - `0.75 / 1.00`：进一步放大 `COV / μ` 代价，已不具备主线价值

3. **B6 与 B4 共同支持同一结论：当前 Box-Cox 扫描主要在生成“不同指标偏好的对照点”，而不是新的主线突破**
   - 两条路线都没有找到同时改善 `R² / COV / μ` 的中间 λ
   - 越接近线性端，比例稳定性恶化越明显
   - 当前阶段更值得继续的方向，应该转向计划中尚未实现的 `log + smearing`，或进入后续 Stage-B 主干对比，而不是继续深挖 λ 本身

### 12.14 当前阶段性结论（B4 / B6 Box-Cox 全扫描完成后）

截至 2026-04-21，`B4` 与 `B6` 的五点 Box-Cox 扫描均已补完，可形成当前阶段统一判断：

1. **`log` 是当前两条紧凑主线下最稳定、最均衡的 target transform**
   - `B4_log`：`R² = 0.9849`，`COV = 0.1363`，`μ = 1.0103`
   - `B6_log`：`R² = 0.9845`，`COV = 0.1386`，`μ = 1.0102`

2. **中间 Box-Cox λ 点的收益具有明显“单指标偏置”**
   - 某些点可带来更高 `R²` 或更低 `RMSE`
   - 某些点可带来略高 `a20-index`
   - 但没有任何一个点能同时压低 `COV` 并保持更优 `μ`

3. **越向线性端移动，`COV / μ` 恶化越明显**
   - `B4_boxcox_1.00` 与 `B6_boxcox_1.00` 都已给出明确反证
   - 说明当前问题并不只是 target transform 端点选得不对，而更像是误差结构本身尚未被当前主线充分吸收

4. **下一阶段不建议继续围绕 λ 做更多细化扫描**
   - 当前更值得投入的方向是：
     - `log + smearing` 路线验证
     - 或按计划进入 Stage-B，冻结共享 protocol 后做多 backbone 对照
