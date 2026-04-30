# 角点检测优化总结

## 修改文件

### 1. `src/common/util.py`

#### 1.1 `slice()` — 新增 step 参数
- 支持稀疏采样，`step` 控制每隔几个顶点取一个点
- 用于 stdev 扫描时减少弯曲区域的密集点对趋势方向的干扰

#### 1.2 `average_of_angles()` — 新增 weights 参数
- 支持加权角度平均，直线段权重大，弯曲段权重小

#### 1.3 `angular_stdev()` — 新增 weights 参数
- 支持加权角度标准差计算

#### 1.4 `colinearity()` — 改用段方向计算
- **之前**：计算"从基准点到每个采样点"的角度，再做平均
- **现在**：计算相邻采样点之间的连线方向（段方向），做加权平均
- 权重 = `_straightness_weights()` 返回的直线度权重
- 段权重 = (两端点直线度均值) × 段长度

#### 1.5 `_straightness_weights()` — 新增函数
- 为每个采样点计算权重 = 直线度 × 距离
- **直线度计算**：取每个段与它后面第2个段（skip=2）比较方向变化量
  - 变化量 < π/3（60°）：`ratio = (1 - change/60°)`
  - 权重 = `ratio^6`（非线性快速衰减）
  - 0°变化 = 权重1.0，20°变化 ≈ 0.09，45°变化 ≈ 0
- **距离权重**：每个点的权重乘以它到相邻点的距离
  - 稀疏的直线段每个点权重高
  - 密集的弯曲区域即使点多，累积权重也被压低
- 最后做2轮平滑避免权重跳变

### 2. `src/common/vector.py`

#### 2.1 `from_vertex()` — spoke 方向修正
- `colinearity` 现在返回段方向，spoke_h（从角点向后扫）的段方向指向角点
- 需要加 π 翻转，变成"从角点向外"的方向
- `a_ih = a_ih + math.pi`

#### 2.2 `from_vertex()` — stdev 稀疏采样
- 保持扫描距离 `vec_len_for_stdev = round(8 * scalar)`（约734px）
- 新增 `stdev_step = max(1, round(0.75 * scalar))`（约7个顶点一跳）
- 实际采样点从约77个减少到约11个，覆盖相同距离范围
- 效果：减少弯曲区域的密集点对 stdev 的干扰

#### 2.3 `from_vertex()` — centroid_symmetry 新指标
- 计算质心到角点连线与两条 spoke 的有符号角度关系
- `angle_h_signed`：从质心方向到 spoke_h 的有符号角度（-180°~+180°）
- `angle_j_signed`：从质心方向到 spoke_j 的有符号角度
- 理想情况：一正一负，绝对值各约 half_angle（≈45°）
- `asymmetry = |angle_h + angle_j|`（理想≈0）
- `magnitude_dev = ||angle_h| - half| + ||angle_j| - half|`
- `centroid_symmetry = asymmetry + magnitude_dev`

#### 2.4 `score()` — 移除 bbox 边缘距离惩罚
- **之前**：`score_with_bbox()` 对离 bbox 边缘远的候选点加惩罚
- **现在**：`score_with_bbox()` 直接返回 `self.score()`
- 原因：贴边不代表是角点，很多假角点恰好在 bbox 边缘上

#### 2.5 `score()` — 钝角惩罚
- 100°起开始轻微惩罚（> π/18 = 10°超过90°）
- 120°后急剧上升（> π/6 = 30°超过90°，额外4倍惩罚）
- 示例：130°的总 angle 项是 90°的11倍，163°是33倍

#### 2.6 `score()` — 降低 stdev 权重
- 从 `11.0 * stdev²` 降到 `5.0 * stdev²`
- 原因：高权重会过度惩罚经过 tab/blank 区域的真实角点

#### 2.7 `_score_4_candidates()` — 加入 centroid_symmetry
- 在组合评分中加入 `0.5 * sum(各候选的 centroid_symmetry)`
- 仅用于从候选中选取最佳4个角点，不影响候选生成和过滤

#### 2.8 `merge_nearby_candidates()` — 改用像素距离
- **之前**：用顶点索引距离（`2 * scalar`个索引），不处理环绕
- **现在**：用像素距离（`5 * scalar ≈ 48px`），正确处理环绕
- 保留评分更低（更好）的候选点
- 解决了多边形首尾相接处候选点不合并的问题

## 修改前后对比

| Piece | 修改前 | 修改后 | 最大偏差(px) |
|-------|--------|--------|-------------|
| 1 | OK | OK | 12 |
| 2 | C0错误 | OK | 9 |
| 3 | OK | OK | 11 |
| 4 | C2,C3错误 | OK | 12 |
| 5 | C0错误 | OK | 6 |
| 6 | C0错误 | OK | 5 |
| 7 | OK | OK | 11 |
| 8 | OK | OK | 13 |
| 9 | C2错误 | OK | 21 |
| 10 | C1错误 | OK | 19 |

修改前：6/10 正确，修改后：10/10 正确
