# 计划：为角点检测算法添加"直线交点距离"因子

## 问题分析

当前角点检测算法（`Candidate.from_vertex()` + `score()`）使用以下因子判断角点：
- **angular_width**：两条辐条（spoke）之间的角度，理想值为 90°
- **offset_from_center**：角平分线与指向质心方向的偏差
- **stdev**：辐条上点的方向标准差（衡量直线程度）
- **curve_score**：该点是否在曲线上（0=角点，1=圆弧）

**核心缺陷**：当拼图块的角是钝角时（>100°），算法容易将其误判为突起/凹陷上的点而拒绝。而突起/凹陷上的点虽然角度可能接近90°，但其两侧的趋势线是近似平行的，不是在顶点处相交的。

## 新因子原理

```
角点情况：
    |
    |  ← 趋势线2
    |/
    * ← 顶点（两条趋势线在此处相交）
   /
  / ← 趋势线1

突起情况：
  ____
 /    \  ← 突起
/      \
-------- ← 趋势线1 和 趋势线2 几乎平行（同一条直边）
   * ← 顶点在两条平行线中间，远离交点
```

- **角点**：两侧各有一条较长的直线段，这两条直线在顶点附近相交
- **突起/凹陷**：两侧的直线段近似平行（属于同一条边），顶点在两条平行线中间，远离交点
- **平边**：完全没有突起凹陷的直边，两侧直线完全共线

## 实现步骤

### 步骤 1：在 `Candidate.__init__()` 中添加 `line_intersection_dist` 属性

**文件**: [vector.py](file:///f:/work_Puzzle_github/puzzle-bot/src/common/vector.py#L271-L280)

在 `__init__` 方法中添加新参数 `line_intersection_dist=10000`，默认值设为较大值（表示无效/平行线情况）。

### 步骤 2：在 `Candidate.from_vertex()` 中计算直线交点距离

**文件**: [vector.py](file:///f:/work_Puzzle_github/puzzle-bot/src/common/vector.py#L197-L269)

在 `from_vertex()` 方法中，计算完 `curve_score` 之后、构造 `Candidate` 对象之前，添加以下逻辑：

```python
# 计算两侧趋势线的交点距离
trend_len = round(18 * scalar)     # 趋势线采样窗口长度
trend_skip = round(3 * scalar)     # 跳过顶点附近弧度区域

pts_backward = util.slice(vertices, i - trend_len - trend_skip, i - trend_skip)
pts_forward = util.slice(vertices, i + trend_skip + 1, i + trend_len + trend_skip)

line_intersection_dist = 10000  # 默认大值

if len(pts_backward) >= 3 and len(pts_forward) >= 3:
    angle_back = util.trendline(pts_backward)
    angle_fwd = util.trendline(pts_forward)

    # 检查两条趋势线是否近似平行
    angle_diff = util.compare_angles(angle_back, angle_fwd)
    is_parallel = angle_diff < (20 * math.pi / 180)  # 夹角小于20°视为平行

    if not is_parallel:
        line_back = util.line_from_angle_and_point(angle=angle_back, point=pts_backward[-1], length=500)
        line_fwd = util.line_from_angle_and_point(angle=angle_fwd, point=pts_forward[0], length=500)
        intersection_pt = util.intersection(line_back, line_fwd)
        if intersection_pt is not None:
            line_intersection_dist = util.distance(v_i, intersection_pt)
```

**关键设计决策**：
- `trend_skip = 3*scalar`：跳过顶点附近的弧形区域，确保趋势线反映的是直线段的方向
- `trend_len = 18*scalar`：足够长的窗口来捕捉突起/凹陷两侧的直线段
- 平行判定阈值 20°：如果两条趋势线夹角小于 20°，视为平行（突起/平边情况）
- `util.intersection()` 返回 `None` 时保持默认大值

### 步骤 3：在 `Candidate.score()` 中加入交点距离惩罚项

**文件**: [vector.py](file:///f:/work_Puzzle_github/puzzle-bot/src/common/vector.py#L282-L290)

在 score 公式中添加交点距离的惩罚：

```python
def score(self):
    angle_error = max(0, self.angle - math.pi/2)
    penalty = 0.0
    if angle_error > math.pi / 4:
        penalty += 0.5 * (angle_error - math.pi / 4)
    if angle_error > math.pi / 3:
        penalty += 2.0 * (angle_error - math.pi / 3)

    # 交点距离惩罚：交点越远，越不可能是角点
    intersection_penalty = 0.0
    if self.line_intersection_dist > 15:  # 15像素以内不惩罚
        intersection_penalty = 0.05 * (self.line_intersection_dist - 15)

    score = (0.5 * (angle_error + penalty)) + (0.4 * self.offset_from_center) + (5.0 * (self.stdev ** 2)) + (0.8 * self.curve_score) + intersection_penalty
    return score
```

**参数说明**：
- `15` 像素容差：角点处的弧度可能导致交点略有偏移，15px 以内不惩罚
- `0.05` 权重系数：交点距离每增加 1px 加 0.05 分。例如交点距离 50px → 惩罚 1.75 分，100px → 惩罚 4.25 分

### 步骤 4：回退之前过度的角度放宽（可选调整）

上一次修改将 `CORNER_MAX_ANGLE_DEG` 从 150 放宽到 180，score 阈值从 3.0 放宽到 5.0。加入了交点距离因子后，可以考虑：

- **保留当前值**（180° / 5.0），让新因子负责区分突起和钝角角点
- **或者微调回**（如 170° / 4.5），避免过于宽松

建议先保持当前值不变，观察效果后再决定是否微调。

### 步骤 5：运行完整流水线验证

1. 运行 `python run_vectorize.py` 重新生成所有 SVG/JSON
2. 运行 `python visualize_problematic.py` 检测问题拼图块
3. 重点检查之前有问题的 7 块：#121, #76, #87, #23, #94, #48, #95
4. 检查总体问题块数量是否从 81 下降
5. 根据结果微调参数（`trend_len`, `trend_skip`, 平行判定阈值, 交点距离容差, 权重系数）

## 预期效果

- **钝角角点**（如 120°-150°）：虽然角度偏离 90°，但两侧趋势线确实在顶点附近相交 → 交点距离小 → 不会被误判为突起
- **突起/凹陷上的点**：两侧趋势线近似平行 → 交点距离大（或判定为平行） → 高惩罚 → 被正确排除
- **平边上的点**：两侧趋势线共线 → 判定为平行 → 高惩罚 → 被正确排除
