# 纹理连续性验证 — 实施计划（独立验证版）

## 目标

**不修改任何现有代码**，新建完全独立的验证脚本，读取已有的 `connectivity.json` 结果，用 PNG 原图的纹理信息对每个匹配对进行连续性检验，输出验证报告。

## 核心思路

正确拼接时，接缝两侧的颜色差异小（来自原图的平滑过渡区域）；错误拼接时，颜色差异大。

**三级验证策略**：

1. **纹理丰富度检测** → 判断边缘区域是否有足够的纹理信息
2. **颜色差异检测**（始终执行）→ Lab 空间 ΔE，过滤颜色明显不同的
3. **梯度方向一致性**（仅纹理丰富时执行）→ 沿边颜色变化趋势是否一致

---

## 关键设计原则

### 原则 1：采样向内偏移，避开边缘阴影区

```
拼图块截面示意：

     表面纹理区域（可靠）
  ┌─────────────────────┐
  │  ← 可靠的像素区域 →  │
  │                     │
  │   ✓ 在这里采样      │ ← 向内偏移 inner_offset 像素
  │                     │
  └──────边缘──────────────┘ ← 边缘处容易有：
         ↑                   - 拼图厚度的阴影
       接缝                  - 边缘提取的误差
                             - 背景色渗入
                             - 反光/高光

采样策略：
  ❌ 不在边界上采样（距边界 0~inner_offset 像素，不可靠）
  ✓ 从 inner_offset 开始向内采样（距边界 inner_offset ~ inner_offset+band_width）
```

**`inner_offset`** 默认 5-8 像素。

### 原则 2：纹理稀疏时自动降级

```
情况 A：纹理丰富（如天空渐变、花纹、文字）
  → 执行颜色 + 梯度双重验证（AND 逻辑：两个都差才排除）

情况 B：纹理稀疏（如大片白色区域，只有几个散点）
  → 只执行颜色验证，跳过梯度（因对齐误差会导致梯度方向不准）
```

---

## 数据来源（全部读取已有文件，不修改）

| 数据 | 路径 | 内容 |
|------|------|------|
| 匹配结果 | `{output}/5_connectivity/connectivity.json` | 每个 piece 每条边的匹配列表：pid, si, error, shift_x, shift_y |
| 边几何数据 | `{output}/4_deduped/side_{id}_{i}.json` | vertices（顶点坐标）、piece_center（拼图中心）、is_edge |
| 彩色图 | `{output}/3_vector/color_{id}.png` | 拼图块的 BGR 彩色图（矢量化阶段保存） |

---

## 实施步骤

### Step 1：新建 `src/common/texture_verify.py` — 核心验证模块

完全独立的模块，不 import connect、pieces 等，只依赖 cv2、numpy。

#### 1.1 数据加载函数

```python
def load_side_data(deduped_dir, piece_id, side_index):
    """读取 side_{pid}_{si}.json，返回 vertices, piece_center, is_edge"""

def load_color_image(vector_dir, piece_id):
    """加载 color_{pid}.png，返回 (bgr_image, binary_mask) 或 (None, None)
       mask 从彩色图生成：非零像素 = 拼图块"""
```

#### 1.2 `extract_inner_band(color_image, side_vertices, piece_center, binary_mask, ...)`

沿边提取向内偏移后的像素条带。

**算法**：
```
输入：彩色图(BGR)、边顶点列表、piece中心坐标、二值掩膜

对 side_vertices 均匀采样 n_samples 个点：
  对每个采样点 v[i]：
    1. 计算切线方向：tangent = v[i+1] - v[i-1]
    2. 计算法线方向：normal = rotate_90_degrees(tangent)，归一化
    3. 确定法线朝向：
       如果 normal 指向远离 piece_center（外侧）→ 取反
       确保 normal 指向 piece 内侧（朝向中心方向）
    4. 沿法线方向采样（向内偏移后）：
       for d in range(inner_offset, inner_offset + band_width):
         point = v[i] + normal * d
         如果 point 在掩膜内，取 color_image[point] 的颜色
       对每个采样点，取 band_width 个像素的**平均值**作为该点的颜色

输出：
  band_colors: np.array, shape=(n_valid_samples, 3), BGR 平均颜色
  band_gray:   np.array, shape=(n_valid_samples,), 灰度值
```

#### 1.3 `compute_texture_richness(band_gray)`

判断纹理丰富度。

```
1. diffs = np.abs(np.diff(band_gray))
2. significant = np.sum(diffs > GRADIENT_SIGNIFICANCE_THRESHOLD)
3. ratio = significant / (len(band_gray) - 1)
4. 返回 ratio (0~1)
   < 0.1 → 低纹理
   >= 0.1 → 有纹理
```

#### 1.4 `compute_seam_color_diff(band_a, band_b)`

```
1. BGR → Lab
2. 逐点 ΔE = sqrt((L_a-L_b)² + (a_a-a_b)² + (b_a-b_b)²)
3. 返回 mean(ΔE), median(ΔE)
```

#### 1.5 `compute_gradient_consistency(band_a_gray, band_b_gray)`

```
1. grad_a = np.diff(band_a_gray)
   grad_b = np.diff(band_b_gray)
2. sign_agree = np.sign(grad_a) == np.sign(grad_b)
3. magnitude = np.maximum(np.abs(grad_a), np.abs(grad_b))
4. total_mag = sum(magnitude)
   if total_mag < 1e-6: return None
5. return sum(sign_agree * magnitude) / total_mag
```

#### 1.6 `verify_match(vector_dir, deduped_dir, pid_a, si_a, pid_b, si_b, shift)` — 单对验证

```python
def verify_match(vector_dir, deduped_dir, pid_a, si_a, pid_b, si_b, shift):
    """
    验证一对匹配的纹理连续性。
    返回 dict: {
        'reject': bool,
        'color_diff_mean': float,
        'color_diff_median': float,
        'grad_score': float or None,
        'texture_a': float,
        'texture_b': float,
        'texture_level': str,        # 'rich' / 'low'
        'reason': str,               # 排除原因或 'ok'
        'n_samples': int,
    }
    """
    # 1. 加载数据
    side_a = load_side_data(deduped_dir, pid_a, si_a)
    side_b = load_side_data(deduped_dir, pid_b, si_b)
    color_a, mask_a = load_color_image(vector_dir, pid_a)
    color_b, mask_b = load_color_image(vector_dir, pid_b)

    if color_a is None or color_b is None:
        return {'reject': False, 'reason': 'no_color_data', ...}

    # 2. 提取条带
    band_a_colors, band_a_gray = extract_inner_band(color_a, side_a['vertices'], side_a['piece_center'], mask_a)
    # B 的 vertices 需要翻转（镜像拼接）
    band_b_colors, band_b_gray = extract_inner_band(color_b, side_b['vertices'][::-1], side_b['piece_center'], mask_b)

    # 3. 对齐（取较短的长度）
    n = min(len(band_a_colors), len(band_b_colors))
    band_a_colors = band_a_colors[:n]
    band_b_colors = band_b_colors[:n]
    band_a_gray = band_a_gray[:n]
    band_b_gray = band_b_gray[:n]

    if n < 5:
        return {'reject': False, 'reason': 'too_few_samples', ...}

    # 4. 颜色差异（始终计算）
    color_diff_mean, color_diff_median = compute_seam_color_diff(band_a_colors, band_b_colors)

    # 5. 纹理丰富度
    tex_a = compute_texture_richness(band_a_gray)
    tex_b = compute_texture_richness(band_b_gray)
    min_tex = min(tex_a, tex_b)

    # 6. 自适应判定
    if min_tex < TEXTURE_LOW_THRESHOLD:
        # 低纹理 → 只做颜色
        reject = color_diff_mean > COLOR_DIFF_REJECT_LOOSE
        return {..., 'texture_level': 'low', 'grad_score': None,
                'reason': 'color_reject' if reject else 'ok'}

    # 纹理丰富 → 颜色 + 梯度 AND
    grad_score = compute_gradient_consistency(band_a_gray, band_b_gray)
    if grad_score is None:
        reject = color_diff_mean > COLOR_DIFF_REJECT_LOOSE
        return {..., 'texture_level': 'rich', 'grad_score': None,
                'reason': 'color_reject' if reject else 'ok'}

    reject = (color_diff_mean > COLOR_DIFF_REJECT_STRICT
              and grad_score < GRAD_REJECT_THRESHOLD)
    return {..., 'texture_level': 'rich', 'grad_score': grad_score,
            'reason': 'color_gradient_reject' if reject else 'ok'}
```

---

### Step 2：新建 `pipline/run_texture_verify.py` — 独立运行脚本

**这是唯一的入口文件**，读取已有结果，运行验证，输出报告。

```python
#!/usr/bin/env python3
"""
独立的纹理连续性验证脚本。
读取已有的 connectivity.json，对每个匹配对进行纹理验证。
不修改任何现有文件。

用法：
  cd pipline
  python run_texture_verify.py

输入：
  {OUTPUT}/5_connectivity/connectivity.json   ← connect.py 的输出
  {OUTPUT}/4_deduped/side_{id}_{i}.json       ← 边几何数据
  {OUTPUT}/3_vector/color_{id}.png            ← 彩色图

输出：
  {OUTPUT}/5_connectivity/texture_verify_report.json  ← 验证报告
"""

import os, sys, json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, VECTOR_DIR
from common.texture_verify import verify_match

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
VECTOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)


def main():
    # 1. 读取 connectivity.json
    conn_path = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    with open(conn_path) as f:
        connectivity = json.load(f)

    print(f"Loaded connectivity: {len(connectivity)} pieces")

    # 2. 遍历所有匹配对
    total_matches = 0
    verified = 0
    rejected = 0
    no_color = 0
    results = {}

    for pid_str, fits_list in connectivity.items():
        pid_a = int(pid_str)
        results[pid_str] = [[], [], [], []]

        for si, matches in enumerate(fits_list):
            for m in matches:
                total_matches += 1
                pid_b = m['pid']
                sj = m['si']
                shift = (m.get('shift_x', 0), m.get('shift_y', 0))

                result = verify_match(VECTOR_PATH, DEDUPED_PATH,
                                      pid_a, si, pid_b, sj, shift)

                results[pid_str][si].append({
                    'pid': pid_b,
                    'si': sj,
                    'error': m['error'],
                    'reject': result['reject'],
                    'color_diff': round(result['color_diff_mean'], 2),
                    'grad_score': round(result['grad_score'], 3) if result['grad_score'] is not None else None,
                    'texture_level': result['texture_level'],
                    'reason': result['reason'],
                })

                if result['reason'] == 'no_color_data':
                    no_color += 1
                else:
                    verified += 1
                    if result['reject']:
                        rejected += 1

    # 3. 输出报告
    report_path = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 4. 打印统计
    print(f"\n{'=' * 60}")
    print(f"Texture Verification Report")
    print(f"{'=' * 60}")
    print(f"  Total matches:    {total_matches}")
    print(f"  Verified:         {verified}")
    print(f"  No color data:    {no_color}")
    print(f"  Rejected:         {rejected} ({rejected/max(verified,1)*100:.1f}%)")
    print(f"  Kept:             {verified - rejected}")
    print(f"  Report saved to:  {report_path}")
    print(f"{'=' * 60}")

    # 5. 打印被排除的匹配
    if rejected > 0:
        print(f"\nRejected matches:")
        for pid_str, fits_list in results.items():
            for si, matches in enumerate(fits_list):
                for m in matches:
                    if m['reject']:
                        print(f"  {pid_str}[{si}] -> {m['pid']}[{m['si']}] "
                              f"ΔE={m['color_diff']:.1f} grad={m['grad_score']} "
                              f"tex={m['texture_level']} reason={m['reason']}")


if __name__ == '__main__':
    main()
```

---

### Step 3：新建 `pipline/visualize_texture_bands.py` — 可视化调试脚本

用于人工查看接缝条带和颜色差异，帮助调整阈值。

```python
#!/usr/bin/env python3
"""
可视化纹理验证的条带和差异。
对指定的匹配对，生成接缝两侧的条带对比图。

用法：
  python visualize_texture_bands.py <pid_a> <si_a> <pid_b> <si_b>
"""

# 输出：
#   1. 两块拼图的彩色图，标注采样点和法线方向
#   2. 两条带的颜色对比图（左A右B）
#   3. ΔE 沿边的分布曲线
#   4. 梯度方向一致性图
```

---

## 文件清单（全部新建，不修改现有文件）

| 文件 | 说明 |
|------|------|
| `src/common/texture_verify.py` | 核心模块：条带提取、颜色差异、梯度一致性、验证判定 |
| `pipline/run_texture_verify.py` | 独立运行脚本：读 connectivity.json → 验证 → 输出报告 |
| `pipline/visualize_texture_bands.py` | 可视化调试：查看条带和差异，帮助调参 |

---

## 阈值参数（可配置，放在 texture_verify.py 顶部）

```python
# 采样参数
INNER_OFFSET = 6              # 向内偏移，避开边缘阴影（像素）
BAND_WIDTH = 15               # 条带宽度（像素）
N_SAMPLES = 30                # 沿边采样点数

# 纹理丰富度阈值
GRADIENT_SIGNIFICANCE_THRESHOLD = 8   # 灰度差多少算"显著变化"（0-255）
TEXTURE_LOW_THRESHOLD = 0.1           # 低于此值视为低纹理

# 颜色差异阈值（Lab ΔE）
COLOR_DIFF_REJECT_STRICT = 60.0      # 纹理丰富时（配合梯度 AND）
COLOR_DIFF_REJECT_LOOSE = 80.0       # 低纹理时（单独使用，更宽松）

# 梯度一致性阈值
GRAD_REJECT_THRESHOLD = 0.15         # 低于此值视为不匹配
```

## 后续集成（验证通过后）

验证结果确认有效后，再考虑将 `texture_verify` 集成到 `connect.py` 中作为实时过滤。届时只需：
1. `connect.py` 中 import 并调用 `verify_match()`
2. 匹配循环中增加 `if result['reject']: continue`

## 兼容性

- 不修改任何现有文件，完全独立运行
- 如果 `color_{id}.png` 不存在，跳过验证，标记为 `no_color_data`
- 输出独立的报告文件，不影响现有 connectivity.json
