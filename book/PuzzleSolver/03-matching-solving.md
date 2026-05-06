# 阶段 3：边匹配、图像匹配与求解

## 目标

基于归一化后的拼图块数据，构建边连接图，尝试求解拼图。

匹配维度：
1. **边几何匹配**（复用现有代码，核心）
2. **图像颜色匹配**（新增，辅助验证）
3. **目标图匹配**（新增，全局定位）

---

## 3.1 边几何匹配 — 复用 `connect.py` + `sides.py`

### 可直接复用

- `sides.Side` 类 — 边数据模型
- `sides.Side.error_when_fit_with()` — 两条边的几何误差计算
- `sides.Side.rotated()` — 旋转对齐
- `util.resample_polyline()` — 重采样到 26 个等距点
- `util.error_between_polylines()` — polyline 误差计算

### 改造 `connect.py`

```python
def build(input_path, output_path, image_features=None):
    """
    构建连接图

    与原版差异：
    1. image_features 参数可选，用于颜色匹配增强
    2. 不再依赖 batch.json
    """
    ps = pieces.Piece.load_all(input_path, resample=True)

    with multiprocessing.Pool() as pool:
        results = [
            pool.apply_async(
                _find_potential_matches_for_piece,
                (ps, piece_id, image_features)
            )
            for piece_id in ps.keys()
        ]
        out = [r.get() for r in results]

    ps = {piece_id: piece for (piece_id, piece) in out}
    return _save(ps, output_path)


def _find_potential_matches_for_piece(ps, piece_id, image_features=None):
    """
    对每条边，找到所有可能匹配的其他边

    评分策略（综合几何 + 颜色）：
    """
    piece = ps[piece_id]

    for si, side in enumerate(piece.sides):
        if side.is_edge:
            continue

        for other_piece_id, other_piece in ps.items():
            if other_piece_id == piece_id:
                continue

            for sj, other_side in enumerate(other_piece.sides):
                if other_side.is_edge:
                    continue

                # === 几何误差（复用原有逻辑） ===
                geo_error = side.error_when_fit_with(other_side)

                if geo_error > sides.SIDE_MAX_ERROR_TO_MATCH:
                    continue

                # === 颜色连续性评分（新增） ===
                color_score = 1.0  # 默认不影响
                if image_features:
                    color_score = image_match.color_continuity_score(
                        image_features[piece_id]['side_bands'][si],
                        image_features[other_piece_id]['side_bands'][sj]
                    )

                # === 综合评分 ===
                # 几何误差权重高（主要依据），颜色作为辅助验证
                combined_error = geo_error / max(color_score, 0.1)

                if combined_error <= sides.SIDE_MAX_ERROR_TO_MATCH:
                    piece.fits[si].append((other_piece.id, sj, combined_error))

        # 排序 + 精简（复用原有逻辑）
        piece.fits[si] = sorted(piece.fits[si], key=lambda x: x[2])
        least_error = piece.fits[si][0][2]
        piece.fits[si] = [f for f in piece.fits[si] if f[2] <= least_error * WORST_MULTIPLIER]

    return (piece_id, piece)
```

---

## 3.2 图像颜色匹配 — `src/common/image_match.py`

### 3.2.1 边颜色连续性评分

```python
def color_continuity_score(band_a, band_b):
    """
    评估两条边在颜色上是否连续

    band_a: piece_a 的某条边的外侧颜色条带
    band_b: piece_b 的对应边的内侧颜色条带（翻转后）

    如果两块拼图确实相邻，那么 band_a 和 band_b 在接缝处
    的颜色应该是连续的（图案跨越两块拼图）

    返回：0.0 ~ 2.0 的分数
    - 1.0 = 完美连续
    - < 1.0 = 不太连续（惩罚）
    - > 1.0 = 非常连续（奖励）
    """
    # 方法 A：像素级 SSD
    diff = np.abs(band_a - band_b)
    ssd = np.sum(diff ** 2) / band_a.size
    score = 1.0 / (1.0 + ssd / 1000.0)

    # 方法 B：颜色直方图相似度（对拼图块内纹理更鲁棒）
    hist_a = compute_color_histogram(band_a)
    hist_b = compute_color_histogram(band_b)
    similarity = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    return 0.5 * score + 0.5 * max(0, similarity)
```

### 3.2.2 拼图块与目标图匹配

```python
def match_piece_to_target_grid(piece_features, target_cells):
    """
    将拼图块的颜色特征与目标图的所有网格位置匹配

    返回：list[(row, col, score)]，按 score 降序排列

    用途：
    1. 辅助求解：限制拼图块只能放在颜色匹配的网格位置
    2. 缩小搜索空间：减少 DFS 的分支
    """
    scores = []
    for row in range(target_cells.shape[0]):
        for col in range(target_cells.shape[1]):
            cell_features = target_cells[row][col]
            score = compute_target_match_score(piece_features, cell_features)
            scores.append((row, col, score))

    return sorted(scores, key=lambda x: -x[2])


def compute_target_match_score(piece_features, cell_features):
    """
    计算拼图块与目标图某个网格位置的匹配分数

    比较维度：
    1. 中心区域颜色相似度
    2. 整体颜色直方图相似度
    3. 边缘颜色与目标图对应位置边缘的相似度
    """
    center_sim = 1.0 / (1.0 + np.sum(
        (np.array(piece_features['center_color']) - np.array(cell_features['center_color'])) ** 2
    ) / 1000.0)

    hist_sim = cv2.compareHist(
        piece_features['overall_histogram'],
        cell_features['histogram'],
        cv2.HISTCMP_CORREL
    )

    return 0.4 * center_sim + 0.6 * max(0, hist_sim)
```

### 3.2.3 关于拼图块边缘纹理的特殊处理

```python
def extract_color_band_ignoring_edge(color_image, side_vertices, band_width=15):
    """
    提取颜色条带时，跳过拼图块边缘的强纹理区域

    问题：拼图块的边缘（凸起/凹槽）在照片中有强烈的阴影和轮廓线
    这些强特征会干扰颜色匹配

    解决方案：
    1. 从边缘向内偏移若干像素后再采样
    2. 或者用 mask 排除边缘像素
    """
    EDGE_OFFSET = 5  # 跳过边缘 5 像素

    # 沿法线方向，从 EDGE_OFFSET 处开始采样到 band_width
    # 这样只采样拼图块内部的图案，不受边缘影响
```

---

## 3.3 目标图处理 — `src/common/target.py`

```python
class TargetImage:
    """
    目标图（包装盒图片）处理

    将目标图分割为 W×H 网格，提取每个格子的颜色/纹理特征
    """

    def __init__(self, image_path, puzzle_width, puzzle_height):
        self.image = cv2.imread(image_path)
        self.width = puzzle_width
        self.height = puzzle_height
        self.cells = {}  # (row, col) → cell_features

        self._process()

    def _process(self):
        """处理目标图"""
        # 1. 检测目标图中的有效区域（可能有白边/黑边需要裁切）
        self._detect_bounds()

        # 2. 透视校正（如果包装盒照片有倾斜）

        # 3. 分割为 W×H 网格
        self._grid_segment()

        # 4. 提取每个格子的特征
        self._extract_cell_features()

    def _grid_segment(self):
        """将目标图分割为 W×H 等分网格"""
        h, w = self.image.shape[:2]
        cell_w = w / self.width
        cell_h = h / self.height

        for row in range(self.height):
            for col in range(self.width):
                x1 = int(col * cell_w)
                y1 = int(row * cell_h)
                x2 = int((col + 1) * cell_w)
                y2 = int((row + 1) * cell_h)

                self.cells[(row, col)] = {
                    'bounds': (x1, y1, x2, y2),
                    'image': self.image[y1:y2, x1:x2],
                }

    def _extract_cell_features(self):
        """提取每个格子的颜色/纹理特征"""
        for (row, col), cell in self.cells.items():
            cell_img = cell['image']

            # 中心颜色
            ch, cw = cell_img.shape[:2]
            center = cell_img[ch//4:3*ch//4, cw//4:3*cw//4]
            cell['center_color'] = np.mean(center, axis=(0,1)).tolist()

            # 颜色直方图
            hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
            cell['histogram'] = {
                'h': cv2.calcHist([hsv], [0], None, [36], [0, 180]),
                's': cv2.calcHist([hsv], [1], None, [32], [0, 256]),
                'v': cv2.calcHist([hsv], [2], None, [32], [0, 256]),
            }

            # 边缘颜色（每条边的中段）
            cell['edge_colors'] = {
                'top': np.mean(cell_img[:ch//6, cw//4:3*cw//4], axis=(0,1)).tolist(),
                'bottom': np.mean(cell_img[5*ch//6:, cw//4:3*cw//4], axis=(0,1)).tolist(),
                'left': np.mean(cell_img[ch//4:3*ch//4, :cw//6], axis=(0,1)).tolist(),
                'right': np.mean(cell_img[ch//4:3*ch//4, 5*cw//6:], axis=(0,1)).tolist(),
            }

    def get_candidate_positions(self, piece_features, top_n=5):
        """
        给定拼图块的颜色特征，返回目标图上最可能的 top_n 个位置
        """
        scores = []
        for (row, col), cell in self.cells.items():
            score = compute_target_match_score(piece_features, cell)
            scores.append((row, col, score))

        scores.sort(key=lambda x: -x[2])
        return scores[:top_n]
```

---

## 3.4 求解器改造 — 改造 `board.py`

### 原有求解器特点
- DFS + 优先队列
- 从角块开始，螺旋式填充
- 约束：边类型匹配 + 邻居连接性

### 改造点

#### 3.4.1 支持不完整拼图
```python
class Board:
    def __init__(self, width, height):
        # 原有逻辑不变
        pass

    def can_place(self, piece_id, fits, x, y, orientation):
        # 原有约束检查不变
        # 但增加：如果某个位置的邻居是空的（因为缺失的拼图块），
        # 不要求该方向有连接
        pass

    @property
    def missing_positions(self):
        """返回所有未放置的位置"""
        positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self._board[y][x] is None:
                    positions.append((x, y))
        return positions


def build(connectivity=None, input_path=None, output_path=None,
          target=None, piece_features=None):
    """
    改造后的求解器

    新增参数：
    - target: TargetImage 对象（可选）
    - piece_features: 每块的颜色特征（可选）
    """
    # ... 原有逻辑 ...

    # 新增：如果拼图块不完整，尝试部分求解
    if len(ps) < PUZZLE_WIDTH * PUZZLE_HEIGHT:
        return build_partial(ps, target, piece_features)
    else:
        return build_from_corner(ps, start_piece_id=corners[0], edge_length=edge_length)


def build_partial(ps, target=None, piece_features=None):
    """
    部分求解：拼图块不全时的策略

    策略：
    1. 仍然尝试完整求解，但允许某些位置留空
    2. 利用目标图的颜色约束来定位已知块
    3. 输出部分解（已确定的块 + 未确定的位置）
    """
    # 1. 如果有目标图，先用颜色匹配为每块确定候选位置
    if target and piece_features:
        position_constraints = {}
        for piece_id, features in piece_features.items():
            candidates = target.get_candidate_positions(features, top_n=10)
            position_constraints[piece_id] = candidates

    # 2. 尝试 DFS 求解，但有位置约束
    # 3. 如果无法完整求解，返回部分解
    pass
```

#### 3.4.2 目标图颜色约束（可选增强）
```python
def can_place(self, piece_id, fits, x, y, orientation):
    """
    原有检查 + 目标图颜色约束
    """
    # ... 原有检查 ...

    # 新增：目标图颜色约束（宽松的，作为软约束而非硬约束）
    if self.target_image and piece_id in self.piece_features:
        expected = self.target_image.cells.get((y, x))
        if expected:
            score = compute_target_match_score(
                self.piece_features[piece_id], expected
            )
            if score < MIN_TARGET_COLOR_SCORE:
                return False, f"Color mismatch with target at ({x},{y})"

    return True, None
```

---

## 3.5 文件变更清单（本阶段）

| 文件 | 操作 | 关键改动 |
|------|------|---------|
| `src/common/connect.py` | **改造** | 集成颜色匹配评分 |
| `src/common/image_match.py` | **补充** | 颜色连续性、目标图匹配函数 |
| `src/common/target.py` | **新增** | 目标图处理、网格分割、特征提取 |
| `src/common/board.py` | **改造** | 支持不完整求解、目标图约束 |
| `src/common/sides.py` | **保留** | 无需修改 |
| `src/common/pieces.py` | **保留** | 无需修改 |
