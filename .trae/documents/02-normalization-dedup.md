# 阶段 2：归一化、去重与数据库构建

## 目标

将提取出的拼图块（可能有重复、大小不一、角度不同）归一化为统一格式，去重后构建拼图数据库。

---

## 2.1 拼图块归一化 — `src/common/preprocess.py` (续)

### 问题
从不同照片提取的同一块拼图可能有不同的：
- 尺寸（拍摄距离不同）
- 角度（拍摄角度不同）
- 颜色（光照条件不同）

### 归一化策略

#### Step 1：尺寸归一化
```python
def normalize_piece_size(binary, color, target_piece_size=500):
    """
    将拼图块缩放到统一尺寸

    target_piece_size: 归一化后拼图块的"直径"（宽或高的较大值）
    """
    # 1. 找到拼图块的 bounding box
    # 2. 计算缩放比例 = target_piece_size / max(width, height)
    # 3. 同时缩放二值图和彩色图
    # 4. 保持纵横比，用黑色/透明填充背景
```

#### Step 2：角度归一化（暂不做旋转归一化）
```
注意：不做全局旋转归一化！
原因：每块拼图可能有不同的初始旋转角度
但后续向量化会自动检测 4 个角和 4 条边
边匹配时会做旋转对齐（sides.py 的 Side.rotated()）

只需要确保：
- 拼图块是"正面朝上"的（通过轮廓面积判断，正面>背面）
- 拼图块没有严重倾斜（透视校正已在预处理中处理）
```

#### Step 3：颜色归一化
```python
def normalize_piece_color(color_image, mask):
    """
    对彩色拼图块做颜色归一化

    只处理 mask 内的像素（拼图块本身），不处理背景
    """
    # 1. 提取 mask 内的像素
    # 2. 白平衡校正（基于拼图块本身的颜色分布）
    # 3. 可选：颜色空间转换到 LAB，对 L 通道做 CLAHE
```

---

## 2.2 向量化（复用现有代码）— `src/common/vector.py`

### 可完全复用的部分
- `find_border_raster()` — 边界检测
- `vectorize()` — 轮廓追踪
- `find_four_corners()` — 角点检测
- `extract_four_sides()` — 边提取
- `enhance_corners()` — 角点增强

### 需要小改的部分

#### `Vector.process()` 中增加颜色特征提取
```python
def process(self, output_path, metadata, color_image=None, ...):
    # 原有流程不变
    self.find_border_raster()
    self.vectorize()
    self.find_four_corners()
    self.extract_four_sides()
    self.enhance_corners()

    # 新增：提取颜色特征
    if color_image is not None:
        self.color_features = extract_piece_color_features(
            color_image, self.sides, self.incenter
        )

    # 新增：提取缩略图
    if color_image is not None:
        self.thumbnail = generate_thumbnail(color_image, self.border)
```

#### `Vector.save()` 中增加颜色数据保存
```python
def save(self, output_path, metadata, color_image=None):
    # 原有保存逻辑不变（sides JSON + SVG）

    # 新增：保存颜色特征
    if hasattr(self, 'color_features'):
        metadata['color_features'] = self.color_features

    # 新增：保存彩色图
    if color_image is not None:
        color_path = pathlib.Path(output_path).joinpath(f"color_{self.id}.png")
        cv2.imwrite(str(color_path), color_image)

    # 新增：保存缩略图
    if hasattr(self, 'thumbnail'):
        thumb_path = pathlib.Path(output_path).joinpath(f"thumb_{self.id}.png")
        cv2.imwrite(str(thumb_path), self.thumbnail)
```

### 颜色特征提取 — `src/common/image_match.py`

```python
def extract_piece_color_features(color_image, sides, incenter):
    """
    提取拼图块的颜色/纹理特征

    返回一个字典，包含：
    - overall_histogram: 整体颜色直方图
    - side_bands: 每条边两侧的颜色条带
    - center_color: 中心区域的颜色
    - texture_features: LBP 或 Gabor 纹理特征
    """
    features = {}

    # 1. 整体颜色直方图（HSV 空间）
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    features['overall_histogram'] = {
        'h': cv2.calcHist([hsv], [0], mask, [36], [0, 180]),
        's': cv2.calcHist([hsv], [1], mask, [32], [0, 256]),
        'v': cv2.calcHist([hsv], [2], mask, [32], [0, 256]),
    }

    # 2. 每条边两侧的颜色条带（用于边匹配时的颜色连续性验证）
    features['side_bands'] = {}
    for i, side in enumerate(sides):
        band = extract_color_band_along_side(color_image, side.vertices, band_width=15)
        features['side_bands'][i] = band

    # 3. 中心区域颜色（用于目标图匹配）
    cx, cy = incenter
    radius = min(color_image.shape[:2]) // 6
    center_patch = color_image[cy-radius:cy+radius, cx-radius:cx+radius]
    features['center_color'] = np.mean(center_patch, axis=(0,1)).tolist()

    return features


def extract_color_band_along_side(color_image, side_vertices, band_width=15):
    """
    沿着一条边的方向，提取边两侧各 band_width 像素宽的颜色条带

    返回：numpy array of shape (n_points, band_width, 3)

    实现方式：
    1. 对 side_vertices 中的每个点
    2. 计算该点的法线方向
    3. 沿法线方向采样 band_width 个像素
    4. 返回所有采样点的颜色
    """
```

---

## 2.3 智能去重 — 重写 `src/common/dedupe.py`

### 去重策略

原项目依赖机器人坐标判断重复。新系统没有任何空间坐标信息，必须完全基于图像内容去重。

#### Step 1：粗筛 — 感知哈希（pHash）
```python
def compute_piece_hash(binary_image, color_image):
    """
    计算拼图块的感知哈希
    对缩放、旋转、轻微颜色变化不敏感

    策略：
    1. 将二值图缩放到 32x32
    2. 计算 DCT 哈希
    3. 同时对彩色图计算颜色直方图哈希
    """
    binary_hash = imagehash.phash(Image.fromarray(binary_image * 255))
    color_hash = compute_color_histogram_hash(color_image)
    return binary_hash, color_hash


def find_duplicate_candidates(pieces, hash_threshold=15):
    """
    粗筛：感知哈希距离 < threshold 的对可能是重复
    """
    candidates = []
    for i, j in itertools.combinations(pieces.keys(), 2):
        hash_dist = pieces[i].hash - pieces[j].hash
        if hash_dist < hash_threshold:
            candidates.append((i, j, hash_dist))
    return candidates
```

#### Step 2：精筛 — 几何形状匹配
```python
def verify_duplicate(piece_a, piece_b):
    """
    对疑似重复的两块拼图，用几何形状确认

    复用原项目 dedupe.py 的 _compare() 逻辑：
    - 尝试 4 种旋转排列
    - 比较四条边的几何相似度
    - 如果相似度超过阈值，确认为重复
    """
    sides_a = piece_a.sides
    sides_b = piece_b.sides
    score = _compare(sides_a, sides_b)
    return score < DUPLICATE_GEOMETRIC_THRESHOLD
```

#### Step 3：选优 — 多属性评分
```python
def pick_best_duplicate(duplicates):
    """
    如果同一块拼图出现在多张照片中，选择"最好"的版本

    评分标准（按优先级）：
    1. 完整性：完整（不碰边界）> 不完整
    2. 分辨率：像素数多的 > 像素数少的
    3. 清晰度：拉普拉斯方差（清晰度指标）高的 > 低的
    4. 颜色质量：亮度适中的 > 过暗/过亮的
    """
    scored = []
    for piece in duplicates:
        score = 0
        if piece.is_complete:
            score += 1000
        score += piece.pixel_count
        score += compute_sharpness(piece.color_image) * 100
        scored.append((score, piece))

    return max(scored, key=lambda x: x[0])[1]
```

#### Step 4：多视图融合（高级，可选）
```python
def fuse_multi_view(duplicates):
    """
    如果同一块拼图有多个视角的照片，可以融合信息：

    1. 用最清晰的照片作为主图
    2. 用其他角度的照片验证/补充轮廓（如某个角被遮挡）
    3. 用多张照片的颜色平均值做颜色归一化
    """
    # 第一版可以不做，直接选最优
    pass
```

---

## 2.4 缺失检测与报告

```python
def check_completeness(pieces, expected_count):
    """
    检查是否有缺失的拼图块

    expected_count: 用户声明的总片数（如 300）
    actual_count: 去重后的实际片数

    如果 actual_count < expected_count:
    - 报告缺失数量
    - 标记为"不完整数据库"
    - 后续求解时允许部分解
    """
    actual_count = len(pieces)
    missing = expected_count - actual_count

    if missing > 0:
        print(f"警告：预期 {expected_count} 片，实际识别 {actual_count} 片，缺失 {missing} 片")
        print("可能原因：")
        print("  1. 某些拼图块在照片中被遮挡或未拍到")
        print("  2. 某些拼图块太小被当作噪点过滤")
        print("  3. 拼图块重叠导致未被分离")
        print("建议：")
        print("  - 调整 MIN_PIECE_AREA 参数")
        print("  - 重新拍摄缺失的拼图块")
        print("  - 系统将在缺失状态下继续构建，后续可补充")

    return missing
```

---

## 2.5 数据库管理 — `src/common/database.py`

```python
class PuzzleDatabase:
    """
    管理拼图数据库的完整生命周期

    支持操作：
    - create_from_photos(): 从照片创建新数据库
    - load(): 加载已有数据库
    - add_piece(): 添加新的拼图块（实时阶段用）
    - save(): 保存数据库
    - get_piece(): 获取指定拼图块
    - find_match(): 查找匹配的拼图块
    """

    def __init__(self, puzzle_width, puzzle_height):
        self.width = puzzle_width
        self.height = puzzle_height
        self.pieces = {}           # piece_id → PieceData
        self.connectivity = None   # 连接图
        self.solution = None       # 求解结果
        self.target = None         # 目标图数据
        self.metadata = {
            'expected_count': puzzle_width * puzzle_height,
            'actual_count': 0,
            'missing_count': 0,
            'is_complete': False,
        }

    def create_from_photos(self, photo_dir, target_path=None):
        """
        阶段 A 的完整流程：
        1. 预处理所有照片
        2. 分割提取拼图块
        3. 向量化每块拼图
        4. 去重
        5. 检查完整性
        6. 构建连接图
        7. 尝试求解
        """
        pass

    def add_piece(self, photo_path):
        """
        阶段 B 的核心操作：
        1. 预处理照片
        2. 提取拼图块
        3. 在已有数据库中查找匹配
        4. 如果是新块，添加并重新求解
        """
        pass

    def save(self, path):
        """保存数据库到磁盘"""
        pass

    @classmethod
    def load(cls, path):
        """从磁盘加载数据库"""
        pass


class PieceData:
    """单个拼图块的所有数据"""
    def __init__(self):
        self.id: int
        self.binary: np.ndarray      # 归一化二值图
        self.color: np.ndarray       # 归一化彩色图
        self.thumbnail: np.ndarray   # 缩略图
        self.sides: List[Side]       # 四条边
        self.color_features: dict    # 颜色/纹理特征
        self.is_edge: bool           # 是否边缘块
        self.is_corner: bool         # 是否角块
        self.photo_source: str       # 来源照片文件名
        self.is_complete: bool       # 是否完整
        self.hash: imagehash         # 感知哈希
```

---

## 2.6 文件变更清单（本阶段）

| 文件 | 操作 | 关键改动 |
|------|------|---------|
| `src/common/preprocess.py` | **补充** | 尺寸/颜色归一化函数 |
| `src/common/vector.py` | **小改** | `process()` 和 `save()` 增加颜色特征 |
| `src/common/image_match.py` | **新增** | 颜色特征提取函数 |
| `src/common/dedupe.py` | **重写** | 基于哈希+几何的去重，多视图融合 |
| `src/common/database.py` | **新增** | 拼图数据库管理类 |
| `src/common/config.py` | **补充** | 去重、归一化相关参数 |
