# 阶段 1：照片预处理与拼图块提取

## 目标

将用户用手机拍摄的多张拼图块照片（可能包含多块、有透视畸变、背景杂乱）处理为：
- 每块拼图的独立二值图（用于向量化）
- 每块拼图的独立彩色图（用于图像匹配）
- 每块拼图的位置元数据

---

## 1.1 照片预处理 — `src/common/preprocess.py`

### 输入
- 手机拍摄的照片（JPEG/PNG，可能竖屏/横屏，有 EXIF 信息）

### 处理步骤

#### Step 1：EXIF 自动旋转
```python
def auto_rotate(image_path):
    """
    读取 EXIF 方向标签，自动旋转到正确朝向
    不像原项目那样报错，而是自动修正
    """
    # 使用 PIL.ImageOps.exif_transpose() 自动应用 EXIF 旋转
    # 不再强制横屏
```

#### Step 2：颜色空间归一化
```python
def normalize_color(image):
    """
    归一化颜色，减少不同光照条件的影响
    """
    # 1. 白平衡校正（Gray World 算法或类似）
    # 2. 可选：直方图均衡化（CLAHE，限制对比度）
    # 3. 保存归一化后的图像
```

#### Step 3：可选透视校正
```python
def perspective_correct(image, corners=None):
    """
    如果用户拍照有明显倾斜，尝试透视校正

    策略：
    a) 自动模式：检测图片中的最大四边形轮廓（桌面/纸板边界）
       - 用 cv2.findContours + cv2.approxPolyDP 找四边形
       - 用 cv2.getPerspectiveTransform + cv2.warpPerspective 校正
    b) 半自动模式：如果自动检测失败，用户可手动指定四个角点

    注意：第一版可以跳过此步骤，只要求用户尽量正上方拍摄
    """
```

### 输出
- 预处理后的照片（统一方向、颜色归一化）
- 保存到 `1_preprocessed/` 目录

---

## 1.2 自适应背景分割 — `src/common/segment_phone.py`

### 核心挑战
原项目用固定阈值 `SEG_THRESH=145` 做二值化（假设黑色背景+白色拼图）。
手机照片背景不均匀，必须自适应。

### 方案设计（三级策略，按优先级尝试）

#### 策略 A：自适应阈值（推荐首选）
```python
def segment_adaptive(image_gray):
    """
    使用 OpenCV 自适应阈值，适应不均匀光照
    """
    # 1. 高斯模糊去噪
    blurred = cv2.GaussianBlur(image_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # 2. 自适应阈值（局部自适应）
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 根据背景明暗选择
        blockSize=51,
        C=10
    )

    # 3. 形态学操作（闭运算填补小孔，开运算去除小噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
```

#### 策略 B：Otsu 自动阈值
```python
def segment_otsu(image_gray):
    """
    全局 Otsu 阈值，适合背景与前景区分明显的情况
    """
    _, binary = cv2.threshold(
        image_gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
```

#### 策略 C：GrabCut 语义分割（备选）
```python
def segment_grabcut(image):
    """
    使用 OpenCV GrabCut 算法，对复杂背景有效
    但速度较慢，适合少量照片
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # 假设拼图块在图片中心区域
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
```

### 背景颜色自适应
```python
def detect_bg_brightness(image_gray):
    """
    自动检测背景是亮色还是暗色
    通过检测图片边缘区域（最外圈 10%）的亮度中值判断
    """
    border_pixels = get_border_pixels(image_gray, ratio=0.1)
    median_brightness = np.median(border_pixels)
    return 'light' if median_brightness > 128 else 'dark'

    # 如果背景亮色 → THRESH_BINARY_INV（拼图块为 0，背景为 1，然后取反）
    # 如果背景暗色 → THRESH_BINARY（拼图块为 1，背景为 0）
```

### 输出
- 每张照片的二值掩码图
- 同时保留原始彩色图的裁切版本
- 保存到 `2_segmented/` 目录

---

## 1.3 拼图块提取 — 改造 `src/common/extract.py` + 复用 `find_islands.py`

### 与原项目的差异

| 原项目 | 新项目 |
|--------|--------|
| 一张照片 = 一个连通域（机器人精确拍摄） | 一张照片可能含多个拼图块 |
| 只提取二值图 | 同时提取二值图 + 彩色图 |
| 固定面积阈值 `MIN_PIECE_AREA` | 自适应面积阈值（基于图片总面积的比例） |
| 碰到图片边界的区域被忽略 | 保留碰到边界的区域（但标记为"可能不完整"） |

### 改造后的提取逻辑
```python
def extract_pieces(binary_image, color_image, photo_id):
    """
    从一张预处理后的照片中提取所有拼图块

    返回：list[PieceCandidate]
    """
    # 1. 复用 find_islands.remove_stragglers() 清理噪点
    cleaned = remove_stragglers(binary_image)

    # 2. 复用 find_islands.extract_islands() 提取连通域
    # 但修改参数：
    # - min_area: 根据图像总面积动态计算（如总面积的 0.5%）
    # - ignore_border: False（不忽略边界区域，但标记）
    islands = extract_islands(cleaned, min_island_area=dynamic_min_area, ignore_border=False)

    # 3. 对每个连通域：
    pieces = []
    for island_mask, origin_row, origin_col in islands:
        # 3a. 裁切二值图
        binary_crop = crop_from_mask(island_mask)

        # 3b. 裁切彩色图（用 mask 掩码）
        color_crop = cv2.bitwise_and(
            crop_region(color_image, origin_row, origin_col),
            crop_region(color_image, origin_row, origin_col),
            mask=island_mask
        )

        # 3c. 标记是否完整（碰到图片边界 = 可能不完整）
        is_complete = not touches_border(island_mask)

        pieces.append(PieceCandidate(
            binary=binary_crop,
            color=color_crop,
            origin=(origin_col, origin_row),
            photo_id=photo_id,
            is_complete=is_complete
        ))

    return pieces
```

### 完整性标记
```python
class PieceCandidate:
    binary: np.ndarray       # 二值图
    color: np.ndarray        # 彩色图
    origin: Tuple[int, int]  # 在原图中的位置
    photo_id: str            # 来源照片
    is_complete: bool        # 是否完整（没碰到边界）

    # 如果 is_complete=False：
    # - 仍可用于后续处理，但降低匹配优先级
    # - 如果有同一块拼图的完整照片，优先使用完整版本
```

---

## 1.4 文件变更清单（本阶段）

| 文件 | 操作 | 关键改动 |
|------|------|---------|
| `src/common/preprocess.py` | **新增** | EXIF旋转、颜色归一化、可选透视校正 |
| `src/common/segment_phone.py` | **新增** | 自适应背景分割（三级策略） |
| `src/common/extract.py` | **改造** | 多块提取、保留彩色、完整性标记 |
| `src/common/find_islands.py` | **小改** | 放宽边界限制、动态面积阈值 |
| `src/common/config.py` | **改造** | 新增分割相关参数 |
