# 阶段 4：实时识别与人类可读输出

## 目标

1. **在线阶段**：用户拍照手中的拼图块 → 系统识别 → 告诉用户放在哪
2. **输出生成**：生成人类可读的拼图指引

---

## 4.1 实时识别 — `src/common/realtime.py`

### 核心流程

```
用户拍照
   │
   ▼
┌──────────────┐
│ 预处理照片     │  （同阶段 A 的预处理：EXIF旋转、颜色归一化）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 分割提取拼图块  │  （同阶段 A 的分割：自适应阈值 + 连通域提取）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 向量化        │  （复用 vector.py）
└──────┬───────┘
       │
       ▼
┌───────────────────────┐
│ 在数据库中查找匹配      │
│                       │
│  策略 1：几何匹配       │──→ 找到 → 返回位置和旋转
│  （与已有块比较四条边）   │
│                       │
│  策略 2：颜色匹配       │──→ 找到 → 验证几何匹配结果
│  （与已有块比较颜色特征） │
│                       │
│  策略 3：目标图匹配     │──→ 找到 → 缩小搜索范围
│  （与目标图网格比较）    │
└──────┬────────────────┘
       │
  ┌────▼─────┐
  │ 已知块？   │
  └─┬──────┬─┘
   是│      │否
     ▼      ▼
  返回位置  新增到数据库 → 增量求解 → 返回位置
```

### 实现代码

```python
class RealtimeIdentifier:
    """
    实时拼图块识别器

    依赖预构建的 PuzzleDatabase
    """

    def __init__(self, database: PuzzleDatabase):
        self.db = database
        self.matcher = PieceMatcher(database)

    def identify(self, photo_path) -> IdentificationResult:
        """
        识别照片中的拼图块，返回识别结果

        处理流程：
        1. 预处理照片
        2. 分割提取（可能有多块）
        3. 对每块尝试匹配
        """
        preprocessed = preprocess.preprocess_phone_photo(photo_path)
        pieces = segment.extract_pieces(preprocessed)

        results = []
        for piece in pieces:
            result = self.matcher.match(piece)
            results.append(result)

        return IdentificationResult(results)


class PieceMatcher:
    """拼图块匹配器"""

    def __init__(self, database: PuzzleDatabase):
        self.db = database

    def match(self, piece_candidate) -> MatchResult:
        """
        尝试将一个拼图块候选匹配到数据库中的已知块

        返回：MatchResult
        """
        # Step 1: 向量化
        vector = Vector(piece_candidate.binary, ...)
        vector.find_border_raster()
        vector.vectorize()
        vector.find_four_corners()
        vector.extract_four_sides()
        vector.enhance_corners()

        # Step 2: 用四条边与数据库中所有块比较
        best_matches = []

        for db_piece_id, db_piece in self.db.pieces.items():
            for si in range(4):
                for sj in range(4):
                    if vector.sides[si].is_edge and db_piece.sides[sj].is_edge:
                        continue
                    error = vector.sides[si].error_when_fit_with(db_piece.sides[sj])
                    if error < MATCH_THRESHOLD:
                        best_matches.append((db_piece_id, si, sj, error))

        # Step 3: 如果几何匹配找到
        if best_matches:
            best = min(best_matches, key=lambda x: x[3])
            db_piece_id = best[0]

            if self._verify_color(piece_candidate, self.db.pieces[db_piece_id]):
                return MatchResult(
                    status='known',
                    db_piece_id=db_piece_id,
                    solution_position=self.db.get_solution_position(db_piece_id),
                    confidence=1.0 - best[3] / MATCH_THRESHOLD
                )

        # Step 4: 没有匹配到 → 新块
        return self._handle_new_piece(piece_candidate, vector)

    def _verify_color(self, candidate, db_piece):
        """颜色验证：确认几何匹配的结果在颜色上也合理"""
        if not hasattr(db_piece, 'color_features') or \
           not hasattr(candidate, 'color_features'):
            return True

        score = image_match.compute_color_similarity(
            candidate.color_features['overall_histogram'],
            db_piece.color_features['overall_histogram']
        )
        return score > COLOR_VERIFY_THRESHOLD

    def _handle_new_piece(self, candidate, vector):
        """处理新的拼图块：添加到数据库并增量求解"""
        new_id = self.db.next_available_id()
        self.db.add_piece(new_id, candidate, vector)
        self._update_connectivity(new_id)
        solution_found = self._incremental_solve(new_id)

        if solution_found:
            position = self.db.get_solution_position(new_id)
            return MatchResult(
                status='new_solved',
                db_piece_id=new_id,
                solution_position=position,
                confidence=0.8
            )
        else:
            return MatchResult(
                status='new_unsolved',
                db_piece_id=new_id,
                solution_position=None,
                confidence=0.5
            )

    def _update_connectivity(self, new_id):
        """增量更新连接图：只为新块计算匹配"""
        new_piece = self.db.pieces[new_id]

        for si, side in enumerate(new_piece.sides):
            if side.is_edge:
                continue
            for db_piece_id, db_piece in self.db.pieces.items():
                if db_piece_id == new_id:
                    continue
                for sj, other_side in enumerate(db_piece.sides):
                    if other_side.is_edge:
                        continue
                    error = side.error_when_fit_with(other_side)
                    if error <= sides.SIDE_MAX_ERROR_TO_MATCH:
                        new_piece.fits[si].append((db_piece_id, sj, error))
                        db_piece.fits[sj].append((new_id, si, error))

    def _incremental_solve(self, new_id):
        """
        增量求解：将新块插入现有解中

        策略：
        1. 如果已有完整解 → 找到新块应插入的位置
        2. 如果是部分解 → 尝试将新块加入空位
        3. 如果无解 → 尝试从新块开始局部求解
        """
        if self.db.solution is not None:
            return self._insert_into_existing_solution(new_id)
        else:
            return self._try_solve_with_new_piece(new_id)


class IdentificationResult:
    """识别结果"""
    results: list  # list[MatchResult]

    def __str__(self):
        output = []
        for r in self.results:
            if r.status == 'known':
                pos = r.solution_position
                output.append(
                    f"拼图块 #{r.db_piece_id} → "
                    f"位置 ({pos['x']}, {pos['y']}), "
                    f"旋转 {pos['rotation']}°, "
                    f"置信度 {r.confidence:.0%}"
                )
            elif r.status == 'new_solved':
                pos = r.solution_position
                output.append(
                    f"新拼图块 #{r.db_piece_id} → "
                    f"位置 ({pos['x']}, {pos['y']}), "
                    f"旋转 {pos['rotation']}° (新识别)"
                )
            else:
                output.append(
                    f"新拼图块 #{r.db_piece_id} → "
                    f"位置未知，已添加到数据库"
                )
        return '\n'.join(output)


class MatchResult:
    status: str              # 'known', 'new_solved', 'new_unsolved'
    db_piece_id: int         # 数据库中的 ID
    solution_position: dict  # {'x': int, 'y': int, 'rotation': float}
    confidence: float        # 0.0 ~ 1.0
```

---

## 4.2 人类可读输出 — `src/common/output.py`

### 输出设计原则
1. **可视化为主**：图片 > 文字
2. **编号对应**：每块拼图有唯一编号，所有输出统一使用
3. **位置明确**：在目标图上标注位置，让用户知道"放在哪"
4. **分步引导**：按顺序告诉用户"接下来放哪块"

### 4.2.1 拼图块编号对照图

```python
def generate_piece_catalog(database, output_dir):
    """
    生成 piece_catalog.html

    内容：所有拼图块的缩略图网格，每个带编号
    分组：角块、边块、内部块
    标注：是否已求解、在解中的位置
    """
    html = """<!DOCTYPE html><html><head><style>
        .grid { display: grid; grid-template-columns: repeat(auto-fill, 120px); gap: 10px; }
        .piece { border: 1px solid #ccc; padding: 5px; text-align: center; }
        .piece img { width: 100px; height: 100px; object-fit: contain; }
        .corner { border-color: red; }
        .edge { border-color: orange; }
        .inner { border-color: green; }
        .solved { background: #e8f5e9; }
        .unsolved { background: #fff3e0; }
    </style></head><body>"""

    # 角块
    html += "<h2>角块</h2><div class='grid'>"
    for pid, piece in database.pieces.items():
        if piece.is_corner:
            html += piece_card(pid, piece, database)
    html += "</div>"

    # 边块
    html += "<h2>边块</h2><div class='grid'>"
    for pid, piece in database.pieces.items():
        if piece.is_edge and not piece.is_corner:
            html += piece_card(pid, piece, database)
    html += "</div>"

    # 内部块
    html += "<h2>内部块</h2><div class='grid'>"
    for pid, piece in database.pieces.items():
        if not piece.is_edge and not piece.is_corner:
            html += piece_card(pid, piece, database)
    html += "</div>"

    html += "</body></html>"

    with open(os.path.join(output_dir, 'piece_catalog.html'), 'w') as f:
        f.write(html)
```

### 4.2.2 标注目标图

```python
def generate_annotated_target(database, output_dir):
    """
    生成 annotated_target.png

    在目标图上：
    1. 画出 W×H 网格线
    2. 每个格子标注对应的拼图块编号
    3. 已求解的位置用绿色边框
    4. 未求解的位置用红色虚线边框
    5. 缺失的位置用灰色填充
    """
    target_img = database.target.image.copy()
    h, w = target_img.shape[:2]
    cell_w = w / database.width
    cell_h = h / database.height

    for row in range(database.height):
        for col in range(database.width):
            x1 = int(col * cell_w)
            y1 = int(row * cell_h)
            x2 = int((col + 1) * cell_w)
            y2 = int((row + 1) * cell_h)

            piece_id = database.get_piece_at_position(col, row)

            if piece_id is not None:
                cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                text = f"#{piece_id}"
                cv2.putText(target_img, text, (x1+5, y1+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            else:
                cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 200), 1)

    cv2.imwrite(os.path.join(output_dir, 'annotated_target.png'), target_img)
```

### 4.2.3 解方案网格

```python
def generate_solution_grid(database, output_dir):
    """
    生成 solution_grid.txt

    格式：与原项目 Board.__repr__() 类似
    每个格子显示：编号 + 方向箭头

    方向箭头说明：
    ↑ = 该块的"上方边"（side 0）朝上（正常朝向）
    → = 该块顺时针旋转了 90°
    ↓ = 该块旋转了 180°
    ← = 该块逆时针旋转了 90°
    """
    pass
```

### 4.2.4 实时识别结果输出

```python
def generate_identification_output(result: IdentificationResult, database, output_dir):
    """
    为实时识别生成输出

    输出内容：
    1. 终端文字提示（即时）
    2. 更新标注目标图
    3. 更新 HTML 对照表
    """

    # 1. 终端输出
    print("=" * 50)
    print(result)
    print("=" * 50)

    for r in result.results:
        if r.status in ('known', 'new_solved'):
            pos = r.solution_position
            print(f"  → 放在目标图的第 {pos['y']+1} 行，第 {pos['x']+1} 列")
            print(f"  → 旋转方向：{describe_rotation(pos['rotation'])}")

    # 2. 更新标注图
    generate_annotated_target(database, output_dir)

    # 3. 更新 HTML
    generate_piece_catalog(database, output_dir)


def describe_rotation(angle_rad):
    """将弧度转换为人类可读的旋转描述"""
    degrees = int(round(angle_rad * 180 / math.pi)) % 360
    descriptions = {
        0: "不需要旋转",
        90: "顺时针旋转 90°",
        180: "旋转 180°（倒过来）",
        270: "逆时针旋转 90°",
    }
    closest = min(descriptions.keys(), key=lambda x: abs(x - degrees))
    if abs(closest - degrees) <= 15:
        return descriptions[closest]
    return f"旋转约 {degrees}°"
```

---

## 4.3 文件变更清单（本阶段）

| 文件 | 操作 | 关键改动 |
|------|------|---------|
| `src/common/realtime.py` | **新增** | 实时识别器、匹配器、增量求解 |
| `src/common/output.py` | **新增** | HTML 对照图、标注目标图、解方案网格 |
| `src/common/database.py` | **补充** | 增量添加、位置查询方法 |
