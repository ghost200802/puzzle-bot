# 移除机器人模式，只保留手机拍照模式

## 目标
从 puzzle-bot 项目中彻底移除 robot 模式相关的代码、文件、配置和定义，只保留最简洁的 phone 拍照模式代码。

---

## 第一步：删除整个文件（8个文件 + 1个目录）

| # | 文件路径 | 原因 |
|---|---------|------|
| 1 | `src/common/move.py` | 机器人运动计算，仅被 solve.py robot 分支调用 |
| 2 | `src/common/spacing.py` | 机器人松紧调整，仅被 solve.py robot 分支调用 |
| 3 | `src/common/bmp.py` | 机器人照片转 BMP，仅被 process.py robot 分支调用 |
| 4 | `src/c/find_islands.c` | 机器人 C 语言岛屿提取，仅被 extract.py batch_extract 调用 |
| 5 | `src/c/.gitignore` | C 目录的 gitignore，随 C 文件一起删除 |
| 6 | `src/scripts/segmentation_diff_test.py` | 机器人调试脚本，使用 SEGMENT_DIR 和 BMP 函数 |
| 7 | `src/scripts/dedupe_test.py` | 机器人调试脚本，使用 BMP 函数 |
| 8 | `src/scripts/count_white_black.py` | 机器人调试脚本，使用 binary_pixel_data_for_photo |
| 9 | `src/scripts/luminance_distribution.py` | 机器人遗留调试工具 |

删除后检查 `src/c/` 目录是否为空，若为空则删除该目录。
检查 `src/scripts/` 目录是否为空，若为空则删除该目录。

---

## 第二步：编辑 `src/common/config.py`

**删除以下内容：**

1. **模块文档字符串**（第1-7行）：更新为仅描述 phone 模式
2. **MODE 变量**（第12-13行）：`MODE = os.environ.get(...)` — 不再需要模式切换
3. **TIGHTEN_RELAX_PX_W / TIGHTEN_RELAX_PX_H**（第20-21行）：仅被 robot spacing.py 使用
4. **SCALE_BMP_TO_WIDTH**（第25行）：仅被 robot bmp.py 使用
5. **CROP_TOP_RIGHT_BOTTOM_LEFT**（第26行）：仅被 robot bmp.py 和 extract.py batch_extract 使用
6. **MIN_PIECE_AREA**（第27行）：仅被 robot extract.py batch_extract 使用
7. **SEG_THRESH**（第29行）：仅被 robot bmp.py 使用
8. **Robot parameters 整段**（第32-33行）：`APPROX_ROBOT_COUNTS_PER_PIXEL`
9. **PHOTO_BMP_DIR**（第43行）：机器人第1步目录
10. **SEGMENT_DIR**（第46行）：机器人第2步目录
11. **TIGHTNESS_DIR**（第61行）：机器人第7步目录
12. **条件包装**（第64-78行）：`if MODE == 'phone':` 条件判断 — 将 PHONE_* 参数直接展开为顶层常量

**保留内容：**
- `PUZZLE_WIDTH`, `PUZZLE_HEIGHT`, `PUZZLE_NUM_PIECES`
- `MAX_PIECE_DIMENSIONS`（被 vector.py 共享使用）
- `DUPLICATE_CENTROID_DELTA_PX`（被 dedupe.py 使用）
- `PHOTOS_DIR`, `VECTOR_DIR`, `DEDUPED_DIR`, `CONNECTIVITY_DIR`, `SOLUTION_DIR`
- 所有 `PHONE_*` 参数（展开为顶层常量）

---

## 第三步：编辑 `src/process.py`

**删除以下内容：**

1. **模块文档字符串中 robot 相关描述**（第4-6行）
2. **robot 目录常量导入**（第21-22行）：`PHOTO_BMP_DIR`, `SEGMENT_DIR`, `TIGHTNESS_DIR`
3. **robot 专用常量导入**（第23行）：`MIN_PIECE_AREA`, `MAX_PIECE_DIMENSIONS`, `CROP_TOP_RIGHT_BOTTOM_LEFT`
4. **robot 导入分支**（第31-32行）：`else: from common import bmp, extract, util, vector, dedupe`
5. **robot_states 参数**（第35行）：从 `batch_process_photos` 函数签名中移除
6. **robot 分发逻辑**（第50-55行）：`else: return _batch_process_robot(...)`
7. **整个 robot 模式代码块**（第194-339行）：包括 `_batch_process_robot`, `_bmp_all`, `_extract_all`, `_vectorize_all` 四个函数

**修改内容：**
- 函数 `batch_process_photos` 简化签名，移除 `robot_states` 参数
- 函数体中移除 MODE 判断，直接调用 `_batch_process_phone`
- 更新导入，移除 `MODE` 和 robot 相关常量

---

## 第四步：编辑 `src/solve.py`

**删除以下内容：**

1. **TIGHTNESS_DIR 导入**（第10行）
2. **MODE 导入**（第10行）
3. **robot 分支**（第41-51行）：`if MODE == 'robot' and start_at <= 7:` 整个代码块

**修改内容：**
- 第36行：`if puzzle is not None and MODE == 'phone':` → `if puzzle is not None:`
- output 模块的导入从条件导入改为直接导入（放在文件顶部）

---

## 第五步：编辑 `src/run_batch.py`

**删除以下内容：**

1. **模块文档字符串中 robot 描述**（第7行）
2. **MODE 导入**（第19行）
3. **robot 目录常量导入**（第20行）：`PHOTO_BMP_DIR`, `SEGMENT_DIR`, `TIGHTNESS_DIR`
4. **util 导入**（第23行）：仅在 robot 代码中使用
5. **robot 目录创建**（第29-30行）：`_prepare_new_run` 中的 robot 目录
6. **robot argparse 参数**（第71-83行）
7. **MODE 分发逻辑**（第47行、第88-91行）
8. **`_run_robot_mode()` 整个函数**（第137-161行）

**修改内容：**
- `_prepare_new_run` 中移除 robot 目录
- `main()` 中移除 MODE 判断，直接执行 phone 模式逻辑
- argparse 只保留 phone 模式参数

---

## 第六步：编辑 `src/common/extract.py`

**删除以下内容：**

1. **模块文档字符串中 robot 描述**（第4-5行）
2. **robot 专用导入**（第9-11行）：`import re`, `import subprocess`, `import pathlib`（`re` 仅被 `batch_extract` 使用）
3. **robot 专用配置导入**（第16行）：`MIN_PIECE_AREA, CROP_TOP_RIGHT_BOTTOM_LEFT, MODE`
4. **`batch_extract()` 整个函数**（第99-137行）：robot 模式 C 库提取

**保留内容：**
- `PieceCandidate` 类
- `extract_pieces_from_segmented()` 函数
- `PHONE_MIN_PIECE_AREA_RATIO` 导入

---

## 第七步：编辑 `src/common/dedupe.py`

**删除以下内容：**

1. **`deduplicate()` 函数**（第18-126行）：robot 模式去重（包含 `_photo_space_to_robot_space` 内嵌函数）
2. **`_pick_best_dupe()` 函数**（第129-150行）：仅被 robot `deduplicate()` 调用

**保留内容：**
- `_compare()` 函数（被 `deduplicate_phone` 调用）
- `compute_piece_hash()`, `find_duplicate_candidates()`, `deduplicate_phone()`
- `_copy_piece_files()`, `pick_best()`
- `DOUBLE_CHECK_GEOMETRIC_DUPLICATE_THRESHOLD`, `SIDE_MISALIGNMENT_RAD`
- `from common.config import *`（不再导入 APPROX_ROBOT_COUNTS_PER_PIXEL，但通过 `*` 导入不会报错，因为 config.py 已删除该常量）

---

## 第八步：编辑 `src/common/connect.py`

**修改内容：**
- 第9行：删除 `from common.config import MODE`（导入了但从未在条件分支中使用）

---

## 第九步：编辑 `src/common/vector.py`

**修改内容：**
- 第33行：更新注释 `# Default SCALAR for backward compatibility (robot mode pieces ~945px wide)` → 移除 robot 提及

---

## 第十步：编辑 `src/common/util.py`

**删除以下函数和常量：**

1. **`EXPECTED_PHOTO_ORIENTATION`**（第27行）：仅被 `get_photo_orientation` 使用
2. **`get_photo_orientation()`**（第48-55行）：仅被 `binary_pixel_data_for_photo` 调用
3. **`binary_pixel_data_for_photo()`**（第58-86行）：仅被 robot bmp.py 和 robot 脚本使用
4. **`threshold_pixels()`**（第89-96行）：仅被 `binary_pixel_data_for_photo` 使用
5. **`remove_stragglers()`**（第892行起）：phone 模式使用 `find_islands.py` 中的版本，util.py 版本仅被 robot 脚本引用
6. **`remove_tiny_islands()`**（第963行起）：仅被 robot 调试脚本 dedupe_test.py 使用
7. **`normalized_ssd()`**（第867行起）：仅被 robot 调试脚本使用

**额外清理：**
- 第2行：`from PIL import Image, ExifTags` → `from PIL import Image`（`ExifTags` 仅被 `get_photo_orientation` 使用）

**保留内容：**
- `load_bmp_as_binary_pixels()`：被 vector.py（共享代码）和测试使用
- 所有其他几何工具函数（distance, angle, rotate, resample 等）

---

## 第十一步：编辑测试文件

### `tests/test_milestone1.py`
- **删除** `test_robot_params_preserved()` 测试方法（第38-44行）
- **更新** `test_phone_mode_is_default()`：移除对 `MODE` 的测试（config.py 中不再有 MODE 变量）

### `tests/test_integration.py`
- **更新** 第242行注释：`"With robot photos, not all extracted regions are valid puzzle pieces."` → 移除 robot 提及

---

## 第十二步：验证

1. 运行现有测试，确保所有测试通过
2. 检查所有 Python 文件中不再有对 `robot`、`MODE`、`move`、`spacing`、`bmp.py`、`batch_extract`（extract.py 中的 robot 函数）的引用
3. 确认 `grep -r "from common import bmp"` 无结果
4. 确认 `grep -r "from common import move"` 无结果
5. 确认 `grep -r "from common import spacing"` 无结果
6. 确认 `grep -r "MODE =="` 无结果

---

## 不需要修改的文件（确认干净）

以下文件已确认无 robot 相关代码，无需修改：

- `solve_puzzle.py` — 纯 phone 模式管线
- `run_pipeline.py` — 纯 phone 模式管线
- `run_e2e.py` — 纯 phone 模式端到端管线
- `src/common/preprocess.py` — phone 预处理
- `src/common/segment_phone.py` — phone 分割
- `src/common/find_islands.py` — Python 版岛屿提取（phone 使用）
- `src/common/board.py` — 棋盘求解
- `src/common/sides.py` — 边数据处理
- `src/common/pieces.py` — 拼图块数据
- `src/common/target.py` — 目标图像匹配
- `src/common/image_match.py` — 图像匹配
- `src/common/pipeline_utils.py` — 管线工具
- `src/common/real_time.py` — 实时识别（phone 模式功能）
- `src/common/output.py` — 输出生成
- `src/common/__init__.py` — 空文件
- `tests/conftest.py`
- `tests/helpers.py`
- `tests/test_milestone2.py`
- `tests/test_milestone3.py`
- `tests/test_milestone4.py`
- `tests/test_normalization.py`
- `tests/test_corner_detection.py`
