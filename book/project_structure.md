# Puzzle-Bot 项目结构

## 目录总览

```
puzzle-bot/
├── pipline/                     # 流程脚本（pipeline runners）
├── src/
│   ├── common/                  # 核心库
│   ├── check/                   # 检查/调试/可视化脚本
│   ├── scripts/                 # 辅助脚本
│   └── c/                       # C 扩展
├── tests/                       # 测试
├── book/                        # 文档
├── input/                       # 输入数据（照片）
├── output/                      # 输出数据（运行时生成）
├── .gitignore
├── README.md
└── requirements.txt
```

## 流程脚本 `pipline/`

| 文件 | 说明 |
|------|------|
| `run_vectorize.py` | 矢量化流程：BMP → SVG/JSON，多进程处理 |
| `run_dedup.py` | 去重流程：多进程几何匹配 + 直方图匹配NCC验证，输出100个唯一拼图块 |
| `run_new_puzzles.py` | 完整流程：分割 → BMP → 矢量化 → 去重 → 连通性 → 求解 |
| `run_pipeline.py` | 单图端到端流程（input/puzzles/1.png） |
| `run_e2e.py` | 端到端流程（使用 example_data） |
| `solve_puzzle.py` | 单图求解：分割 → 矢量化 → 去重 → 连通性 → 求解 |

**运行方式**：从项目根目录执行 `python -m pipline.run_xxx` 或 `python pipline/run_xxx.py`

## 核心库 `src/common/`

| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置：目录名、拼图尺寸、分割参数、去重参数、手机模式参数 |
| `vector.py` | 矢量化：BMP → 轮廓提取 → 角点检测 → 边分割 → SVG/JSON 输出 |
| `dedupe.py` | 去重：感知哈希 + 几何比较 |
| `connect.py` | 连通性构建：边匹配 → 连接图 |
| `board.py` | 棋盘求解：连接图 → 拼图解 |
| `output.py` | 输出生成：解 → 可视化图片 |
| `sides.py` | 边处理：轮廓采样、边类型分类（凸/凹/平） |
| `pieces.py` | 拼图块数据结构 |
| `find_islands.py` | 连通域检测：分割二值图中的独立块 |
| `extract.py` | 块提取：从分割图中提取单个拼图块 |
| `segment_phone.py` | 手机照片分割：自适应阈值 + 形态学处理 |
| `preprocess.py` | 预处理：照片裁剪、缩放、灰度转换 |
| `image_match.py` | 图像匹配：NCC、RMSE、仿射变换 |
| `bmp.py` | BMP 文件读写工具 |
| `spacing.py` | 间距调整 |
| `target.py` | 目标检测 |
| `move.py` | 移动指令 |
| `real_time.py` | 实时模式 |
| `pipeline_utils.py` | 流程工具函数 |
| `database.py` | 数据库操作 |
| `util.py` | 通用工具函数 |

## 检查/调试脚本 `src/check/`

| 文件 | 说明 |
|------|------|
| `check_squareness.py` | 方正度检查：验证矢量化后的拼图块角度是否接近90°，输出报告到 `__check/` |
| `check_segmentation.py` | 分割检查：从输入图片运行分割并输出 BMP |
| `check_bmps.py` | BMP 检查：读取并打印 BMP 文件信息 |
| `check_corners.py` | 角点检查：解析 SVG 验证角点位置 |
| `show_dup_groups.py` | 重复组可视化：展示去重后的重复组及其匹配信息 |
| `show_sig_groups.py` | 签名组可视化：按边签名分组展示去重后的拼图块 |
| `debug_ncc.py` | NCC 调试：对比两块拼图的匹配过程，含直方图匹配、差异图 |
| `analyze_ncc.py` | NCC 分析：统计去重元数据中的 NCC 分布 |
| `visualize_problematic.py` | 问题块可视化：标记方正度不合格的拼图块 |

## 辅助脚本 `src/scripts/`

| 文件 | 说明 |
|------|------|
| `count_white_black.py` | 统计 BMP 黑白像素比例 |
| `dedupe_test.py` | 去重测试脚本 |
| `luminance_distribution.py` | 亮度分布分析 |
| `segmentation_diff_test.py` | 分割差异测试 |

## C 扩展 `src/c/`

| 文件 | 说明 |
|------|------|
| `find_islands.c` | 连通域检测的 C 实现（性能优化） |

## 测试 `tests/`

| 文件 | 说明 |
|------|------|
| `test_corner_detection.py` | 角点检测测试 |
| `test_normalization.py` | 归一化测试 |
| `test_integration.py` | 集成测试 |
| `test_milestone1.py` ~ `test_milestone4.py` | 里程碑测试 |
| `conftest.py` | pytest 配置 |
| `helpers.py` | 测试辅助函数 |

## 数据目录（运行时生成）

```
output/puzzle_new/
├── 2_piece_bmps/       # 分割后的单块 BMP
├── 2_piece_colors/     # 分割后的单块彩色 PNG
├── 3_vector/           # 矢量化结果（SVG + JSON）
├── 4_deduped/          # 去重后的唯一块（SVG + JSON）
├── 5_connectivity/     # 连通性图
├── 6_solution/         # 求解结果
└── __check/            # 检查报告和可视化
    ├── squareness_report.txt
    ├── flagged_pieces_grid.png
    ├── dedup_match_meta.json
    ├── dup_groups_visual.png
    └── sig_groups_visual.png
```

## 流程依赖关系

```
输入照片 → segment_phone → find_islands → extract
                                         ↓
                                    2_piece_bmps/
                                         ↓
                                     vectorize
                                         ↓
                                    3_vector/ (SVG + JSON)
                                         ↓
                                      dedup
                                         ↓
                                    4_deduped/ (100 unique pieces)
                                         ↓
                                     connect
                                         ↓
                                    5_connectivity/
                                         ↓
                                      board
                                         ↓
                                    6_solution/
                                         ↓
                                      output
                                         ↓
                                    最终拼图图像
```

## 跨模块引用关系

```
pipline/run_vectorize.py ──→ src/check/check_squareness.py
pipline/run_dedup.py    ──→ src/check/show_dup_groups.py
                        ──→ src/check/show_sig_groups.py

src/check/show_dup_groups.py ──→ pipline/run_dedup.py (复用 load_pieces, classify_side)
src/check/show_sig_groups.py ──→ pipline/run_dedup.py (复用 load_pieces, get_piece_signature)
```
