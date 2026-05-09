# 拼图求解管线 — 总体架构

## 一、管线概述

本系统将手机拍摄的拼图块照片自动求解为完整的拼图布局。整个处理流程由 7 个顺序步骤组成，通过端到端脚本 `pipline_e2e_solve.py` 串联执行，每一步的输出作为下一步的输入。

```
输入照片 ─→ Step 1 ─→ Step 2 ─→ Step 3 ─→ Step 4 ─→ Step 5 ─→ Step 6 ─→ Step 7 ─→ 最终结果
           分割提取   向量化     去重       连通图构建  求解       目标图匹配  目标引导求解
```

---

## 二、七步流程总览

| 步骤 | 名称 | 入口脚本 | 输入 | 输出 | 核心逻辑 |
|------|------|---------|------|------|---------|
| 1 | 分割提取 | `run_splitpieces.py` | 原始照片目录 | 独立拼图块 BMP + 彩色 PNG | 从照片中分离出每块拼图的二值图和彩色图 |
| 2 | 向量化 | `run_vectorize.py` | 拼图块 BMP | 轮廓 JSON + SVG | 对每块拼图提取边界轮廓、四角、四边 |
| 3 | 去重 | `run_dedup.py` | 轮廓 JSON + 彩色图 | 去重后的轮廓 JSON | 通过几何+纹理匹配去除重复拼图块 |
| 4 | 连通图 | `run_connect.py` | 去重后的轮廓 JSON | 连通图 JSON | 计算所有拼图块之间的边匹配关系 |
| 5 | 求解 | `run_solve.py` | 连通图 JSON | 解方案网格 + 装配图 | DFS 搜索拼图块的排列组合 |
| 6 | 目标图匹配 | `run_matchtarget.py` | 解方案 + 目标图 | 匹配报告 + 对齐目标图 | 将求解结果与目标图对齐并验证 |
| 7 | 目标引导求解 | `run_targetedsolve.py` | 匹配报告 + 目标图 | 最终解方案 | 利用目标图 NCC 置信度修正和补充求解 |

---

## 三、数据流与目录结构

每一步在输出根目录下创建对应的子目录：

```
output_root/
├── 0_photos/                  # 原始输入照片（Step 1 读取）
├── 2_piece_bmps/              # Step 1 输出：每块拼图的二值 BMP
├── 2_piece_colors/            # Step 1 输出：每块拼图的彩色 PNG（带透明通道）
├── 3_vector/                  # Step 2 输出：每块拼图的 4 条边 JSON + SVG
│   ├── side_{pid}_0.json      # 每条边的顶点序列、凸凹类型、长度等
│   ├── side_{pid}_1.json
│   ├── side_{pid}_2.json
│   ├── side_{pid}_3.json
│   └── {pid}_outline.svg
├── 4_deduped/                 # Step 3 输出：去重后的边 JSON（子集）
├── 5_connectivity/            # Step 4 输出：连通图
│   ├── connectivity.json      # 每块拼图每条边的候选匹配列表
│   ├── piece_edge_info.json   # 每块的边类型标记（平边/凸/凹）
│   └── connectivity_summary.json
├── 6_solution/                # Step 5 输出：求解结果
│   ├── solution_grid.txt      # 文字版解方案（编号+方向箭头）
│   ├── solution_meta.json     # 拼图尺寸 {width, height}
│   ├── assembly.png           # 装配预览图
│   └── milestone/             # 求解过程中的里程碑快照
├── target_aligned.png         # Step 6 输出：对齐后的目标图
├── target_match_report.json   # Step 6 输出：每块的 NCC 匹配报告
├── target_match_visual.png    # Step 6 输出：匹配可视化
└── targeted_solve/            # Step 7 输出：最终求解结果
    ├── puzzle_transparent.png  # 透明背景装配图
    ├── targeted_solve_report.json
    └── progress_iter*.png     # 迭代过程快照
```

---

## 四、端到端执行方式

通过 `pipline_e2e_solve.py` 统一调度：

- **输入**：`-i` 输入照片目录、`-o` 输出根目录、`-t` 目标图（可选，Steps 6-7 需要）
- **步骤控制**：`-s` 起始步骤、`-e` 结束步骤，支持从任意步骤开始/结束
- **执行方式**：每个步骤作为独立子进程运行，任一步骤失败则终止管线

---

## 五、文档索引

| 文件 | 内容 |
|------|------|
| `00-overview-architecture.md` | **本文件** — 管线总体架构 |
| `01-split-pieces.md` | Step 1：照片分割与拼图块提取 |
| `02-vectorize.md` | Step 2：拼图块轮廓向量化 |
| `03-deduplicate.md` | Step 3：重复拼图块检测与去除 |
| `04-connectivity.md` | Step 4：边匹配连通图构建 |
| `05-solve.md` | Step 5：DFS 求解 |
| `06-match-target.md` | Step 6：目标图匹配与对齐 |
| `07-targeted-solve.md` | Step 7：目标图引导的精确求解 |
