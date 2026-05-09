# puzzle-bot

拼图自动求解系统。基于手机拍摄的拼图块照片，自动识别每块拼图的形状，通过几何匹配和 DFS 搜索求解拼图的完整布局。

假设拼图块构成一个矩形网格，每块有四条边（平边、凸边或凹边）。

## 快速开始

```bash
# 运行完整的 7 步管线
python pipline/pipline_e2e_solve.py \
  -i ./photos \
  -o ./output \
  -t ./target.jpg

# 或只运行部分步骤（如 Step 1-5，不含目标图）
python pipline/pipline_e2e_solve.py \
  -i ./photos \
  -o ./output \
  -s 1 -e 5
```

## 求解管线

整个处理流程由 7 个步骤顺序执行，每步的输出作为下一步的输入：

### Step 1：分割提取

从手机拍摄的照片中分离出每块拼图。利用 Alpha 通道或阈值分割提取前景连通域，对过大区域（多块粘连）通过腐蚀拆分。输出每块的二值图（BMP）和彩色图（PNG）。

### Step 2：向量化

对每块拼图的二值图进行轮廓追踪，检测四个角点，将轮廓分为四条边并分类（平边/凸/凹）。输出每条边的顶点坐标序列（JSON）和轮廓可视化（SVG）。

### Step 3：去重

通过签名分组（边凸凹类型组合）缩小搜索范围，然后对同组块对进行几何轮廓匹配（RMSE）和纹理验证（NCC），确认重复后按图像对比度选优。

### Step 4：连通图构建

计算所有去重后拼图块之间的边匹配关系。对每条非平边搜索所有可能的匹配边，通过预过滤（长度、凸凹互补）和精确误差计算（重采样后逐点 RMSE）筛选候选，构建完整的边匹配连通图。

### Step 5：求解

从角块出发，通过螺旋式 DFS 搜索拼图块的排列组合。自动推断拼图尺寸（边链追踪），利用 NCC 增强的匹配优先排序加速搜索，支持多角块尝试和里程碑保存。

### Step 6：目标图匹配

将求解结果与拼图包装盒照片对齐。通过透视校正、方向/翻转判定将目标图校正到装配坐标系，然后逐块在目标图上搜索最佳匹配位置（平移+旋转 NCC），生成每块的匹配置信度。

### Step 7：目标引导求解

以 Step 6 中高置信度的块为锚点，利用连通性约束和目标图 NCC 验证，迭代放置低置信度和未放置的拼图块。最后通过松弛搜索（降低阈值、无连通性约束）尽可能补全。

## 输出目录结构

```
output/
├── 2_piece_bmps/          # Step 1: 拼图块二值图
├── 2_piece_colors/        # Step 1: 拼图块彩色图（带透明通道）
├── 3_vector/              # Step 2: 轮廓向量数据（side_{pid}_{0-3}.json）
├── 4_deduped/             # Step 3: 去重后的向量数据
├── 5_connectivity/        # Step 4: 边匹配连通图
├── 6_solution/            # Step 5: 求解结果（网格、装配图）
├── target_aligned.png     # Step 6: 对齐后的目标图
├── target_match_report.json
└── targeted_solve/        # Step 7: 最终求解结果
```

## 技术细节

- 详细的设计文档：[book/PuzzleSolver/](book/PuzzleSolver/)
- 原始项目技术文档（机器人版本）：[Google Docs](https://docs.google.com/document/d/1BUCGdZCe8JevGYF3pJ4ZjPqpcSgA7LF0kV6sWbHrT1Q/)
- 原始项目演示视频（Mark Rober 合作）：[YouTube](https://www.youtube.com/watch?v=Sqr-PdVYhY4)
