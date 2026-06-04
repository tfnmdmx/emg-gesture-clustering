# 3D 手部姿态预览 — 设计文档

> ⚠️ **§2 的核心约束已被推翻。** 本文 §2 "FK 来源:自包含近似 FK,不依赖 emg2pose/plotly/torch" 的决策已不成立:当前 `emg_label/hand3d.py` 改用 **emg2pose 的 torch 正向运动学**(`HandPositionErrorCalculator.angles_to_positions`,从 `$EMG2POSE_VIS_DIR/calculate_hand_error.py` 懒加载)。因此 §4/§5 提到的符号标定常量 `FLEX_SIGN/ABD_SIGN` 与近似 FK 几何**均不存在于当前代码**。本文其余部分(§3 关键点/骨骼布局、§5.2 `plot_cluster_hands`、§5.3 `cluster.py` 接入、`draw_hand` 配色与统一视角)仍基本属实。

- 日期: 2026-05-27
- 状态: 已实现并在真实数据上验证;但 FK 实现已从"自包含 numpy 近似"改为 **emg2pose torch FK**(`hand3d.angles_to_positions`),原 §2 决策与 `FLEX_SIGN/ABD_SIGN` 标定不再适用
- 修订(同日,实测后): 因初版 3D 图区分度不足,追加三项改进:
  1) 聚类/姿态特征改为 **apex-hold**(保持窗口顶点,取代“段末40%”),让渲染的是真正定型的姿态(已实现:`features.apex_pose_feature`);
  2) `draw_hand` 拇指用红色(`thumb_color="crimson"`)、四指蓝色(`finger_color="steelblue"`)、掌线灰虚线,便于辨认拇指(已实现);
  3) montage 用 3/4 俯视视角 **`elev=25, azim=-60`**(原文写的 elev=40/azim=-90 与代码不符)、更大子图(`3.6×3.4` 英寸/格),并经 `skeleton.axis_limits` 取共享立方坐标范围。导出片段范围不变(仍为 EMG 爆发段)。
- 关联: 扩展 `2026-05-26-emg-gesture-segmentation-labeling-design.md` 的聚类预览(条形图不够直观)

## 1. 目标

聚类阶段当前用“20 个关节角度 z-score 条形图”预览每个 cluster 的质心姿态,不够直观。
改为额外渲染**每个 cluster 质心姿态的 3D 手骨架**拼图,便于用户看图辨认手势并命名。

## 2. 约束与决策(已确认)

> 本节第一条决策**已被推翻**(见顶部 banner):当前实现依赖 emg2pose torch FK,而非自包含近似 FK。下文保留原始记录。

- ~~**FK 来源:自包含近似 FK**,只用已安装的 matplotlib。不依赖 emg2pose / plotly / torch
  (本机均未安装,用户参考脚本来自其它环境)。~~ → **取代为**:`hand3d.angles_to_landmarks` 调用 emg2pose `HandPositionErrorCalculator.angles_to_positions`(torch),从 `$EMG2POSE_VIS_DIR/calculate_hand_error.py` 懒加载;输出单位 mm。
- **输出:静态 3D 拼图 PNG**(非交互、非动画)。(仍属实)
- ~~关节角度通道映射为**推断值**(见 §4),实现时用静息姿态做符号标定。~~ → 由 emg2pose FK 直接处理 20-DOF 角度;无 §4 的符号标定常量。Manus ergonomics 角度(度)由 `_to_radians` 按范围自动转弧度。
- ~~近似 FK 不保证与 emg2pose 解剖完全一致~~ → 现已是 emg2pose 解剖 FK;目标仍是“能辨认手势”。

## 3. 关键点与骨骼(沿用用户脚本,保证可替换性)

21 个关键点,顺序与用户 `visualize_hand_3d.py` 完全一致:

```
0 THUMB_TIP   1 INDEX_TIP  2 MIDDLE_TIP 3 RING_TIP  4 PINKY_TIP
5 WRIST
6 THUMB_INT   7 THUMB_DIST
8 INDEX_PROX  9 INDEX_INT  10 INDEX_DIST
11 MIDDLE_PROX 12 MIDDLE_INT 13 MIDDLE_DIST
14 RING_PROX  15 RING_INT  16 RING_DIST
17 PINKY_PROX 18 PINKY_INT 19 PINKY_DIST
20 PALM
```

骨骼连接(同脚本)：
- thumb `[5,6,7,0]`、index `[5,8,9,10,1]`、middle `[5,11,12,13,2]`、ring `[5,14,15,16,3]`、pinky `[5,17,18,19,4]`
- 掌线(虚线)：`[20,5][20,8][20,11][20,14][20,17][20,7]`

> 之所以严格对齐该布局:将来若用户提供 emg2pose 的精确 `angles_to_positions`,
> 可直接替换 `angles_to_landmarks` 而下游绘图不变。

## 4. 推断的 20-DOF 通道映射(实现时标定)

> ⚠️ **本节不再适用。** emg2pose torch FK 自带 20-DOF 角度→关节位置的映射,代码中没有此处推断的逐通道映射、符号约定或标定常量。下文仅作历史记录。

由各通道数值范围推断(单位弧度):

- ch0–3:拇指(CMC 屈/展、MCP、IP 近似)
- 食指 ch4–7、中指 ch8–11、环指 ch12–15、小指 ch16–19,每指 4 通道:
  - `[外展(abd), MCP 屈, PIP 屈, DIP 屈]`
  - 外展通道:约 −30°…−8°、std 小;屈曲通道:MCP 约 −30°…85°,PIP/DIP 约 0°…80°(仅正)。

符号约定(初值,标定时校正):屈曲角 >0 → 该段向掌心(−z)累计弯曲;外展在掌平面(绕 z)张开。

## 5. 组件设计

### 5.1 `emg_label/hand3d.py`(新)

> ⚠️ **下文"简化手骨架 FK"已不是实现方式。** 当前 `angles_to_landmarks` 不在 numpy 里手算骨链,而是调用 emg2pose torch FK(见 banner / §2)。常量与 `draw_hand` 仍属实(后者签名已变,见下)。

- 常量:`LANDMARK_NAMES`(21)、`BONE_CONNECTIONS`、`PALM_CONNECTIONS`(同 §3)。(属实)
- `angles_to_landmarks(angles20, side="left") -> np.ndarray (21,3)`(单位 mm)
  - **实现**:`HandPositionErrorCalculator(side="left").angles_to_positions(...)`;`side="right"` 对左手结果镜像 X(规避 emg2pose 右手 init 的 22-vs-21 形状 bug)。
  - 另有批量版 `angles_batch_to_landmarks(angles_batch, side="left") -> (N,21,3)`(单次 FK 调用,montage 用)。
  - ~~下列"腕在原点 / 局部 y-z 平面累计屈曲 / 绕 z 外展 / 规范骨长比例"等为原 numpy 近似设计,未实现:~~
    - 腕在原点;掌平面 = x(横向)、y(指向)、z(掌背法向)。
    - 4 指根(MCP)沿 x 排开于 `y=palm_len` 处;拇指根置于侧面。
    - 每指在局部 y-z 平面内按累计屈曲角弯曲(MCP→PIP→DIP),再绕 z 施加外展角,平移到指根。
    - 拇指用 ch0–3 近似(根方位 + 两段屈曲)。
    - 规范骨长比例(常量);`side="left"` 时沿 x 镜像。
- `draw_hand(ax, landmarks, finger_color="steelblue", thumb_color="crimson")`(签名已从 `color="C0"` 改为分色):在 matplotlib 3D 轴上画关节散点 + 骨线(拇指红、四指蓝)+ 掌线(灰虚线)。

### 5.2 `emg_label/plotting.py` 增加 `plot_cluster_hands(centroids, counts, ids, out_path, side="left")`

- 对每个 cluster 质心(20 维原始角度)整批调用 `angles_batch_to_landmarks`(单次 FK)→ 逐手 `draw_hand`。
- 网格布局(默认 4 列),每子图一只手;**统一视角 `elev=25, azim=-60` 与统一立方坐标范围**(由 `skeleton.axis_limits` 在所有手的点云上算公共范围、并 `set_box_aspect((1,1,1))`),便于横向比较。
- 子图标题:`cluster {id} (n={count})`。保存 PNG。

### 5.3 `cluster.py` 接入

- 在现有 `plot_cluster_preview(...)`(条形图 → `{group}.png`)之后,额外:
  `plot_cluster_hands(centroids, counts, ids, out/clusters/{group}_hands.png, side=...)`。
- `side` 由 group 的手推出:`group.rsplit("-", 1)[-1]`(如 `fgw-0917-left` → `left`);
  非 left/right 时回退 `left`。
- 不改其它阶段。质心仍为 `X[mask].mean(axis=0)`(原始末段保持姿态角度)。

## 6. 标定流程(实现时人工)

> ⚠️ **本节不再适用。** emg2pose FK 是解剖正确的,没有需要人工翻转的符号常量(`FLEX_SIGN/ABD_SIGN` 不存在);第 3 步无对象可改。下文仅作历史记录。

1. 取真实数据全局关节角度中位数(静息姿态),渲染单手 → 应像“中性、略放松、张开的手”。
2. 渲染 2–3 个差异大的 cluster 质心 → 屈曲/外展方向是否合理。
3. ~~若方向反了(手指向手背弯、张开方向错),翻转对应符号常量,重渲染。~~(无符号常量可翻转)
4. 目测通过即定稿。

## 7. 测试策略

合成数据单测(不依赖真实 npz):
- `angles_to_landmarks`:
  - 输出形状 `(21,3)`、无 NaN。
  - 几何合理性:某指 MCP 屈曲角增大 → 该指尖到腕的距离变小(更卷曲)或 z 更低(更靠掌心)。
  - 全零角度(伸直)→ 指尖 y 坐标显著大于指根 y(手指伸展)。
- `plot_cluster_hands`:冒烟测试,合成几个质心 → 生成非空 PNG。

## 8. 范围外

- 交互式 HTML / plotly、动画、单文件姿态序列渲染。
- 与 emg2pose 的精确解剖一致(仅保证可辨认 + 接口可替换)。
- 拇指的高精度建模(近似即可)。
