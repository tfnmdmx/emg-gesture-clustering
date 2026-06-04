# 方法说明 — EMG 手势切分、聚类与真值集构建

本文档解释切分/聚类/真值集构建的实现原理。当前流水线是**统一检测器 + 双产物**：

- **统一动作检测器**（live）：`emg_label/action_segmentation.segment_recording` 以 pose-speed 迟滞状态机为主干（spine）画出每次运动起点与手定型处，EMG 包络作为仲裁器（R2）拆分过长合并段；一个 action = 一段 motion run + 紧随其后的**真实 hold**，右边界落在下一个 onset，段与段**无缝拼接、不加 pad**。`segment.py` 对每条录制都调用它。
- 该检测器把结果同时写成两份 CSV：`segments.csv`（EMG burst 路，交叉引用 + 聚类特征缓存）与 `clips.csv`（每个 action = motion+hold 段，作为打标单元；`static_in_*` 零宽、`static_out_*` 即真实 hold）。同时输出 `recordings.csv` 含每条录制的 QC 指标。
- **聚类**只用 hold 内的静态 apex 姿态向量（`cluster.py` 是 live 聚类器）。

> 说明（已被取代）：早期设计把 pose 通路写成「关节速度 + 鲁棒阈值 → static_hold / transition_motion → 组装为独立的 `static→motion→static` clip（含 pre/post 静态裁剪与 pad）」，对应 `pose_segmentation.static_motion_intervals` / `build_smc_clips`。这两个函数**已不在 live 路径上**，仅供 `diag_seg.py` 做新旧对比，详见 §3。

代码：
[emg_label/action_segmentation.py](../emg_label/action_segmentation.py)（**live 统一检测器**） ·
[emg_label/segmentation.py](../emg_label/segmentation.py) ·
[emg_label/pose_segmentation.py](../emg_label/pose_segmentation.py)（pose_speed/robust_threshold 仍被 live 复用；static_motion_intervals/build_smc_clips/velocity_peak_segments 为 legacy/诊断用） ·
[emg_label/features.py](../emg_label/features.py) ·
[emg_label/qc.py](../emg_label/qc.py) ·
[emg_label/io_utils.py](../emg_label/io_utils.py) ·
[emg_label/skeleton.py](../emg_label/skeleton.py) ·
[segment.py](../segment.py) ·
[cluster.py](../cluster.py) ·
[export_clips.py](../export_clips.py)

---

## 0. 输入数据形态

支持**两种 npz 布局**，`io_utils.load_npz` 自动识别：

### 0.1 处理后格式（旧）
| 字段           | shape   | 含义                                            |
|----------------|---------|-------------------------------------------------|
| `emg`          | (T, 16) | 16 通道 EMG，已滤波，2000 Hz                    |
| `joint_angles` | (T, 20) | 20 维关节角，弧度，**已对齐到 EMG 时间轴**      |

`emg` 与 `joint_angles` 必须等长，否则 raise（契约保护）。

### 0.2 原始格式（reference 用的）
| 字段                                | shape       | 说明                                |
|-------------------------------------|-------------|-------------------------------------|
| `emg`                               | (T_e, 16)   | 同上                                |
| `manus_<hand>_ergonomics`           | (T_p, 20)   | 关节角，pose 时间轴（与 EMG 不同长）|
| `manus_<hand>_skeleton`             | (T_p, 25, 7)| 25 关节 XYZ + 四元数（渲染用，可选）|
| `manus_<hand>_timestamps_raw`       | (T_b,)      | 块级时间戳（稀疏，T_b ≪ T_p）       |
| `record_t0` / `record_t1`           | scalar      | 录制起止时间                        |

加载流程：
1. 找 `manus_<hand>_ergonomics`（hand 由 `--meta` 或父目录名 `{date}-{hand}` 推出）；
2. 用 `manus_*_timestamps_raw` + `record_t0/t1` 重建 `pose_t` 与 `emg_t`；
3. 把 `pose_t` 按 `pose_t[-1] / emg_t[-1]` 缩放（吸收模态间时钟漂移）；
4. 对 20 列分别 `np.interp(emg_t, pose_t, ja[:, d])` → 返回 (T_e, 16)+(T_e, 20)，下游零改动。
5. **单位归一到弧度**：Manus ergonomics 存的是**度**，`load_npz` 在加载时若检测到取值明显超过 `2π` 就整体乘 `π/180` 转弧度，使两种布局统一为弧度。**因此 `pose_speed`、`move_enter/move_exit` 等阈值的单位都是 rad/s（不是 deg/s）**，下游（apex 特征、emg2pose 蒙皮）一律按弧度处理。

骨骼数据通过 `io_utils.load_skeleton(path, hand)` 单独加载，同样对齐到 EMG 轴；用于 `export_clips.py` 的关键帧渲染（无 torch 依赖）。

`fs=2000` 是 EMG 采样率，约定为整条流水线的统一时间网格。

---

## A. 本项目与 reference 的关系

`reference/gesture_velocity_segmentation.py`（723 行）**只做切分与可视化，不做聚类**——通读全文无任何 `KMeans / sklearn / cluster` 代码（grep 计数为 0）。reference 的产出是带 `segment_type ∈ {static_hold, transition_motion, static_motion_static}` 的段落 DataFrame，给下游"做什么"是开放的。（原配套的 `gesture_velocity_segmentation.ipynb` 已删除；保留的 notebook 仅 `render_segmentation_30s.ipynb`。）

本项目**从 reference 借用**：

- pose-speed 信号（§3.1）与鲁棒阈值 `robust_threshold`（§3.2）——live 检测器仍直接复用这两个函数
- 6 帧关键帧渲染思路（§8）
- 数据源约定：默认只用 `/mnt/pose_data/`，跳过 `/mnt/force_data/`（除非 `--allow-force`）

本项目**额外有**：

- EMG burst 切分（§1, §2）—— reference 不切 EMG
- 双信号交叉引用（§4）—— 双通路独有
- 录制级 QC：EMG-pose lag、NaN 占比、`recordings.csv` 持久化（§5）
- KMeans 聚类（§7）+ 按被试归一化（§7.2）
- 结构化 CSV 输出 `segments.csv / clips.csv / recordings.csv`（§9）

**算法差异点**单独说明：

- pose-speed 公式等价但实现细节不同（§3.1 末）
- EMG 包络公式与 reference 完全一致（§1）

---

## 1. EMG 包络（baseline-centered RMS）

[`segmentation.py:7 emg_envelope`](../emg_label/segmentation.py#L7)：

```python
baseline = np.nanmedian(emg, axis=0, keepdims=True)   # 16 路各自的中位基线
centered = emg - baseline
rms      = np.sqrt(np.nanmean(centered**2, axis=1))   # 跨通道 RMS
env      = uniform_filter1d(rms, size=round(0.150*fs))
```

三步：

1. **去基线**：减每通道时间维中位数。原始 EMG 围绕 0 振荡，但通道间常有几 mV～几十 mV 的直流偏置（电极阻抗、放大器漂移）。直接整流会把偏置当成"始终在用力"。
2. **跨通道 RMS**：`sqrt(mean(centered², axis=1))`。RMS 测量瞬时**激活方差**——肌电的物理量；同时对单通道尖峰不敏感（被求和均值平滑）。这是 reference 同款公式。
3. **平滑**：150 ms uniform filter。压随机放电脉冲串的高频毛刺，保留动作整体起伏。

关键性质：肌肉一放松，env 就回基线；不依赖手的姿态是否回中性，**连续不同手势之间只要有一瞬放松就能切开**——这是 EMG 包络比"关节角到中性距离"切分更优的原因（关节距离在连续手势间一直很高，会把多个动作粘成一段）。

---

## 2. EMG burst 切分 — Otsu + 迟滞 + 鲁棒兜底

目标：在 env 上找出每次"动作"的 `[start, end)`。

### 2.1 自动阈值

[`segmentation.auto_thresholds`](../emg_label/segmentation.py)：

```python
t           = otsu(env)                                   # 双峰间的最佳分割谷
rest_center = median(env[env <= t])                       # 静息模式中心
span        = t - rest_center
enter       = rest_center + enter_k * span                # enter_k 默认 0.8
exit        = rest_center + exit_k  * span                # exit_k  默认 0.4
```

env 直方图天然双峰（一大堆静息低值 + 一小撮动作高值），Otsu 最大化类间方差自动定谷。进入/退出阈按 `enter_k / exit_k` 落在 `rest_center → 谷` 的连线上（0 = 静息中心，1.0 = 谷）。**`enter_k=0.8 < 1.0` 把进入阈压到谷以下**，以抓住"少数高 burst 把谷抬高、漏掉的低幅 burst"；`enter_k=1.0, exit_k=0.5` 复现旧的"在谷处进入、半程退出"行为。对应 CLI `--emg-enter-k`（默认 0.8）/ `--emg-exit-k`（默认 0.4）。

**鲁棒兜底**：当 Otsu 退化（`enter ≤ exit` 或 `span ≤ 0`，近常值信号或单峰），自动退到 MAD：

```python
enter = median + 1.2 * 1.4826 * MAD
exit  = median + 0.5 * 1.4826 * MAD
```

这条 fallback 杜绝"整文件被静默返 0 段"——真值集场景里不能有这种沉默丢弃。

### 2.2 迟滞扫描

[`segmentation.hysteresis_segments`](../emg_label/segmentation.py)：

```
非动作中 & env ≥ enter → 开启一段，记 start
动作中 & env <  exit  → 关闭，收 (start, i)
```

两个阈值（而非一个）的理由：单阈值会让阈值附近的抖动反复跨越，把一个动作切成几段。enter > exit 之间是"缓冲区"，抖动不触发状态切换。与施密特触发器同原理。

### 2.3 后处理

[`segmentation.filter_segments`](../emg_label/segmentation.py)：

- **合并**：相邻段间隔 < `min_rest_gap_s`（默认 0.2 s）→ 视作动作中途的瞬时下凹，合并。
- **丢弃**：段长 < `min_action_s`（默认 0.4 s）→ 视作噪声毛刺，删掉。

> 实现顺序与 reference 一致：先合并相邻段，再按 `min_action_s` 丢弃过短的幸存者。

### 2.4 hold 窗口（聚类要用）

[`segmentation.hold_windows`](../emg_label/segmentation.py)：每个 burst 之后的低肌电"持姿期"边界——`hold_end = next_burst_start`（末段为 `min(n, end + tail_samples)`，segment.py 传入 `tail_samples = fs`，即 1 s）。写到 `segments.csv` 的 `hold_end_sample` 列，供聚类阶段直接取，不用下游再重算。

每段 apex（`apex_sample` 列）= `[start, hold_end)` 内最大关节偏离帧；既是聚类特征锚点，也供 overview 图标红线。

---

## 3. 统一动作检测器（pose-speed 主干 + EMG 仲裁）— live 切分

EMG burst 切的是"用力的瞬间"，但**人能识别的稳定姿态在 hold 期**。当前采集协议是 `move → hold → move → hold` **无中性回归**：每个 hold 就是刚摆好的手势姿态本身，相邻两个 hold 是**不同的非中性姿态**。所以 hold 不能用"接近中性"来检测——它表现为 **pose-speed 低且 EMG 低**（关节停了且肌肉放松了）。

live 检测器 [`action_segmentation.segment_recording`](../emg_label/action_segmentation.py) 用**非对称融合**：

- **pose-speed 迟滞状态机 = 主干（spine）**：画出每次运动起点与手定型处，能抓住 EMG 漏掉的轻柔/缓慢手势；一个自然手势内部的短暂速度下凹不会让 run 结束（dip-merge），消除了"速度峰"切分器 2 倍过切。
- **EMG 包络 = 仲裁器（arbiter）**：相邻两个非中性 hold 之间手几乎不动（只换发力肌肉），pose-speed 没有谷；但肌肉确实在两次发力间放松。EMG 据此**重切过长的合并 run（R2）**，并（可选）否决"背靠背 hold"欠切（R7，默认关闭）。

**一个 action = 一段 motion run + 紧随其后的真实 hold**，右边界落在下一个 onset 处，所有段**无缝拼接、无人工 pad、不裁 hold**。

> 注意：本节描述的是 live 检测器。早期 reference 风格的 `static_motion_intervals` / `build_smc_clips`（独立组装含 pad 的 `static→motion→static` clip）以及 `velocity_peak_segments`（速度峰切分）**均已退役**，仅供 `diag_seg.py` 做新旧对比，见 §3.5。但 live 检测器仍**直接复用** `pose_segmentation.pose_speed`（§3.1）与 `robust_threshold`（§3.2）。

### 3.1 pose-speed 信号（live 复用）

把 20 维关节角变化压成一条标量"运动强度"曲线，类似 EMG 包络但测量的是手在不在动而非手在不在用力。

代码 [`pose_segmentation.pose_speed`](../emg_label/pose_segmentation.py) 五步：

```
joint_angles (T, 20)
        │
        ▼  np.diff(axis=0)               ① 相邻帧关节角差，shape = (T-1, 20)
        │
        ▼  np.linalg.norm(axis=1)        ② 20 维差向量的长度，shape = (T-1,)
        │                                  "20 个关节这一步整体动了多远"
        ▼  * fs                          ③ 换算每秒变化率（rad/s）
        │
        ▼  concat([delta[0]], delta)     ④ 首帧补 delta[0]，长度恢复 T
        │
        ▼  uniform_filter1d(size=0.25s)  ⑤ 平滑抹高频抖动，留下动作整体起伏
        │
        ▼
pose_speed (T,)
```

- **L2 范数取代 maxima**：用 `‖d ja‖` 而不是单关节最大变化，是因为很多手势靠多个关节联动（捏指 = 拇指+食指同动）；范数把"20 维空间里走了多远"汇成一个数。
- **`* fs` 等价于 `/ dt`**：单位是 rad/s，跨录制可比较。
- **250 ms 平滑窗口**：reference 的默认值，足够压住手指抖动、又不会糊掉 0.2 s 量级的快速动作。

**Manus 偶尔遮挡丢帧 → NaN**：live 检测器在调用 `pose_speed` 之前先用 `action_segmentation.interpolate_short_nan_gaps`（默认 `--nan-max-gap-s 0.20`）逐通道**线性填补短于该阈值的内部 NaN 段**——否则一帧 NaN 经 250 ms 平滑会糊成 ~250 ms 的 NaN-speed 空洞、被读成运动而把一个 hold 切成两段。头/尾 NaN 段（无两侧锚点）与超长 NaN 段保留为 NaN：因 `NaN <= threshold == False` 落入 motion 候选（安全默认：未知段不晋升为静态），孤立噪声块再由 `min_motion_s` 滤掉。`np.errstate(invalid='ignore')` 静音 `np.diff` 的 NaN warning。

#### 与 reference 的实现差异

| 维度       | reference                              | 当前实现                       |
|------------|----------------------------------------|--------------------------------|
| 分母 dt    | `np.diff(pose_t)` 真实时间差，坏值补 median | 固定 `1/fs`（假定等间隔）      |
| NaN        | `nan_to_num → 0`（NaN 区变 static 候选）| 保留 NaN（变 motion 候选）     |
| 平滑函数   | `np.convolve('same')`                  | `scipy.uniform_filter1d('nearest')` |
| 窗口长度   | `0.25s / median_dt` 个样本             | `round(0.25s * fs)` 个样本     |

**为什么差异不影响结果**：

- **dt 来源**：`io_utils.load_npz` 已经把 pose 重采样到等间隔 EMG 轴，所以 `* fs` 严格等价于 reference 在重采样后的 `/diff(pose_t)`。
- **NaN 哲学**：reference 假设"未知 → 静止"，我们假设"未知 → 运动"。我们选后者是因为真值集场景里**漏切真手势比误切伪段更糟**。两者在数值上都会被 `min_motion_s` / `min_static_s` 滤掉孤立噪声块，差异仅出现在长 NaN 区段——这种录制本就该被 `pose_nan_frac` QC 列剔除（§5.2）。
- **卷积 vs uniform_filter1d**：窗口相同、边界处理细节略不同，结果可视化级一致。

### 3.2 鲁棒阈值（自动 move 阈的来源）

[`pose_segmentation.robust_threshold`](../emg_label/pose_segmentation.py)：

```
threshold = max( P35(speed), median(speed) + 1.5 * 1.4826 * MAD(speed) )
```

不假设双峰，对各种 duty-cycle 都稳健。可调参数：`pose_pct`（默认 35，CLI `--pose-pct`）、`pose_mad`（默认 1.5，CLI `--pose-mad`）。

**这是 live 检测器自动 move 阈的来源**（`action_segmentation.resolve_move_thresholds`，默认 `auto_move_thresh=True`）：因为 pose-speed 的尺度在不同录制间能差 ~4 倍（jm-0503 ~2.8 vs ~10.4 rad/s），固定的 rad/s 阈值无法通用。所以**每条录制**用 `robust_threshold(pose_speed, pose_pct, pose_mad)` 求出谷点作为 `move_enter`，`move_exit = move_exit_frac(默认 0.5) * move_enter`。仅当 `--no-auto-move-thresh` 时才退回固定的 `--move-enter`/`--move-exit`（默认 `radians(60)`/`radians(25)` rad/s ≈ 60/25 deg/s，作为 ablation/近静态兜底）。

### 3.3 pose 迟滞状态机（主干）

[`action_segmentation.pose_hysteresis`](../emg_label/action_segmentation.py)：在 pose-speed 上跑 `STATIC ↔ MOVING` 两态机，返回 `(motion_runs, holds)`（holds = motion_runs 的补集）。

- `STATIC → MOVING`：`spd ≥ move_enter`。
- `MOVING → STATIC`：`spd` 落到 `max(move_exit, settle_frac * peak_run)`（且上限钳到 `0.9*move_enter`）以下并持续 ≥ `min_static_s`（默认 0.35 s）才算定型。`settle_frac`（默认 0.25，CLI `--settle-frac`）的**相对 settle 项**能关掉一个卡在 `move_exit..move_enter` 死区的"紧张微抖 hold"，单一绝对阈值会让它永远开着（欠切）。
- **dip-merge**：一个短到不满足 `min_static_s` 驻留的速度下凹不会让 run 结束——消除速度峰切分器的过切。
- `min_motion_s`（默认 0.20 s）：太短的 motion run 直接丢弃。
- NaN speed 既不触发 motion onset，也不计入 settled。

### 3.4 组装 action（motion run + 真实 hold，无缝拼接）

`segment_recording` 把检测结果组装成按时间排序的 action 列表，每个 action 字段为 `start/end/motion_start/motion_end/hold_start/hold_end/apex/fusion_type/review_flag`。流程：

1. **R1 onset**：每个通过绝对偏移闸门 `seg_range_ok`（`pose_min_range`，默认 `radians(15)`）的 motion run 起点入 onset 列表。
2. **R2 拆分**（`--enable-r2`，默认 ON）：对长于 `pose_long_seg_s`（默认 2.5 s）的"永不 settle"run，在其内部干净的 EMG-rest 间隙（`clean_rest_gaps`）处加切点；若无干净间隙则退回 `_valley_split`（在超 `move_enter` 的峰之间的速度谷处切）。
3. **R7 否决**（`--enable-r7`，**默认 OFF**）：正常长度的 hold 里若含一个干净 EMG burst（两侧都有 EMG-rest）则注入一刀。在常规数据上会过切（~125 ms pose 平滑群延迟把 EMG burst 落进前一个 hold），故默认关闭。
4. **snap + 去重 + 排序**：所有 onset 用 `snap_to_motion_foot` 拉回上升沿脚部，去重排序——共享边界使段间**无缝拼接**。
5. 对每对相邻 onset `(s, e)`：`motion_end = me`（该区间内第一个 pose-settle），`hold = [me, e)`，`start = s`，`end = e`（即下一个 onset）。

每个 action 由这些**绝对 sample 锚点**导出（**注意**：与旧 SMC 不同，没有 `pre/post_static_s` 裁剪、没有 `pad_s` 缓冲、没有 hold 上限）：

```
start == motion_start == onset
motion_end  = 该区间内第一个 pose-settle
hold_start == motion_end
hold_end == end == 下一个 onset (末段为 n)
```

写到 `clips.csv` 时：`static_in_*` 设为**零宽**（`= start`，因为前一手势的 hold 已是它自己的段，不再回补 pre-motion 静态期），`static_out_*` **直接等于真实 hold**（并由新增的 `hold_start/hold_end` 列显式命名）。

**apex** = `features.apex_index` 在 **hold 窗口内**取的最大偏离帧（held 姿态即手势本身；若在整段 `[start,end)` 上取，max-deviation 会落到 transition 而非 hold）。hold 退化（连续 run，`hold_end ≈ hold_start`）时回退到 `[start, hold_end)` 或直接取 `start`。

**review_flag**（写入 `clips.csv` / `segments.csv` 邻接）取值：`long`（motion 超 `pose_long_seg_s`，连续手势）、`nohold`（无真实 hold）、`slow`（轻柔手势、rest 级 EMG，仍保留）、`long_static`（hold 超 `max_hold_s`，默认 4 s）、`''`（正常）。**fusion_type** = `both`（motion 区间内有 EMG burst 重叠）或 `pose_only`。

### 3.5 burst ↔ clip 交叉引用

EMG burst（`segment_emg` 的产出，写入 `segments.csv`）与 action/clip（统一检测器的产出，写入 `clips.csv`）各自独立产生。把两套段落按**时间重叠最多**做配对，写到两份 CSV 里：

- `segments.csv.matched_clip_id` — 该 burst 对应哪个 clip（`-1` = 无匹配）
- `clips.csv.matched_emg_seg_idx` — 该 clip 对应哪个 burst（`-1` = EMG 没切出来）

代码 [`pose_segmentation.match_bursts_to_clips` / `match_clips_to_bursts`](../emg_label/pose_segmentation.py)。

> 退役对比器：`diag_seg.py` 仍用 `static_motion_intervals` / `build_smc_clips`（旧 SMC，带 `pre/post_static_s`/`pad_s` 裁剪与 `merge_gap_s` 合并）和 `velocity_peak_segments`（速度峰切分，`pose_prom_k`/`pose_bound_frac`/`pose_min_gesture_s`/`pose_peak_merge_gap_s`）做新旧切分的过切对比。这些参数仅作为 `Config` 的 legacy/诊断字段保留，**不再是 `segment.py` 的 CLI flag**，live 路径也不读它们。

**为什么要做这件事**：两套对不上的位置就是有问题的位置，三档解读：

| 情况                                         | 含义                                                         | 处理                                     |
|----------------------------------------------|--------------------------------------------------------------|------------------------------------------|
| 互相匹配                                     | 高置信度手势：两个独立证据都确认                             | 直接进真值集                             |
| `matched_clip_id == -1`（EMG 有、pose 没）   | 等长发力（手没动只在使劲）？或 EMG 噪声伪段？                | 人工复核                                 |
| `matched_emg_seg_idx == -1`（pose 有、EMG 没）| 缓慢轻柔手势 EMG 没明显波动？或 EMG 阈值定太高？             | **优先复核**——往往是 EMG 单方案的真漏切 |

`recordings.csv` 的 `n_burst_only` / `n_clip_only` 列汇总每条录制的分歧数量；轻柔手势多的录制 clip-only 通常远多于 burst-only，说明纯 EMG 会大量漏切，而 pose 主干能补回——这正是统一检测器以 pose-speed 为主干的原因。

### 3.6 EMG 包络 vs pose-speed 对比

两者各自看到了手势的不同侧面，组合用比任何一种单跑都靠谱：

| 维度                         | EMG 包络                                       | pose-speed                              |
|------------------------------|------------------------------------------------|-----------------------------------------|
| 公式                         | `sqrt(mean((emg - 中位基线)², axis=1))` + 平滑 | `‖d ja / dt‖` + 平滑                    |
| 测量的物理量                 | 肌肉激活方差（用力大小）                       | 关节角变化率（手在动多快）              |
| 静息时                       | 回基线                                         | 接近 0                                  |
| **持姿时（用力维持姿势）**   | **仍然高**（肌肉在维持）                       | **0**（关节没动）                       |
| **等长发力**（按力传感器）   | **高**                                         | **0**（手没动）                         |
| **缓慢被动张开**（轻柔手势） | 弱信号，可能漏切                               | **能检测到**                            |
| 阈值方法                     | Otsu（双峰）+ MAD fallback                     | `max(P35, median + 1.5·MAD)`（单边，per-recording 自动）|
| 切出来的单元                 | burst（用力的一段）                            | motion run + 紧随的真实 hold（无缝拼接）|
| 在统一检测器中的角色         | 仲裁器（R2 拆分过长 run）+ 交叉引用 + 聚类特征缓存 | **主干（spine）**：画 onset 与 hold     |

**关键观察**：两者在持姿期表现相反——EMG 在维持姿势时仍然高（肌肉在保持），pose-speed 早就 0 了。这恰恰是组合使用的物理基础：单看任意一种都有大类盲区，两者一起几乎涵盖所有"手势状态"。

---

## 4. 每段 QC 列

每个 burst / clip 都附带 QC 指标，供按需筛选：

| 列                                                  | 解释                                                   |
|-----------------------------------------------------|--------------------------------------------------------|
| `emg_rms`                                           | 该区间 RMS（去基线之前的原始尺度）                     |
| `envelope_peak`                                     | 该区间 env 峰值                                        |
| `pose_range`                                        | `‖max(ja) - min(ja)‖`（关节偏移范围）                  |
| `mean_pose_speed` / `max_pose_speed`（仅 clip）     | 整 clip 区间速度统计                                   |
| `duration_s` / `motion_duration_s` / `hold_duration_s`（clip） | 整段 / motion 段 / hold 段时长（hold 由 `[hold_start, hold_end)` 算）|

典型用法：`clips[(envelope_peak < 5) & (pose_range < 0.5)]` 一行过滤微弱伪段。

---

## 5. 录制级 QC — EMG-pose lag + NaN 占比

新模块 [emg_label/qc.py](../emg_label/qc.py)，写到 `recordings.csv`（每录制一行）。

### 5.1 EMG-pose lag

互相关 EMG 包络与 pose-speed，找最高相关对应的偏移：

```python
lag_s, corr = qc.estimate_emg_pose_lag(env, pose_speed, fs,
                                        max_lag_s=1.0, min_corr=0.05)
```

约定：**lag > 0 = EMG 领先 pose**。生理上肌电应在动作前 50–300 ms 触发，所以健康范围 `(0, 0.4]` s。

`qc.lag_status(lag)` 分四档：
- `ok` — 在 (0, 0.4] s
- `early` — < 0（EMG 滞后动作；通常对齐 bug）
- `late` — > 0.4（漂移或弱信号伪峰）
- `nan` — 信号无效（全 NaN、近常值、最佳相关低于 `min_corr`）

NaN 鲁棒：内部把 NaN 替换为 0（z-score 后），全 NaN 切片直接返 NaN；`np.errstate` 静音 numpy 抛出的 invalid warning。

### 5.2 NaN 占比

`pose_nan_frac` / `emg_nan_frac` = 任一通道为 NaN 的样本占比。原始 Manus 偶有遮挡丢帧。`process_one` 在 `pose_nan_frac > 0.01` 时打印告警（与下面 §5.3 的筛选阈值一致）；NaN 占比偏高的录制基本上是被插值"造"出来的，不建议进真值集。

### 5.3 recordings.csv schema

| 列                                                                                  | 说明                                                   |
|-------------------------------------------------------------------------------------|--------------------------------------------------------|
| `source_file, source_path, group, subject, hand`                                    | 溯源                                                   |
| `n_samples, duration_s`                                                             | 规模                                                   |
| `n_bursts, n_clips, n_burst_only, n_clip_only`                                      | 双信号产出与分歧                                       |
| `emg_pose_lag_s, emg_pose_corr, lag_flag`                                           | 时序 QC                                                |
| `pose_nan_frac, emg_nan_frac`                                                       | 完整性 QC                                              |
| `enter_thresh, exit_thresh, pose_thresh, pose_exit_thresh`                           | 阈值快照（复盘用）：EMG 进/退阈 + 该录制的 `move_enter`/`move_exit`（自动 move 阈，rad/s）|
| `rec_pose_range, pose_static`                                                       | 全录制关节偏移 `‖max-min‖`（rad）；`pose_static=1` 表示整条未过绝对偏移闸门（被判静态、返 0 段）|

典型筛选：

```python
import pandas as pd
r = pd.read_csv("out/recordings.csv")
clean = r[(r.lag_flag == "ok") & (r.pose_nan_frac < 0.01)]   # 进真值集
review = r[r.lag_flag.isin(["early", "late"])]               # 人工复核
```

---

## 6. 聚类特征：hold 窗口里的 apex 姿态

### 6.1 为什么聚类只用 joint_angles，EMG 完全不参与

聚类问的是 "**哪些段是同一个手势**"。回答这问题的是手摆成什么形状，而不是用力多大：

- 同一个"比 1"，用力做和轻轻做，EMG 形状完全不同；但贴的标签应该是同一个手势。
- EMG 的形状还受太多无关因素影响：当天皮肤汗、电极位置略有偏移、肌肉疲劳……
- joint_angles 直接描述了"手摆成什么形状"——同一手势在不同时间、不同力度做，关节角都差不多。**这才是手势的身份证**。

所以分工：

| 信号           | 在切分阶段              | 在聚类阶段     | 在 QC 阶段          |
|----------------|-------------------------|----------------|---------------------|
| EMG            | 仲裁器（R2 拆分）+ 单独切 burst | **不用**   | 算 EMG-pose lag     |
| pose-speed     | 主干：画 onset 与 hold  | **不用**       | 间接（lag 的另一端） |
| joint_angles   | 算 pose-speed 的源 + 绝对偏移闸门 | **特征向量来源** | 算 pose_nan_frac    |

聚类**不用** EMG 段切片，而是用每段"持姿期"里的代表性姿态向量——下面 §6.2 是具体怎么取的。

### 6.2 apex 姿态特征

代码 [`features.apex_pose_feature`](../emg_label/features.py)（apex 帧由 [`features.apex_index`](../emg_label/features.py) 求得，两者共用同一帧，overview 图也标这帧）：

```python
# hold 窗口 = [start, window_end)  ← segments 用 hold_end_sample，clip 用其 hold
dev    = smooth(‖ja - rest‖)            # 每帧偏离中性多远（nan 帧置 -inf，绝不夺 argmax）
apex   = argmax(nan_to_num(dev, -inf))  # 最到位的瞬间
feature = nanmedian(ja[apex ± 50ms])    # 该帧附近 ±50ms 的 nan-aware 中位姿态 → (20,)
```

- `rest = nanmedian(整条录制的 joint_angles)` ≈ 中性手姿态（nan-aware）。
- **为什么取 hold 而非段内？** motion run 是"手正在移动到位"的过程；真正能区分手势的定型姿态在其**之后**的低速持姿期。
- **`nanmedian` 而非单帧**：抗抖动、抗坏帧，且 ±50ms 窗口内的少量丢帧不会把整条特征向量污染成 NaN（否则会传到 zscore/KMeans）。
- **这个特征只用 joint_angles**，与 FK/emg2pose 无关。
- **统一提取入口**：cluster.py / plot_cluster_features.py / evaluate.py **全部经 `features.feature_by_seg` 取特征**——它优先读 `out/features/{stem}.npz` 缓存（segment.py 写的、确定性 → 与重算逐位相同），缓存缺失或 stale 时回退用同一套 nan-aware `rest` 重算。这保证聚类、画图、评估打分的都是同一批特征。

---

## 7. 聚类（无监督 + 事后命名）

整个聚类**没有任何标签参与训练**：

- 切分纯信号处理（阈值检测）。
- 聚类是 **KMeans**，只看"哪些段的姿态向量彼此像"。
- 选 k 用 **silhouette**（内部指标，不需要真值）。
- 唯一人工介入：聚类**之后**看每簇代表姿态给簇起名（`labels.csv`）。

### 7.1 分组：每个 (被试,手) 各跑一次 KMeans

```python
for group, gdf in seg_df.groupby("group"):    # group = "subject-hand"
    X  = [apex_pose_feature(...) for seg in gdf]
    Xz = zscore(X)
    labels = KMeans(n_clusters=k, n_init=10).fit_predict(Xz)
```

为什么分组：不同被试手型/标定不同，左右手是镜像。混在一起 KMeans 会先按"谁的手"分。

### 7.2 跨被试合聚 + 按被试归一化

`cluster.py --group-by all` 把所有人合到一个 KMeans；用 `--subject-norm zscore` 在聚类前对**每个被试的特征**去均值除标准差，压低"个体身份"对距离的污染。归一化只作用于聚类特征，**簇心仍用原始绝对姿态**渲染 3D 手图。详见 [docs/汇报文档/多人单手聚类结果与分析.md](汇报文档/多人单手聚类结果与分析.md)。

### 7.3 K 怎么定 + 过/欠聚类

- 手动 `--k 18`（覆盖 `--k-min`/`--k-max`），或不传走 silhouette 在 `[k_min, k_max]`（默认 `[12, 30]`）扫。
- 固定 `--k` 若大于该组样本数会被 `clustering.select_k_and_cluster` 钳到 `n-1` 并打印告警。
- **K 宁大勿小**：过聚类（同一手势拆两簇）可恢复——标注时填相同名字即合并；欠聚类（两手势并一簇）不可恢复。

---

## 8. 真值集构建工作流（推荐主路径）

```
segment.py ──► segments.csv (burst)
            ├► clips.csv    (clip = motion run + 真实 hold，打标单元)
            └► recordings.csv (lag / NaN / 阈值快照)

      ↓ 按 lag_flag、pose_nan_frac 筛 recordings.csv

export_clips.py ──► clips_export/{stem}/c0000.npz   (每条录制一个子目录)
                ├► clips_export/{stem}/c0000.png   (6 帧关键帧)
                └► clips_export/index.html         (跨录制总览 + QC 摘要 + matched_burst 标红)

      ↓ 人浏览 index.html，在 clips.csv 填 gesture_label 列

→ clips_labeled.csv  (真值集)
```

每个 clip 的 6 帧 PNG = `clip_start → pre-motion → motion 1/3 → motion 2/3 → apex → clip_end`，足够标注员一眼辨识手势。

### 8.1 6 帧渲染的两条路径

[export_clips.py:_render_keyframes_*](../export_clips.py)：

1. **skeleton 路径（首选，无 torch）**：raw npz 含 `manus_*_skeleton`（25 关节 XYZ+四元数），直接画 XYZ 连线。[emg_label/skeleton.py](../emg_label/skeleton.py) 提供连接表和 `draw_skeleton(ax, xyz)`。生产环境无 torch 也能出图。
2. **emg2pose FK 路径（fallback）**：处理后数据无 skeleton，从 joint_angles 走 FK 渲染（需要 torch）。

策略：每条录制优先 skeleton，缺则回 FK；FK 首次失败后整轮跑禁用 FK，不再无谓尝试。

---

## 9. 输出 schema 速查

### 9.1 `segments.csv`（每个 EMG burst 一行）

| 列                                                                                       | 含义                                          |
|------------------------------------------------------------------------------------------|-----------------------------------------------|
| `source_file, source_path, group, seg_idx`                                               | 溯源                                          |
| `start_sample, end_sample`                                                               | burst 区间                                    |
| `hold_end_sample`                                                                        | hold 窗口终点（聚类直接读）                   |
| `apex_sample`                                                                            | hold 内最大偏离帧                             |
| `duration_s, emg_rms, envelope_peak, pose_range`                                         | QC                                            |
| `matched_clip_id`                                                                        | 该 burst 对应的 clip（-1 = 无）               |

### 9.2 `clips.csv`（每个 pose clip 一行 — 打标单元）

| 列                                                                                                                                       | 含义                                            |
|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| `source_file, source_path, group, subject, hand, clip_id`                                                                                | 溯源                                            |
| `clip_start_sample, clip_end_sample`                                                                                                     | 完整 action 区间（= `[start, 下一个 onset)`，无 pad）|
| `static_in_start/end_sample`                                                                                                            | **零宽**（`= clip_start`，无 pre-motion 静态期）|
| `motion_start/end_sample, static_out_start/end_sample`                                                                                   | motion 段 + 真实 hold（`static_out_* == hold_*`）|
| `hold_start_sample, hold_end_sample`                                                                                                     | 真实 hold 区间（显式命名 `static_out_*` 的别名）|
| `apex_sample`                                                                                                                            | hold 内最稳帧                                   |
| `duration_s, motion_duration_s, hold_duration_s, emg_rms, envelope_peak, mean_pose_speed, max_pose_speed, pose_range`                    | QC                                              |
| `matched_emg_seg_idx`                                                                                                                    | 该 clip 对应的 burst（-1 = EMG 漏切）           |
| `fusion_type`                                                                                                                            | `both`（motion 内有 EMG burst）/ `pose_only`    |
| `review_flag`                                                                                                                            | `long`/`nohold`/`slow`/`long_static`/`''`       |
| `gesture_label`                                                                                                                          | **空，等人填**                                  |

### 9.3 `recordings.csv`（每条录制一行）

见 §5.3。

### 9.4 `segments_clustered.csv`（聚类后）

= `segments.csv` + `cluster_id` 列。供 `export.py` 按 (group, cluster_id) → label 映射归档。

### 9.5 单段导出 npz（`export.py`）

| 键                                                                                              | 内容                                                 |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `emg, joint_angles`                                                                             | (seg_len, 16)+(seg_len, 20) 该 burst 切片            |
| `label, cluster_id, group, source_file, seg_idx, start_sample, end_sample, fs, hand`            | 标注与溯源（`hand` 显式写入，下游 `visualize_segment.py`/`animate_segment.py` 用 `io_utils.side_from_meta` 取它）|

### 9.6 单 clip 导出 npz（`export_clips.py`）

| 键                                                                                                                              | 内容                                                   |
|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `emg, joint_angles`                                                                                                             | (clip_len, 16)+(clip_len, 20) 完整 clip 切片           |
| `fs, clip_start, clip_end`                                                                                                      | 时间锚点                                               |
| `static_in_start/end, motion_start/end, static_out_start/end, apex`                                                             | 子结构（绝对 sample 索引）                             |
| `source_file, group, subject, hand, clip_id, matched_emg_seg_idx`                                                               | 溯源                                                   |

兼容 `visualize_segment.py` 做单段深度可视化。

---

## 10. 完整流水线全景

```
                    ┌─ raw_npz (/mnt/pose_data/...)
                    │     manus_<hand>_ergonomics + manus_<hand>_skeleton
                    ├─ processed_npz (joint_angles 已对齐)
                    │
io_utils.load_npz   │   ← 自动识别 + 对齐 pose 到 EMG 轴 + 度→弧度
                    ▼
   action_segmentation.segment_recording  (统一检测器)
        pose_speed (主干) + robust_threshold 自动 move 阈
        + EMG envelope/burst (仲裁器 R2)  +  dip-merge / NaN 填补
                    │
        ┌───────────┴────────────┐
        │                        │
   EMG bursts                action = motion run + 真实 hold
        │                        │
        │  ←── burst ↔ clip 匹配 ──→
        ▼                        ▼
   segments.csv             clips.csv  ←──── 打标 gesture_label
   (+ features/{stem}.npz 缓存)        │
        │ (聚类路径)               │ (真值集路径)
        ▼                          ▼
  cluster.py → KMeans         export_clips.py
        │                          │
        ▼                          ▼
  segments_clustered.csv      clips_export/{stem}/c*.npz + c*.png + index.html
        │                          │
  填 labels.csv 命名簇         浏览 index.html 填 clips.csv（或用 label_server 写 clip_labels.csv）
        │                          │
        ▼                          ▼
  export.py → segments/<label>/*    clips_labeled.csv (= 真值集)
                                 ┌────────────────────────────┐
                                 │ recordings.csv （全局 QC） │
                                 │ lag_flag, pose_nan_frac    │
                                 └────────────────────────────┘
```

具体命令、参数、典型场景见 [RUNBOOK.md](RUNBOOK.md)。
