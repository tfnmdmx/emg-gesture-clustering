# RUNBOOK — 切分 / 打标 / 聚类 / 导出

两条独立工作流，按目标选：

- **A. 真值集构建（推荐）** — 双信号切分 + 6 帧关键帧画廊 + 人工填 `clips.csv`。
  入口：`segment.py` + `export_clips.py`。
- **B. 无监督聚类 + 事后命名（旧）** — KMeans 聚 + 命名 → 训练数据集。
  入口：`pulse.sh prep ...`（包装层）或单独跑 `segment.py / cluster.py / export.py`。

原理见 [METHOD.md](METHOD.md)。

> `reference/gesture_velocity_segmentation.py` 只做**切分 + 可视化**，不做聚类。本项目 A 路径借鉴它的双信号切分（pose-speed + clip）、关键帧渲染、数据源约定（仅 `/mnt/pose_data/`），并在其上加了 EMG 通路、交叉引用、`recordings.csv` QC、`clips.csv` 标注模板。B 路径的聚类是本项目自己加的，reference 没有。

---

## A. 真值集构建（主路径）

### A.1 一条龙（推荐）

```bash
cd /data/cl_data/action-clustering

# 最常见：meta CSV 喂入 + 并行 8
WORKERS=8 ./pulse.sh raw reference/sample_meta.csv

# 或：直接喂原始数据根（按 {subject}/{date-hand}/{stamp}.npz 递归）
WORKERS=8 ./pulse.sh raw /mnt/pose_data/emg2pose/data
```

`./pulse.sh raw` 内部按顺序跑 `raw-segment → raw-qc → raw-export`：

| 子步骤 | 产物 |
|--------|------|
| `raw-segment` | `out_pose/shards/{stem}/*` + 顶层 `segments.csv / clips.csv / recordings.csv` + `features/{stem}.npz` |
| `raw-qc` | `out_pose/recordings_keep.csv`（lag_flag=ok & pose_nan_frac<0.01 的子集） |
| `raw-export` | `out_pose/clips_export/<key>.{npz,png}` + `index.html`（增量刷新） |

跑完打开 `out_pose/clips_export/index.html` 浏览关键帧，往 `out_pose/clips.csv` 的 `gesture_label` 列填手势名即可。

### A.1.1 三步分开跑（与一条龙等价）

```bash
WORKERS=8 ./pulse.sh raw-segment reference/sample_meta.csv   # 仅 stage 1
./pulse.sh raw-qc                                            # 仅 QC 过滤
./pulse.sh raw-export                                        # 仅 stage 3 关键帧
./pulse.sh raw-status                                        # 看进度
```

### A.1.2 可调环境变量（raw 流）

| 变量 | 默认 | 含义 |
|------|------|------|
| `RAW_OUT` | `out_pose` | 输出目录 |
| `WORKERS` | `1` | segment 并行 worker 数（每条录制独立进程） |
| `SUBJECT` | 空 | 只处理指定 subject（透传 `--only-subject`） |
| `ONLY_HAND` | 空 | 只处理 `left` 或 `right`（透传 `--only-hand`） |
| `NO_OVERVIEW` | `0` | 设 `1` 跳过 `overview.png` 渲染（提速） |
| `RAW_LAG_FLAGS` | `ok` | QC 保留的 lag_flag（逗号分隔，可加 `early`/`late`） |
| `RAW_NAN_MAX` | `0.01` | QC `pose_nan_frac` 上限 |
| `RAW_LIMIT` | 空 | 每个 source_file 限制导出 clip 数（采样查看） |
| `RAW_NO_PNG` | `0` | 设 `1` 跳过关键帧 PNG（无 torch 时用） |

### A.1.3 裸命令等价（不想用 wrapper）

```bash
# 1) 切分（含 features 缓存 + shard 落盘 + 顶层 csv 汇总）
python segment.py --meta reference/sample_meta.csv --out out_pose --workers 8

# 2) 按 QC 过滤可信子集（lag 健康 + NaN 占比低）
python -c "
import pandas as pd
r = pd.read_csv('out_pose/recordings.csv')
keep = r[(r.lag_flag == 'ok') & (r.pose_nan_frac < 0.01)]
print(f'keep {len(keep)}/{len(r)} recordings')
keep.to_csv('out_pose/recordings_keep.csv', index=False)
"

# 3) 导出 clip 切片 + 6 帧关键帧 + HTML 浏览页
python export_clips.py --out out_pose
```

### A.2 数据源三种喂法

`segment.py` 入口接受三种数据源指法：

```bash
# (a) 处理后数据，扁平目录
python segment.py /data/cl_data/ai-infra/processed_data/fgw0917_0502_left --out out_fgw

# (b) 原始数据，按 {subject}/{date-hand}/{stamp}.npz 递归
python segment.py /mnt/pose_data/emg2pose/data --out out_raw --recursive

# (c) 用 sample_meta.csv 驱动（hand 从 meta 取，side=both 自动展两份）
python segment.py --meta reference/sample_meta.csv --out out_pose
```

文件名格式自适应：处理后 `{subject}__{date}-{hand}__{stamp}.npz` 解析 hand；原始 `{stamp}.npz` 退到父目录 `{date}-{hand}` 取 hand。

### A.3 force_data 默认跳过

reference 明确只用 `/mnt/pose_data/`，不用 `/mnt/force_data/`（force 任务是等长收缩、姿态变化少，pose-speed 切出基本是噪声）。本流水线沿用：

```bash
# 默认：force_data 文件自动剔除
python segment.py --meta sample_meta.csv --out out
# → SKIP 10175 force_data file(s) (pass --allow-force to include them)

# 强制要 force_data（不推荐）
python segment.py --meta sample_meta.csv --out out --allow-force
```

### A.4 关键 QC 列怎么用

`recordings.csv`（每条录制一行）：

| 列                                              | 怎么用                                                       |
|-------------------------------------------------|--------------------------------------------------------------|
| `lag_flag`                                      | `ok` 进真值集；`early`/`late` 复核；`nan` 直接弃             |
| `emg_pose_lag_s, emg_pose_corr`                 | 异常 lag 的具体值与置信度                                    |
| `pose_nan_frac, emg_nan_frac`                   | > 0.1 弃用（大量插值假帧）                                   |
| `n_burst_only, n_clip_only`                     | 双信号分歧；clip_only 数高 = 大量 EMG 漏切的真手势           |
| `enter_thresh, exit_thresh, pose_thresh`        | 阈值快照，复盘阈值是否合理                                   |

`clips.csv`（打标单元）的关键过滤列：

| 列                                              | 典型阈值                                                     |
|-------------------------------------------------|--------------------------------------------------------------|
| `motion_duration_s`                             | < 0.25 s 通常是噪声毛刺                                      |
| `pose_range`                                    | < 0.5 rad ≈ 几乎没动                                         |
| `envelope_peak`                                 | < 5 ≈ EMG 没真的发力                                         |
| `matched_emg_seg_idx == -1`                     | EMG 漏切 → **优先复核**（标注前先看这批）                    |

### A.5 打标 UI

`out_pose/clips_export/index.html`：

- 按 `source_file` 分组，每行 = 一个 clip：clip_id / QC 表 / 6 帧关键帧
- `matched_burst = -1` 自动标红
- 关键帧渲染优先用 `manus_*_skeleton`（原始数据，无 torch 依赖）；处理后数据回退到 emg2pose FK（需 torch）

填表只动 `clips.csv` 的 `gesture_label` 列：

```bash
# 不要直接编辑 clips.csv 的其它列（脚本会重写）
# 复制一份再编辑更稳：
cp out_pose/clips.csv out_pose/clips_labeled.csv
# 在 clips_labeled.csv 填 gesture_label
```

留空 = 丢弃该 clip；同名 = 合并；任意手势名（`fist` / `pinch_index` / `one` 等）。

### A.6 单 clip 深度查看

`export_clips.py` 产出的每个 `<key>.npz` 直接喂 `visualize_segment.py`：

```bash
python visualize_segment.py out_pose/clips_export/<key>.npz -o /tmp/x.png
# 四面板：16 通道 EMG / 包络 / 20 关节角 / 3D 手 (start/apex/end)
```

---

## B. 无监督聚类 + 事后命名（旧路径）

适用于"先聚类再批量贴标签做训练集"。整套包了 `pulse.sh` wrapper。

### B.1 TL;DR

```bash
cd /data/cl_data/action-clustering

# 1) 一条命令：建池 + 切分 + 聚类(k=18) + 评估 + 质检图
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right

# 2) 人工命名（看 out/clusters/*_hands.png 和 out/hand_anim/index.html）
cp out/labels_template.csv out/labels.csv
# 编辑 labels.csv 的 label 列：起名 / 同名合并 / 留空丢弃

# 3) 导出
./pulse.sh export
```

### B.2 子命令

```
./pulse.sh prep   <批次...>    建池+切分+聚类+评估+质检
./pulse.sh run                 切分+聚类+评估+质检（池已建好）
./pulse.sh pool   <批次...>    只建池
./pulse.sh segment             阶段 1 切分
./pulse.sh cluster [K]         阶段 2 聚类（K 默认 18；写 auto = silhouette）
./pulse.sh eval                被试无关性评估（pooled 运行）
./pulse.sh qc                  特征图 + 3D 动画画廊
./pulse.sh export              阶段 3 导出带标签 npz
./pulse.sh status              查看进度
./pulse.sh help
```

`<批次>` 写 `processed_data/` 下目录名（如 `fgw0917_0502_left`），或完整路径。

### B.3 可调项（env 变量）

| 变量              | 默认                                      | 含义                                                  |
|-------------------|-------------------------------------------|-------------------------------------------------------|
| `NAME`            | (空)                                      | 一键派生池/输出名（`NAME=fgw` → `POOL=work_pool_fgw, OUT=out_fgw`） |
| `K`               | 18                                        | 聚类簇数                                              |
| `GROUP_BY`        | `subject-hand`                            | 聚类粒度：`subject-hand` / `hand` / `all`             |
| `SUBJECT_NORM`    | `none`                                    | 按被试归一化：`none` / `center` / `zscore`（仅 pooled）|
| `OUT`             | `out`                                     | 输出目录                                              |
| `POOL`            | `work_pool`                               | 池目录                                                |
| `N_GALLERY`       | 3                                         | 动画画廊每类样本数                                    |
| `DATA_ROOT`       | `/data/cl_data/ai-infra/processed_data`   | 批次根目录                                            |
| `PY`              | emg2pose conda python                     | 解释器                                                |

**池/输出名不会自动按数据起**——不同批次都用默认 `work_pool/out` 会互相覆盖。两种避免：
- 显式 `POOL=work_pool_fgw OUT=out_fgw ./pulse.sh prep ...`
- 或 `NAME=fgw ./pulse.sh prep ...`（自动派生）

### B.4 典型场景

**复刻 pilot（单被试左右手）**：
```bash
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right
./pulse.sh export   # 标注后
```

**扩到多被试**（每个 `(被试, 手)` 自动成一组）：
```bash
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right \
                wjh0111_0502_left wjh0111_0503_right
```

**只换 k 重看**（`segments.csv` 不依赖 k）：
```bash
./pulse.sh cluster 24
./pulse.sh qc
```

**全局聚类**（不分被试/手，配合按被试归一化）：
```bash
GROUP_BY=all SUBJECT_NORM=zscore ./pulse.sh run
```
> ⚠️ 不分组直接合聚时 KMeans 常按"谁的手"而非"什么手势"分。归一化 (`SUBJECT_NORM`) 压低这种污染，但效果有限；详见 [docs/汇报文档/多人单手聚类结果与分析.md](汇报文档/多人单手聚类结果与分析.md)。

**对比归一化方式**（无需重切）：
```bash
POOL=old_data/work_pool_4users OUT=out_4users_center  GROUP_BY=all SUBJECT_NORM=center ./pulse.sh run
POOL=old_data/work_pool_4users OUT=out_4users_zscore  GROUP_BY=all SUBJECT_NORM=zscore ./pulse.sh run
```

### B.5 人工命名（唯一人工环节）

1. 看图决定名字：
   - `out/clusters/<group>_hands.png` — 每簇 3D 手姿（**主依据**）
   - `out/hand_anim/index.html` — 每类前 N 个样本动画（判断簇内一致性）
2. `cp out/labels_template.csv out/labels.csv`，只改 `label` 列：
   - 任意名字（`fist`、`pinch_index`、`one` …）
   - **同名合并**：同手势被拆多簇时多行填相同名
   - **留空丢弃**：跳过该簇
3. `./pulse.sh export`

> **缺 labels.csv** 直接 `export`：自动用占位名 `<group>-<cluster_id>` 全部导出（先看结果再回头改名）。

### B.6 产物

| 阶段    | 产物                                                                                          |
|---------|-----------------------------------------------------------------------------------------------|
| segment | `segments.csv`、`clips.csv`、`recordings.csv`、`overview/*.png`                               |
| cluster | `segments_clustered.csv`、`labels_template.csv`、`clusters/<group>{,_hands}.png`              |
| eval    | `eval_metrics.csv`（被试无关性指标），`eval_cache/`                                            |
| qc      | `feature_maps/<group>_features.png`、`hand_anim/index.html`                                   |
| export  | `segments/<label>/*.npz`、`labeled_overview/*.png`                                            |

---

## 常见踩坑

| 现象                                                | 原因 / 解决                                                                                    |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|
| `SKIP N force_data file(s)`                         | A3 默认行为，要 `--allow-force` 才包含（基本只在 debug 时用）                                  |
| `lag=n/a (corr=0.00, nan)`                          | 该录制 EMG 或 pose 信号近常值/全 NaN；`recordings.csv.lag_flag=nan`，弃用                      |
| `lag=+0.500s (corr=0.20, late)`                     | EMG 领先 pose 超 400ms；可能是任务特性（持姿期 EMG 才达峰），看 `corr` 决定是否进真值集        |
| `lag=-0.21s (corr=0.30, early)`                     | EMG 滞后 pose；通常是预处理对齐 bug，需要去查源头 npz                                          |
| `pose_nan_frac > 0.10`                              | Manus 大量丢帧，姿态基本靠插值，弃用                                                           |
| HTML 画廊"no PNG"                                   | 该 clip 渲染失败；首选 skeleton 路径需 raw npz 含 `manus_*_skeleton`，否则走 FK 需要 torch     |
| 第一条 clip 报 `keyframe FK unavailable`            | 处理后数据无 skeleton 且无 torch；用 raw 数据，或装 torch                                      |
| `ModuleNotFoundError: sklearn`                      | 没走 pulse.sh（它锁定了 emg2pose conda python）；或用 `$PY` 显式指                              |
| `OMP: Error #34 ...`                                | 没走 pulse.sh；手动跑 cluster 加 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` |
| pool 里冒出 `<date>_<time>` 组名                    | 该批次文件名无 subject/hand 解析不了；换批次或删掉                                              |
| export 0 个                                         | `labels.csv` 的 label 列全空，或表头被改坏                                                     |
| 簇都长得一样                                        | 段太少 / k 太大；减 k 或加更多文件                                                              |
| 3D 手图右手 IndexError                              | emg2pose 右手 init bug；代码已走 left FK + 镜像，别改回 `side='right'`                          |

---

## 不想用 wrapper？（等价裸命令）

```bash
PY=/home/chenglin/anaconda3/envs/emg2pose/bin/python
cd /data/cl_data/action-clustering

# 真值集路径
$PY segment.py --meta reference/sample_meta.csv --out out_pose
$PY export_clips.py --out out_pose

# 聚类路径
mkdir -p work_pool && (cd work_pool && rm -f *.npz && \
  for d in fgw0917_0502_left fgw0917_0504_right; do \
    for f in /data/cl_data/ai-infra/processed_data/$d/*__*__*.npz; do ln -sf "$f" .; done; done)
$PY segment.py work_pool --out out
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY cluster.py work_pool --out out --k 18
$PY plot_cluster_features.py work_pool --out out
$PY build_anim_gallery.py --out-root out --n 3 --clean
cp out/labels_template.csv out/labels.csv   # 编辑 label 列
$PY export.py work_pool --out out
```

---

## 单段可视化（细看某 clip / 某 burst）

```bash
PY=/home/chenglin/anaconda3/envs/emg2pose/bin/python

# 静态四面板（EMG / 包络 / 关节角 / 3D 手 start-apex-end）
$PY visualize_segment.py out_pose/clips_export/<key>.npz -o /tmp/x.png

# 单段交互式 3D 动画（plotly html）
$PY animate_segment.py out_pose/clips_export/<key>.npz
```
