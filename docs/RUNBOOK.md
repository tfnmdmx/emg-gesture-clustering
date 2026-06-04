# RUNBOOK — 切分 / 打标 / 聚类 / 导出

两条工作流，按目标选。**两条都把数据落进同一个 SQLite 索引 `OUT/index.db`**（store.py 维护），并都以 **clip `(source_file, clip_id)`** 为唯一单元——打标、聚类、评估全用这同一个键，所以"用打标评估聚类"只是一次 JOIN。

- **A. 真值集构建（推荐）** — 双信号切分 + 6 帧关键帧画廊 + Web UI 人工打标（写 `index.db` 的 `annotations` 表）。
  入口：`segment.py` + `export_clips.py` + `label_server.py`（`./pulse.sh label`）。
- **B. 无监督聚类 + 标签评估** — KMeans 聚（apex 或 trajectory 通道）→ 用人工标签评估聚类 → 按手势导出训练集。
  入口：`pulse.sh segment/cluster/cluster-traj/eval/export ...`（包装层）或单独跑 `segment.py / cluster.py / cluster_traj.py / evaluate.py / export.py`。

数据层结构：`OUT/shards/{stem}/` 仍是切分**真值**（每条录制一个 shard）；`OUT/index.db` 是从 shard 重建的派生索引（表：`recordings, bursts, clips, annotations, cluster_runs, cluster_assignments, tombstones`），其中 `annotations` 表是**标签真值**（不在 shard 里）。`segment.py` 跑完会顺带导出顶层便利 CSV `recordings.csv / bursts.csv / clips.csv`（给 diag_seg / cluster_traj / visualize_segment 这些读扁平文件的工具用，bursts/clips 里 JOIN 回了 `source_path`）。

原理见 [METHOD.md](METHOD.md)。

> `reference/gesture_velocity_segmentation.py` 只做**切分 + 可视化**，不做聚类。本项目 A 路径借鉴它的双信号切分（pose-speed + clip）、关键帧渲染、数据源约定（仅 `/mnt/pose_data/`），并在其上加了 EMG 通路、交叉引用、`recordings.csv` QC、Web UI 打标。B 路径的聚类是本项目自己加的，reference 没有。

---

## A. 真值集构建（主路径）

### A.1 一条龙（推荐）

```bash
cd /data/cl_data/action-clustering-compact

# 最常见：meta CSV 喂入 + 并行 8
WORKERS=8 ./pulse.sh raw reference/sample_meta.csv

# 或：直接喂原始数据根（按 {subject}/{date-hand}/{stamp}.npz 递归）
WORKERS=8 ./pulse.sh raw /mnt/pose_data/emg2pose/data
```

`./pulse.sh raw` 内部按顺序跑 `raw-segment → raw-qc → raw-export`：

| 子步骤 | 产物 |
|--------|------|
| `raw-segment` | `out_pose/index.db` + `out_pose/shards/{stem}/{recording.csv, bursts.csv, clips.csv, overview.png}` + 顶层便利 CSV `recordings.csv / bursts.csv / clips.csv` + `features/{stem}.npz` |
| `raw-qc` | `out_pose/recordings_keep.csv`（`lag_flag in RAW_LAG_FLAGS(ok)` & `pose_nan_frac<RAW_NAN_MAX(0.01)` 的**参考**子集，非门控；默认阈值下常近乎空，详见 §QC） |
| `raw-export` | `out_pose/clips_export/{stem}/c{cid:04d}.{npz,png}`（每条录制一个子目录）+ `index.html`（增量刷新） |

> 注：shard 内的录制级文件叫 `recording.csv`（单数），顶层汇总叫 `recordings.csv`（复数）。`bursts.csv` 是 EMG 激活段（旧名 `segments.csv`，已重命名）。

跑完打开 `out_pose/clips_export/index.html` 浏览关键帧；**打标走 Web UI**（`./pulse.sh label`，见 §A.5），标签写进 `index.db` 的 `annotations` 表，不再编辑任何 CSV 的 `gesture_label` 列（该列已从 `clips.csv` 移除）。

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
| `SUBJECTS` | 空 | 只处理这些 subject（逗号分隔，透传 `--subjects`）；按**归一化 id** 匹配，`fgw-0917 == fgw0917`，meta CSV 与 processed_data 用法一致 |
| `SUBJECT` | 空 | `SUBJECTS` 的单值遗留别名（pulse.sh 把 `SUBJECT` 灌进 `SUBJECTS`，最终也是 `--subjects`，**不是** `--only-subject`） |
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
| `lag_flag`                                      | 全局 EMG↔pose 互相关的诊断标签（`ok`/`early`/`late`/`nan`）。**已知在本类准周期数据上有偏**（持姿期 EMG 才达峰，互相关易把绝大多数判成 `ok`），**非硬门控**——见 §QC 说明 |
| `emg_pose_lag_s, emg_pose_corr`                 | 上面 lag 的具体秒数与互相关置信度                            |
| `pose_nan_frac, emg_nan_frac`                   | NaN/插值占比；`raw-qc` 默认 `pose_nan_frac < 0.01` 才保留（见 `RAW_NAN_MAX`），手动复核可放宽 |
| `n_burst_only, n_clip_only`                     | 双信号分歧；clip_only 数高 = 大量 EMG 漏切的真手势           |
| `enter_thresh, exit_thresh, pose_thresh, pose_exit_thresh` | 阈值快照，复盘阈值是否合理                          |

> **§QC `lag_flag` 注意**：`raw-qc`（`./pulse.sh raw-qc`）默认只保留 `RAW_LAG_FLAGS=ok` 且 `pose_nan_frac < RAW_NAN_MAX(0.01)` 的录制写入 `recordings_keep.csv`。但 `lag_flag` 是基于**全局互相关**（`qc.estimate_emg_pose_lag` / `lag_status`）的诊断量，在本项目这种准周期、持姿期 EMG 才达峰的数据上**已知有偏**，常把几乎所有录制判成 `ok`；叠加 0.01 的 NaN 上限后 `recordings_keep.csv` 往往近乎空集。它**不是硬门控**——`export_clips.py` 仍读完整 `clips.csv`，`recordings_keep.csv` 只是参考子集。要让它有用，建议放宽：`RAW_LAG_FLAGS=ok,early,late RAW_NAN_MAX=0.1 ./pulse.sh raw-qc`。

`clips.csv`（打标单元）的关键过滤列：

| 列                                              | 典型阈值                                                     |
|-------------------------------------------------|--------------------------------------------------------------|
| `motion_duration_s`                             | < 0.25 s 通常是噪声毛刺                                      |
| `pose_range`                                    | < 0.5 rad ≈ 几乎没动                                         |
| `envelope_peak`                                 | < 5 ≈ EMG 没真的发力                                         |
| `matched_burst_idx == -1`                       | EMG 漏切 → **优先复核**（标注前先看这批）                    |

### A.5 打标 Web UI（`./pulse.sh label`）

```bash
# 起本地服务（默认 127.0.0.1:8000），读 RAW_OUT/index.db
./pulse.sh label                       # PORT=8000 HOST=127.0.0.1 可覆盖
# 可选：先预渲染手部帧缓存，首开秒出
WORKERS=8 ./pulse.sh label-prewarm
```

`label_server.py` 从 `index.db` 读 clips / recordings / bursts，浏览器里逐 clip 打标，所有标签写回 **`annotations` 表**（不写任何 CSV）：

- 每条标签是一行 annotation，带 `scope`（`clip` / `recording`）+ `kind`（`label` / `invalid`），并携带 **重切分稳定锚点** `(clip_start_sample, clip_end_sample, seg_version)`——以后重切分时 `store.remap_annotations` 按样本区间重叠把旧标签迁到新 clip（重叠不足不会静默丢，标 `note='待复核'`）。
- **软 `invalid`**：可对单个 clip 或整条录制打 `invalid`（`scope='recording'`），被标的 clip / 录制从打标、聚类、评估、导出里排除，但**数据还在**，可撤销。
- **永久删除（"删除录制"按钮 / `POST /api/drop_recording`）**：与软 invalid **不同**——彻底删掉该录制的 db 行 + annotations，写一条 `tombstone`（重切分**不会复活**它），并删掉磁盘上的 shard。不可撤销。

> webui 资源在 `webui/index.html` + `webui/app.js`。3D 手用 plotly：优先本地 `webui/plotly.min.js`（离线可用，**未提交**），缺了回退 CDN。

打标约定：任意手势名（`fist` / `pinch_index` / `one` 等）；同名自动合并；留空（未打标）= 导出时丢弃该 clip。

### A.6 单 clip 深度查看

`export_clips.py` 产出的每个 npz 直接喂 `visualize_segment.py`：

```bash
python visualize_segment.py out_pose/clips_export/<stem>/c0000.npz -o /tmp/x.png
# 四面板：16 通道 EMG / 包络 / 20 关节角 / 3D 手 (start/apex/end)
```

---

## B. 无监督聚类 + 标签评估

适用于"切分 → 打标 → 聚类 → 用标签评估聚类 → 按手势导出训练集"。整套包了 `pulse.sh` wrapper。clip 是唯一单元，标签和聚类共用键，所以评估是一次 JOIN。**不再有"给每个簇起名再按簇导出"那一步**（labels_template.csv / segments_clustered.csv / `cp labels.csv` / `OUT/segments/<label>/` 全部移除）。

### B.1 TL;DR

```bash
cd /data/cl_data/action-clustering-compact

# 1) 切分（建 OUT/index.db + 便利 CSV）。源可以是 meta CSV / 目录 / 'processed'
WORKERS=8 ./pulse.sh segment reference/sample_meta.csv

# 2) 打标（Web UI，写 annotations 表）
./pulse.sh label

# 3) 聚类：apex（静态成形姿）和/或 trajectory（时序轨迹）通道
./pulse.sh cluster          # apex 通道，k=18 -> cluster_runs/<run_id>/
./pulse.sh cluster-traj     # trajectory 通道（可选，深度开发用）

# 4) 用标签评估某个 run（默认最新；RUN= 指定）-> cluster_runs/<run>/metrics.json
./pulse.sh eval

# 5) 按手势导出 -> OUT/gestures/<gesture>/
./pulse.sh export
```

### B.2 子命令

```
./pulse.sh segment [meta|dir|processed]  阶段 1 切分，建 OUT/index.db（无需建池）
./pulse.sh cluster [K]         阶段 2 apex 聚类（K 默认 18；"auto" = silhouette 选 k）-> cluster_runs/
./pulse.sh cluster-traj [K]    阶段 2 trajectory（时序）聚类 -> cluster_runs/（REPR= 选表征）
./pulse.sh eval                标签驱动评估 + 被试无关性（--run RUN= 或最新）-> cluster_runs/<run>/metrics.json
./pulse.sh db                  从 shards 重建 OUT/index.db（+ 便利 CSV）
./pulse.sh export              阶段 3 按手势导出标注 clip -> OUT/gestures/<label>/
./pulse.sh qc                  某 run 特征图 + 动画画廊
./pulse.sh gallery             单独建动画画廊（export 之后）
./pulse.sh run    [meta|dir]   segment + apex cluster + eval
./pulse.sh pool   <批次...>    （可选）把 processed_data 批次汇成 work_pool，只用于挑/并批次
./pulse.sh prep   <批次...>    pool + run（打标前的全部步骤）
./pulse.sh status              查看进度（index.db / cluster_runs / gestures）
./pulse.sh label               Web UI 打标（见 §A.5）
./pulse.sh help
```

`<批次>` 写 `processed_data/` 下目录名（如 `fgw0917_0502_left`），或完整路径。聚类/评估/导出现在通过 `index.db` 里的 `source_path` 找 npz，**不再强制建池**——`pool` 仅在你要手动挑选/合并特定批次时用。也可以用 `SUBJECTS=id1,id2 ./pulse.sh segment processed` 直接按 subject id 从 `$DATA_ROOT` 选数据。

### B.3 可调项（env 变量）

| 变量              | 默认                                      | 含义                                                  |
|-------------------|-------------------------------------------|-------------------------------------------------------|
| `NAME`            | (空)                                      | 一键派生池/输出名（`NAME=fgw` → `POOL=work_pool_fgw, OUT=out_fgw`） |
| `K`               | 18                                        | 聚类簇数（apex 与 trajectory 共用）                   |
| `GROUP_BY`        | `subject-hand`                            | apex 聚类粒度：`subject-hand` / `hand` / `all`        |
| `SUBJECT_NORM`    | `none`                                    | 按被试归一化：`none` / `center` / `zscore`（仅 pooled，cluster 与 eval 共用）|
| `REPR`            | `centered`                                | trajectory 通道表征：`centered` / `velocity` / `raw`（仅 `cluster-traj`） |
| `RUN`             | (空)                                      | `eval` / `qc` 评估/绘图的目标 run_id；空 = `index.db` 里最新一个 run |
| `OUT`             | `out`                                     | 输出目录（含 `index.db`、`cluster_runs/`、`gestures/`）|
| `POOL`            | `work_pool`                               | 池目录（仅 `pool`/`prep` 用）                          |
| `N_GALLERY`       | 3                                         | 动画画廊每类样本数                                    |
| `DATA_ROOT`       | `/data/cl_data/ai-infra/processed_data`   | 批次根目录 / `segment processed` 的源                  |
| `PY`              | emg2pose conda python                     | 解释器                                                |

**池/输出名不会自动按数据起**——不同批次都用默认 `work_pool/out` 会互相覆盖。两种避免：
- 显式 `POOL=work_pool_fgw OUT=out_fgw ./pulse.sh prep ...`
- 或 `NAME=fgw ./pulse.sh prep ...`（自动派生）

### B.4 典型场景

**复刻 pilot（单被试左右手）**：
```bash
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right   # pool + segment + cluster + eval
./pulse.sh label                                       # 打标
./pulse.sh eval                                         # 有标签后重新评估
./pulse.sh export                                       # 按手势导出
```

**扩到多被试**（每个 `(被试, 手)` 自动成一组）：
```bash
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right \
                wjh0111_0502_left wjh0111_0503_right
```

**只换 k 重聚**（切分不依赖 k；每次聚类是一个新的 `cluster_runs/<run_id>/`，旧 run 不被覆盖）：
```bash
./pulse.sh cluster 24
RUN=<run_id> ./pulse.sh qc     # 指定 run 画特征图；不指定则用最新 run
```

**跑 trajectory（时序）通道**（与 apex 通道并存，写同样的 `cluster_runs/` 契约）：
```bash
REPR=centered ./pulse.sh cluster-traj
./pulse.sh eval               # 默认评估最新 run（即刚跑的 trajectory run）
```

**从 shards 重建索引**（`index.db` 丢了/被中断、或想刷新便利 CSV）：
```bash
./pulse.sh db                 # = python segment.py --out OUT --index-only
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

### B.5 打标 + 评估（取代旧的"人工命名"）

不再"给簇起名"。流程是：

1. **打标**：`./pulse.sh label`，在 Web UI 里按 clip 打手势名 / 标 invalid（写 `annotations` 表，见 §A.5）。
2. **看聚类**：每个 run 的 `cluster_runs/<run_id>/gallery/`（apex 是每簇 3D 手姿 `<group>_hands.png`；trajectory 是每簇 medoid 关键帧条 + `index.html`）+ `feature_maps.png`（`./pulse.sh qc`，PCA/t-SNE/热图）。
3. **评估**：`./pulse.sh eval`（`RUN=` 指定 run，默认最新）把聚类 assignment 和 clip 标签 JOIN，算 ARI / NMI / purity + 每簇主导手势占比 + 被试污染 + pose 空间 silhouette/leakage/LOSO，写 `cluster_runs/<run>/metrics.json`。标签太少（<2 标注 clip 或 <2 类）时 ARI/NMI/purity = N/A，不会崩。
4. **导出**：`./pulse.sh export` 按手势名把**已标注**的 clip 分组导出（不按簇）。

> ⚠️ 不分组直接合聚（`GROUP_BY=all`）时 KMeans 常按"谁的手"而非"什么手势"分。归一化 (`SUBJECT_NORM`) 压低这种污染，但效果有限；`eval` 的被试污染指标就是量它的。详见 [docs/汇报文档/多人单手聚类结果与分析.md](汇报文档/多人单手聚类结果与分析.md)。

### B.6 产物

| 阶段    | 产物                                                                                          |
|---------|-----------------------------------------------------------------------------------------------|
| segment | `OUT/index.db`（权威索引）+ 每条录制一个 shard `shards/{stem}/`（`recording.csv` / `bursts.csv` / `clips.csv` / `overview.png`）+ 顶层便利 CSV `recordings.csv` / `bursts.csv` / `clips.csv` + `features/{stem}.npz` |
| label   | 写进 `index.db` 的 `annotations` 表（clip / recording 标签 + invalid + tombstone），不落 CSV |
| cluster / cluster-traj | `cluster_runs/<run_id>/{params.json, clusters.csv}` + `gallery/` + db 里的 `cluster_runs` / `cluster_assignments` 行（`run_id = {YYYYMMDD-HHMMSS}__{paramhash8}`） |
| eval    | `cluster_runs/<run>/metrics.json`（标签驱动 ARI/NMI/purity + 被试污染 + pose 空间探针）       |
| qc      | `cluster_runs/<run>/feature_maps.png`（PCA/t-SNE/热图）、`OUT/hand_anim/index.html`（动画画廊，读 `gestures/`） |
| export  | `OUT/gestures/<label>/<label>__<subject>-<hand>__<stem>__clip{cid:04d}.npz`、`OUT/labeled_overview/*.png` |

---

## 常见踩坑

| 现象                                                | 原因 / 解决                                                                                    |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|
| `SKIP N force_data file(s)`                         | A3 默认行为，要 `--allow-force` 才包含（基本只在 debug 时用）                                  |
| `lag=n/a (corr=0.00, nan)`                          | 该录制 EMG 或 pose 信号近常值/全 NaN；`recordings.csv.lag_flag=nan`，弃用                      |
| `lag=+0.500s (corr=0.20, late)`                     | EMG 领先 pose 超 400ms；可能是任务特性（持姿期 EMG 才达峰），看 `corr` 决定是否进真值集        |
| `lag=-0.21s (corr=0.30, early)`                     | EMG 滞后 pose；通常是预处理对齐 bug，需要去查源头 npz                                          |
| `pose_nan_frac >= RAW_NAN_MAX`（默认 0.01）         | Manus 丢帧多，姿态靠插值；`raw-qc` 默认把它挡在 `recordings_keep.csv` 外（阈值可调）          |
| HTML 画廊"no PNG"                                   | 该 clip 渲染失败；首选 skeleton 路径需 raw npz 含 `manus_*_skeleton`，否则走 FK 需要 torch     |
| 第一条 clip 报 `keyframe FK unavailable`            | 处理后数据无 skeleton 且无 torch；用 raw 数据，或装 torch                                      |
| `ModuleNotFoundError: sklearn`                      | 没走 pulse.sh（它锁定了 emg2pose conda python）；或用 `$PY` 显式指                              |
| `OMP: Error #34 ...`                                | 没走 pulse.sh；手动跑 cluster 加 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` |
| pool 里冒出 `<date>_<time>` 组名                    | 该批次文件名无 subject/hand 解析不了；换批次或删掉                                              |
| export 0 个                                         | 还没打标 / 标注 clip 全为空（导出按 `annotations` 的 `label`，没标=不导）；先 `./pulse.sh label`  |
| `no cluster runs in .../index.db`（eval/qc）        | 还没聚类；先 `./pulse.sh cluster` 或 `cluster-traj`                                            |
| 簇都长得一样                                        | 段太少 / k 太大；减 k 或加更多文件                                                              |
| 3D 手图右手 IndexError                              | emg2pose 右手 init bug；代码已走 left FK + 镜像，别改回 `side='right'`                          |

---

## 不想用 wrapper？（等价裸命令）

所有阶段都通过 `--out`（即 `index.db`）传递；聚类/评估/导出**不再吃位置参数 `work_pool`**，它们从 db 的 `source_path` 找 npz。

```bash
PY=/home/chenglin/anaconda3/envs/emg2pose/bin/python
cd /data/cl_data/action-clustering-compact

# 真值集路径
$PY segment.py --meta reference/sample_meta.csv --out out_pose   # 建 out_pose/index.db + 便利 CSV
$PY export_clips.py --out out_pose
$PY label_server.py --out out_pose --host 127.0.0.1 --port 8000  # Web UI 打标

# 聚类路径（切分后 index.db 已建好）
$PY segment.py --meta reference/sample_meta.csv --out out        # 或 dir / --recursive
# 打标：$PY label_server.py --out out
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY cluster.py --out out --k 18      # apex run -> cluster_runs/<run_id>/
$PY cluster_traj.py --out out --repr centered                    # （可选）trajectory run
$PY evaluate.py --out out                                        # 默认评估最新 run -> metrics.json；--run <id> 指定
$PY plot_cluster_features.py --out out                           # 最新 run 的 feature_maps.png；--run <id> 指定
$PY export.py --out out                                          # 按手势导出已标注 clip -> out/gestures/<label>/
$PY build_anim_gallery.py --out-root out --n 3 --clean           # 动画画廊（读 out/gestures/）

# 从 shards 重建索引：$PY segment.py --out out --index-only
```

---

## 单段可视化（细看某 clip / 某 burst）

```bash
PY=/home/chenglin/anaconda3/envs/emg2pose/bin/python

# 静态四面板（EMG / 包络 / 关节角 / 3D 手 start-apex-end）
$PY visualize_segment.py out_pose/clips_export/<stem>/c0000.npz -o /tmp/x.png

# 单段交互式 3D 动画（plotly html）
$PY animate_segment.py out_pose/clips_export/<stem>/c0000.npz
```
