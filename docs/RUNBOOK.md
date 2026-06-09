# RUNBOOK — 切分 / 打标 / 聚类 / 导出

一条工作流。**所有产物都落进同一个 SQLite 索引 `OUT/index.db`**（store.py 维护），并都以 **clip `(source_file, clip_id)`** 为唯一单元——打标、聚类、评估、导出全用这同一个键，所以"用打标评估聚类"只是一次 JOIN。

流程：`segment <源>`（双信号切分，自动识别数据形态）→ `label`（Web UI 人工打标，写 `annotations` 表）→ `cluster` / `cluster-traj`（KMeans，apex 或 trajectory 通道）→ `eval`（用人工标签评估聚类）→ `export`（按手势导出训练集）。
入口：`pulse.sh`（包装层），或单独跑 `segment.py / label_server.py / cluster.py / cluster_traj.py / evaluate.py / export.py`。

数据层结构：`OUT/shards/{stem}/` 是切分**真值**（每条录制一个 shard）；`OUT/index.db` 是从 shard 重建的派生索引（表：`recordings, bursts, clips, annotations, cluster_runs, cluster_assignments, tombstones`），其中 `annotations` 表是**标签真值**（不在 shard 里）。`segment.py` 跑完会顺带导出顶层便利 CSV `recordings.csv / bursts.csv / clips.csv`（给 diag_seg / cluster_traj / visualize_segment 这些读扁平文件的工具用，bursts/clips 里 JOIN 回了 `source_path`）。

原理见 [METHOD.md](METHOD.md)。

> `reference/gesture_velocity_segmentation.py` 只做**切分 + 可视化**，不做聚类。本项目借鉴它的双信号切分（pose-speed + clip）、数据源约定，并在其上加了 EMG 通路、交叉引用、`recordings.csv` QC、Web UI 打标。聚类是本项目自己加的，reference 没有。

---

## 0. 命令速查 + 环境变量

> 约定：在仓库根目录跑 `./pulse.sh <子命令>`（它内部锁定 `$PY` = emg2pose conda python）。覆盖配置就在命令前导出变量，如 `OUT=out_fgw K=24 ./pulse.sh cluster`。

### 0.1 主线速查

```bash
# 切分（写 OUT/index.db）—— 一个源参数，三种喂法
SUBJECTS=fgw-0917,jm-0503 ONLY_HAND=left WORKERS=32 ./pulse.sh segment processed   # 按人/按手选
DATES=20260502 ONLY_HAND=left WORKERS=32 ./pulse.sh segment processed               # 按时间选（某天）
DATE_FROM=20260501 DATE_TO=20260503 ./pulse.sh segment processed                    # 按时间选（区间）
WORKERS=8 ./pulse.sh segment /data/cl_data/ai-infra/processed_data/fgw0917_0502_left  # 指定批次目录（递归扫）
WORKERS=8 ./pulse.sh segment reference/sample_meta.csv               # meta CSV 驱动

# 打标（Web UI，写 index.db 的 annotations 表，读 OUT/index.db）
./pulse.sh label                                                     # 开 http://127.0.0.1:8000
WORKERS=8 ./pulse.sh label-prewarm                                   # 预渲染全部录制的手部缓存，首开秒级
WORKERS=8 ./pulse.sh label-prewarm --first-per-folder                # 只预热每个文件夹第一条（每个文件夹都能秒开）

# 聚类（两条通道，各产一个 OUT/cluster_runs/{run_id}/）。切分/打标用整库，
# 聚类可只选其中一个子集——同一套 segment 选择器 + SESSIONS=/RECORDINGS=
./pulse.sh cluster                                                   # apex 静态通道 (K=18, GROUP_BY=subject-hand)，全库
SUBJECTS=ghd-1108 SESSIONS=20260501-left ./pulse.sh cluster 17       # 只聚某人某 session（K=17）
K=auto GROUP_BY=all SUBJECT_NORM=zscore ./pulse.sh cluster           # 跨被试合聚 + 归一化 + 自动 k
SUBJECTS=ghd-1108 SESSIONS=20260501-left REPR=centered ./pulse.sh cluster-traj 17  # trajectory 时序通道，同子集

# 看/对比所有 run（数据 + 方法 + 指标 一张表）
./pulse.sh runs                                                      # -> 打印 + 写 OUT/cluster_runs/INDEX.csv

# 评估（标签驱动；写 cluster_runs/<run>/metrics.json）/ 导出（按手势）
./pulse.sh eval                                                      # 评估最新 run（ARI/NMI/purity + 被试污染 + pose 探针）
RUN=20260608-111819__apex__ghd1108_20260501-left__k17__06f66b4f ./pulse.sh eval   # 指定某个 run
./pulse.sh export                                                    # 已打标 clip -> OUT/gestures/<手势>/

# 辅助
./pulse.sh qc        # 某 run 特征图 + 画廊        ./pulse.sh gallery   # 每手势动画画廊（需先 export）
./pulse.sh db        # 从 shards 重建 index.db      ./pulse.sh status    # 看 index.db/cluster_runs/gestures 进度
./pulse.sh run <meta|dir|processed>   # 一键 segment+cluster+eval
```

### 0.2 输出根 OUT

`OUT`（默认 `out`）是**唯一**的输出根，所有命令都读写它：`index.db` + `shards/` + 便利 CSV + `cluster_runs/` + `gestures/` + 缓存全在这一个目录下。不同批次想互不覆盖就显式换名，如 `OUT=out_fgw ./pulse.sh segment processed`。`label` / `label-prewarm` 也读 `OUT/index.db`（直接 `./pulse.sh label` 即可）。

### 0.3 怎么选数据（无需建池）

直接给 `segment` 一个源参数，并用 `SUBJECTS=` 挑被试——**不需要先把 npz 汇成一个池**。`segment.py` 会把每条录制的 `source_path` 写进 `index.db`，下游聚类/评估/导出全靠它定位 npz，所以不存在中间池目录。

```bash
SUBJECTS=fgw-0917,jm-0503 ./pulse.sh segment processed   # 按 subject id 从 $DATA_ROOT 选
```

`SUBJECTS` 按**归一化 id** 匹配（`fgw-0917 == fgw0917`），meta CSV 与 processed_data 用法一致。

### 0.4 环境变量总表（按命令分组）

**全局（所有命令）**
| 变量 | 默认 | 含义 |
|------|------|------|
| `PY` | emg2pose conda python | 解释器（sklearn/torch 都在这个环境） |
| `DATA_ROOT` | `/data/cl_data/ai-infra/processed_data` | `segment processed` 扫描的批次根 |
| `OUT` | `out` | 唯一输出目录（`index.db` + `shards/` + 便利 CSV + `cluster_runs/` + `gestures/` + 缓存） |
| `OMP/MKL/OPENBLAS_NUM_THREADS` | 4 | 线程上限（KMeans 在本机会过度占线程，故钳住） |

**切分 `segment`**
| 变量 | 默认 | 含义 |
|------|------|------|
| `SUBJECTS` | 空 | **按人**：只处理这些被试（逗号分隔；归一化匹配 `fgw-0917`==`fgw0917`；CSV 与 processed 一致）。`SUBJECT`=旧的单值别名 |
| `ONLY_HAND` | 空 | **按手**：只处理 `left` 或 `right` |
| `DATES` | 空 | **按时间**：只处理日期（`YYYYMMDD`）以这些前缀开头的录制——可填某天 `20260502`、某月 `202605`、某年 `2026`，逗号分隔多个 |
| `DATE_FROM` / `DATE_TO` | 空 | **按时间（区间）**：保留日期在 `[DATE_FROM, DATE_TO]`（含端点，`YYYYMMDD`）内的录制；只给一端即开区间 |
| `WORKERS` | 1 | 并行进程数（每条录制一个；近线性提速到磁盘 IO 饱和） |
| `NO_OVERVIEW` | 0 | 设 1 跳过 `overview.png`（CSV-only，提速 ~3-5x，无 torch 时用） |
| `POSE_PCT`/`POSE_MAD`/`MIN_STATIC_S`/`MIN_MOTION_S` | 35/1.5/0.35/0.20 | pose-speed 运动阈派生参数（设了才透传 `--pose-pct ...` 等；调切分密度时降 `POSE_PCT`/`POSE_MAD` 即过切） |

**聚类 `cluster`（apex）/ `cluster-traj`（trajectory）**
| 变量 | 默认 | 用于 | 含义 |
|------|------|------|------|
| `K` | 18 | 两者 | 簇数；`K=auto` 走 silhouette 扫 `[k-min,k-max]` |
| `GROUP_BY` | subject-hand | cluster | `subject-hand`/`hand`/`all`（聚类粒度） |
| `SUBJECT_NORM` | none | cluster, eval | 按被试归一化 `none`/`center`/`zscore`（pooled 跑用） |
| `REPR` | centered | cluster-traj | 时序表征 `centered`/`velocity`/`raw` |

**评估/质检/一键 `eval` / `qc` / `gallery` / `run`**
| 变量 | 默认 | 含义 |
|------|------|------|
| `RUN` | 最新 | 评估/画哪个 `cluster_runs/{run_id}`（不设=最新一个 run） |
| `N_GALLERY` | 3 | 动画画廊每手势采样数 |

**打标 `label` / `label-prewarm`**
| 变量 | 默认 | 含义 |
|------|------|------|
| `PORT` / `HOST` | 8000 / 127.0.0.1 | Web UI 监听 |
| `OVERVIEW_REGEN` | 1 | 1=每次实时重画 overview（修过单位/轴上限后才正确）；0=用 shard 里的静态图（首开更快） |
| `WORKERS` | 4 | `label-prewarm` 的并行数 |

`label-prewarm` 透传 flag（接在子命令后）：`--first-per-folder`（只预热每个文件夹排序第一条，一条代表/folder，全部 folder 都能秒开）；`--subjects id,…`（只预热某些被试）；`--limit N`（只预热前 N 条）。不加 flag = 预热全部录制（数据大时很费盘，慎用：4093 条 ≈ 87 GB 缓存）。

---

## 1. 切分 `segment`

### 1.1 数据源三种喂法

`segment.py` **自动识别数据形态**，所以只有一个源参数：传一个 `meta.csv`、一个目录、或字面量 `processed`（= `$DATA_ROOT`）。

```bash
# (a) 处理后数据，扁平目录（文件名 {subject}__{date}-{hand}__{stamp}.npz）
./pulse.sh segment /data/cl_data/ai-infra/processed_data/fgw0917_0502_left

# (b) 原始数据树，按 {subject}/{date-hand}/{stamp}.npz 递归扫
./pulse.sh segment /mnt/pose_data/emg2pose/data

# (c) 用 sample_meta.csv 驱动（hand 从 meta 取，side=both 自动展两份）
./pulse.sh segment reference/sample_meta.csv

# 字面量 processed = 扫 $DATA_ROOT，配 SUBJECTS= 选被试
SUBJECTS=fgw-0917 ./pulse.sh segment processed
```

文件名格式自适应：处理后 `{subject}__{date}-{hand}__{stamp}.npz` 解析 hand；原始 `{stamp}.npz` 退到父目录 `{date-hand}` 取 hand。给目录时 wrapper 自动加 `--recursive`。

**三种数据选择（可叠加，AND 关系）**：按人 `SUBJECTS=fgw-0917,jm-0503`、按手 `ONLY_HAND=left|right`、按时间 `DATES=`（日期 `YYYYMMDD` 前缀：某天 `20260502` / 某月 `202605` / 某年 `2026`，逗号分隔多个）或 `DATE_FROM=…DATE_TO=…`（`YYYYMMDD` 闭区间）。日期从文件名里解析（session 段的 `YYYYMMDD` 或时间戳），meta CSV 有 `date` 列则优先用它。例：

```bash
SUBJECTS=fgw-0917 ONLY_HAND=left DATES=202605 WORKERS=32 ./pulse.sh segment processed
DATE_FROM=20260501 DATE_TO=20260503 ./pulse.sh segment processed
```

裸命令等价：

```bash
$PY segment.py --meta reference/sample_meta.csv --out out --workers 8     # meta 驱动
$PY segment.py <dir> --recursive --out out --workers 8                    # 目录（递归）
```

### 1.2 force_data 默认跳过

reference 明确只用 `/mnt/pose_data/`，不用 `/mnt/force_data/`（force 任务是等长收缩、姿态变化少，pose-speed 切出基本是噪声）。本流水线沿用：

```bash
# 默认：force_data 文件自动剔除
$PY segment.py --meta reference/sample_meta.csv --out out
# → SKIP 10175 force_data file(s) (pass --allow-force to include them)

# 强制要 force_data（不推荐）
$PY segment.py --meta reference/sample_meta.csv --out out --allow-force
```

### 1.3 关键 QC 列怎么用

`segment` 产物里 `recordings.csv`（每条录制一行）带一批 QC 列。这些列仍在 `recordings` 表里供临时查询；它们**不门控**任何下游步骤——切分把所有合格 clip 都写进 db，复核/排除靠 Web UI 的 invalid。

> 每一列（recordings / clips / bursts / annotations 四张表）+ 打标网页面板字段的逐列释义见 [字段说明.md](字段说明.md)。

| 列                                              | 怎么用                                                       |
|-------------------------------------------------|--------------------------------------------------------------|
| `lag_flag`                                      | 全局 EMG↔pose 互相关的诊断标签（`ok`/`early`/`late`/`nan`）。**已知在本类准周期数据上有偏**（持姿期 EMG 才达峰，互相关易把绝大多数判成 `ok`），只作诊断参考 |
| `emg_pose_lag_s, emg_pose_corr`                 | 上面 lag 的具体秒数与互相关置信度                            |
| `pose_nan_frac, emg_nan_frac`                   | NaN/插值占比；偏高说明 Manus 丢帧多、姿态靠插值，手动复核时关注 |
| `n_burst_only, n_clip_only`                     | 双信号分歧；clip_only 数高 = 大量 EMG 漏切的真手势           |
| `enter_thresh, exit_thresh, pose_thresh, pose_exit_thresh` | 阈值快照，复盘阈值是否合理                          |

`clips.csv`（打标单元）的关键过滤列，复核时按这些排序优先看：

| 列                                              | 典型阈值                                                     |
|-------------------------------------------------|--------------------------------------------------------------|
| `motion_duration_s`                             | < 0.25 s 通常是噪声毛刺                                      |
| `pose_range`                                    | < 0.5 rad ≈ 几乎没动                                         |
| `envelope_peak`                                 | < 5 ≈ EMG 没真的发力                                         |
| `matched_burst_idx == -1`                       | EMG 漏切 → **优先复核**（标注前先看这批）                    |

---

## 2. 打标 Web UI `label`

```bash
# 起本地服务（默认 127.0.0.1:8000），读 OUT/index.db
./pulse.sh label                       # PORT=8000 HOST=127.0.0.1 可覆盖
# 可选：先预渲染手部帧缓存，首开秒出
WORKERS=8 ./pulse.sh label-prewarm
```

`label_server.py` 从 `index.db` 读 clips / recordings / bursts，浏览器里逐 clip 打标，每条录制渲染一个可实时旋转的 3D 手。所有标签写回 **`annotations` 表**（不写任何 CSV）：

- 每条标签是一行 annotation，带 `scope`（`clip` / `recording`）+ `kind`（`label` / `invalid`），并携带 **重切分稳定锚点** `(clip_start_sample, clip_end_sample, seg_version)`——以后重切分时 `store.remap_annotations` 按样本区间重叠把旧标签迁到新 clip（重叠不足不会静默丢，标 `note='待复核'`）。
- **软 `invalid`**：可对单个 clip 或整条录制打 `invalid`（`scope='recording'`），被标的 clip / 录制从打标、聚类、评估、导出里排除，但**数据还在**，可撤销。
- **永久删除（"删除录制"按钮 / `POST /api/drop_recording`）**：与软 invalid **不同**——彻底删掉该录制的 db 行 + annotations，写一条 `tombstone`（重切分**不会复活**它），并删掉磁盘上的 shard。不可撤销。

> webui 资源在 `webui/index.html` + `webui/app.js`。3D 手用 plotly：优先本地 `webui/plotly.min.js`（离线可用，**未提交**），缺了回退 CDN。

打标约定：任意手势名（`fist` / `pinch_index` / `one` 等）；同名自动合并；留空（未打标）= 导出时丢弃该 clip。

裸命令等价：`$PY label_server.py --out out --host 127.0.0.1 --port 8000`。

---

## 3. 聚类 + 标签评估

clip 是唯一单元，标签和聚类共用键，所以评估是一次 JOIN。**没有"给每个簇起名再按簇导出"那一步**——导出按 `annotations` 的人工标签分组，不按簇。

### 3.1 TL;DR

```bash
cd /data/cl_data/action-clustering-compact

# 1) 切分（建 OUT/index.db + 便利 CSV）。源 = meta CSV / 目录 / 'processed'
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

### 3.2 子命令（与 pulse.sh case block 一致）

```
./pulse.sh segment <meta|dir|processed>  阶段 1 切分，建 OUT/index.db（自动识别数据形态）
./pulse.sh label                Web UI 打标（见 §2）
./pulse.sh label-prewarm        预渲染手部帧缓存（WORKERS=N）
./pulse.sh cluster      [K]     阶段 2 apex 聚类（K 默认 18；"auto" = silhouette 选 k）-> cluster_runs/
./pulse.sh cluster-traj [K]     阶段 2 trajectory（时序）聚类 -> cluster_runs/（REPR= 选表征）
./pulse.sh runs                 列出/对比所有 run（数据+方法+ARI/NMI/purity）-> cluster_runs/INDEX.csv
./pulse.sh eval                 标签驱动评估 + 被试无关性（RUN= 或最新）-> cluster_runs/<run>/metrics.json
./pulse.sh export               阶段 3 按手势导出标注 clip -> OUT/gestures/<label>/
./pulse.sh qc                   某 run 特征图（+ 已导出则带动画画廊）
./pulse.sh gallery              单独建动画画廊（export 之后）
./pulse.sh db                   从 shards 重建 OUT/index.db（+ 便利 CSV）
./pulse.sh run <meta|dir|processed>   一键 segment + apex cluster + eval
./pulse.sh status               查看进度（index.db / cluster_runs / gestures）
./pulse.sh help
```

### 3.2b 只对一部分数据聚类（多实验对比）

切分、打标始终用**整库**（一个共享 `OUT/index.db`）；聚类可以只取其中一个**子集**反复做实验，无需重切分/重打标。选择器词汇和 `segment` 完全一致，额外加 `SESSIONS=`（session token，前缀匹配，如 `20260501-left` 也命中 `20260501-left-3`）和 `RECORDINGS=`（精确 source_file，可省 `.npz`），apex / trajectory 通用：

```bash
# 同一个人、同一个 session：apex vs trajectory 对比
SUBJECTS=ghd-1108 SESSIONS=20260501-left ./pulse.sh cluster 17
SUBJECTS=ghd-1108 SESSIONS=20260501-left REPR=centered ./pulse.sh cluster-traj 17
./pulse.sh eval                 # 评最新 run；想评全部就对每个 RUN= 跑一次
./pulse.sh runs                 # 一张表看清每个 run 用了哪份数据 + 哪种方法 + 指标
```

每个 run 都**自带数据范围**：
- **run 目录名**可读，如 `20260608-111900__traj__ghd1108_20260501-left__k17__centered__<hash>`；
- `params.json` 里有完整 `scope`（筛选条件 + 实际命中的 clips/录制/被试/session/日期范围）；
- `./pulse.sh runs` 把所有 run 汇成 `cluster_runs/INDEX.csv`：`run_id / channel / data / n_clips / n_rec / n_subj / k / method / ARI / NMI / purity`。

选择器（聚类时）：`SUBJECTS=`（归一化匹配）、`ONLY_HAND=`、`DATES=`（前缀）、`DATE_FROM=`/`DATE_TO=`、`SESSIONS=`（前缀）、`RECORDINGS=`（精确）。留空 = 全库（tag=`all`）。

### 3.3 典型场景

**复刻 pilot（单被试左右手，一键）**：
```bash
SUBJECTS=fgw-0917 ./pulse.sh run processed   # segment + apex cluster + eval
./pulse.sh label                             # 打标
./pulse.sh eval                              # 有标签后重新评估
./pulse.sh export                            # 按手势导出
```

**扩到多被试**（每个 `(被试, 手)` 自动成一组）：
```bash
SUBJECTS=fgw-0917,wjh-0111 ./pulse.sh segment processed
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
GROUP_BY=all SUBJECT_NORM=zscore ./pulse.sh cluster   # 21 万 clip 合聚 ~几十秒（特征缓存命中）
```
> ⚠️ 不分组直接合聚时 KMeans 常按"谁的手"而非"什么手势"分。归一化 (`SUBJECT_NORM`) 压低这种污染，但效果有限；详见 [docs/汇报文档/多人单手聚类结果与分析.md](汇报文档/多人单手聚类结果与分析.md)。
> **大规模 pooled run（`GROUP_BY=all`，几十万 clip）注意**：`cluster` / `eval` 的 silhouette 是 O(n²)，对 1 万以上自动**子采样**（`sample_size=10000`，KMeans 仍用全量点聚，只有质量分用随机子样本估计），所以才能秒级跑完。但 **`qc`（`plot_cluster_features` 的 t-SNE + 逐样本 silhouette）尚未做大 N 适配**，对几十万 clip 的 pooled run 会很慢/吃内存——这种规模先别对它跑 `qc`（在小一点的 run 或 per-group 上看特征图）。

**对比归一化方式**（无需重切，换 `OUT` 各存一份）：
```bash
OUT=out_center  GROUP_BY=all SUBJECT_NORM=center ./pulse.sh cluster
OUT=out_zscore  GROUP_BY=all SUBJECT_NORM=zscore ./pulse.sh cluster
```

### 3.4 打标 + 评估（取代旧的"人工命名"）

不再"给簇起名"。流程是：

1. **打标**：`./pulse.sh label`，在 Web UI 里按 clip 打手势名 / 标 invalid（写 `annotations` 表，见 §2）。
2. **看聚类**：每个 run 的 `cluster_runs/<run_id>/gallery/`（apex 是每簇 3D 手姿 `<group>_hands.png`；trajectory 是每簇 medoid 关键帧条 + `index.html`）+ `feature_maps.png`（`./pulse.sh qc`，PCA/t-SNE/热图）。
3. **评估**：`./pulse.sh eval`（`RUN=` 指定 run，默认最新）把聚类 assignment 和 clip 标签 JOIN，算 ARI / NMI / purity + 每簇主导手势占比 + 被试污染 + pose 空间 silhouette/leakage/LOSO，写 `cluster_runs/<run>/metrics.json`。标签太少（<2 标注 clip 或 <2 类）时 ARI/NMI/purity = N/A，不会崩。
4. **导出**：`./pulse.sh export` 按手势名把**已标注**的 clip 分组导出（不按簇）。

### 3.5 产物

| 阶段    | 产物                                                                                          |
|---------|-----------------------------------------------------------------------------------------------|
| segment | `OUT/index.db`（权威索引）+ 每条录制一个 shard `shards/{stem}/`（`recording.csv` / `bursts.csv` / `clips.csv` / `overview.png`）+ 顶层便利 CSV `recordings.csv` / `bursts.csv` / `clips.csv` + `features/{stem}.npz` |
| label   | 写进 `index.db` 的 `annotations` 表（clip / recording 标签 + invalid + tombstone），不落 CSV |
| cluster / cluster-traj | `cluster_runs/<run_id>/{params.json, clusters.csv}` + `gallery/` + db 里的 `cluster_runs` / `cluster_assignments` 行（`run_id = {YYYYMMDD-HHMMSS}__{paramhash8}`） |
| eval    | `cluster_runs/<run>/metrics.json`（标签驱动 ARI/NMI/purity + 被试污染 + pose 空间探针）       |
| qc      | `cluster_runs/<run>/feature_maps.png`（PCA/t-SNE/热图）、`OUT/hand_anim/index.html`（动画画廊，读 `gestures/`） |
| export  | `OUT/gestures/<label>/<label>__<subject>-<hand>__<stem>__clip{cid:04d}.npz`、`OUT/labeled_overview/*.png` |

---

## 4. 常见踩坑

| 现象                                                | 原因 / 解决                                                                                    |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|
| `SKIP N force_data file(s)`                         | §1.2 默认行为，要 `--allow-force` 才包含（基本只在 debug 时用）                                |
| `lag=n/a (corr=0.00, nan)`                          | 该录制 EMG 或 pose 信号近常值/全 NaN；`recordings.csv.lag_flag=nan`，弃用                      |
| `lag=+0.500s (corr=0.20, late)`                     | EMG 领先 pose 超 400ms；可能是任务特性（持姿期 EMG 才达峰），看 `corr` 决定是否复核            |
| `lag=-0.21s (corr=0.30, early)`                     | EMG 滞后 pose；通常是预处理对齐 bug，需要去查源头 npz                                          |
| `pose_nan_frac` 偏高                                | Manus 丢帧多，姿态靠插值；该列是诊断量，手动复核时关注                                          |
| 3D 手图 overview 渲染失败                           | 首选 skeleton 路径需 raw npz 含 `manus_*_skeleton`，否则走 FK 需要 torch                       |
| `keyframe FK unavailable`                           | 处理后数据无 skeleton 且无 torch；用 raw 数据，或装 torch                                      |
| `ModuleNotFoundError: sklearn`                      | 没走 pulse.sh（它锁定了 emg2pose conda python）；或用 `$PY` 显式指                              |
| `OMP: Error #34 ...`                                | 没走 pulse.sh；手动跑 cluster 加 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` |
| 冒出 `<date>_<time>` 组名                           | 该批次文件名无 subject/hand 解析不了；换批次或删掉                                              |
| export 0 个                                         | 还没打标 / 标注 clip 全为空（导出按 `annotations` 的 `label`，没标=不导）；先 `./pulse.sh label`  |
| `no cluster runs in .../index.db`（eval/qc）        | 还没聚类；先 `./pulse.sh cluster` 或 `cluster-traj`                                            |
| 簇都长得一样                                        | 段太少 / k 太大；减 k 或加更多文件                                                              |
| 3D 手图右手 IndexError                              | emg2pose 右手 init bug；代码已走 left FK + 镜像，别改回 `side='right'`                          |

---

## 5. 不想用 wrapper？（等价裸命令）

所有阶段都通过 `--out`（即 `index.db`）串联；聚类/评估/导出**不吃位置参数**，它们从 db 的 `source_path` 找 npz。

```bash
PY=python   # 先 conda activate <你的环境>（需含 numpy/sklearn/torch），或指向具体解释器
cd /data/cl_data/action-clustering-compact

$PY segment.py --meta reference/sample_meta.csv --out out         # 建 out/index.db + 便利 CSV（或 dir + --recursive）
$PY label_server.py --out out --host 127.0.0.1 --port 8000        # Web UI 打标
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY cluster.py --out out --k 18   # apex run -> cluster_runs/<run_id>/
$PY cluster_traj.py --out out --repr centered                     # （可选）trajectory run
$PY evaluate.py --out out                                         # 默认评估最新 run -> metrics.json；--run <id> 指定
$PY plot_cluster_features.py --out out                            # 最新 run 的 feature_maps.png；--run <id> 指定
$PY export.py --out out                                           # 按手势导出已标注 clip -> out/gestures/<label>/
$PY build_anim_gallery.py --out-root out --n 3 --clean            # 动画画廊（读 out/gestures/）

# 从 shards 重建索引：$PY segment.py --out out --index-only
```

---

## 6. 单段可视化（细看某 clip / 某 burst）

`export` 产出的每个 clip npz（`OUT/gestures/<label>/...clip0000.npz`）可直接喂可视化工具：

```bash
PY=python   # 先 conda activate <你的环境>（需含 numpy/sklearn/torch），或指向具体解释器

# 静态四面板（16 通道 EMG / 包络 / 20 关节角 / 3D 手 start-apex-end）
$PY visualize_segment.py out/gestures/<label>/<...>clip0000.npz -o /tmp/x.png

# 单段交互式 3D 动画（plotly html）
$PY animate_segment.py out/gestures/<label>/<...>clip0000.npz
```
