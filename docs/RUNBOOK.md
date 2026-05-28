# RUNBOOK — 切分/聚类/导出

全流程现在收敛到**一个入口** [pulse.sh](pulse.sh)。它替你处理:conda python 路径、OMP 线程上限、建池软链、各阶段串联。
不用再背长命令。原理见 [docs/METHOD.md](docs/METHOD.md),设计见 [HANDOFF.md](HANDOFF.md)。

---

K=18 OUT=out_4user POOL=work_pool_4user ./pulse.sh prep fgw0917_0502_left ghd1108_0503_left hzy1217_0503_left lsh0126_0503_left


NAME=4users K=18 GROUP_BY=all ./pulse.sh prep fgw0917_0502_left ghd1108_0503_left hzy1217_0503_left lsh0126_0503_left


NAME=4users K=18 GROUP_BY=all ./pulse.sh prep fgw0917_0502_left ghd1108_0503_left hzy1217_0503_left lsh0126_0503_left

## TL;DR(最常用)

```bash
cd /home/chenglin/FM_PULSE/spilt

# 1) 一条命令:建池 + 切分 + 聚类(k=18) + 质检图
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right

# 2) 人工:看图命名
#    - 打开 out/clusters/*_hands.png 和 out/hand_anim/index.html
#    - cp out/labels_template.csv out/labels.csv,填 label 列
#      (多行同名=合并;留空=丢弃)

# 3) 导出
./pulse.sh export
```

就这三步。下面是细节与可调项。

---

## 子命令一览

```
./pulse.sh prep   <批次...>    建池+切分+聚类+质检,一步到位(标注前的全部)
./pulse.sh pool   <批次...>    只建池(软链 + 打印分组)
./pulse.sh segment             阶段1 切分
./pulse.sh cluster [K]         阶段2 聚类(K 默认 18;写 auto = silhouette 自动选)
./pulse.sh qc                  质检:特征图 + 3D 动画画廊
./pulse.sh export              阶段3 导出带标签 npz
./pulse.sh status              看当前进度(池/各产物存在与否)
./pulse.sh help
```

`<批次>` 可以直接写 `processed_data/` 下的目录名(如 `fgw0917_0502_left`),也可以写完整路径。

---

## 可调项(改默认值,临时 export 即可)

| 变量          | 默认                                      | 含义                                       |
| ------------- | ----------------------------------------- | ------------------------------------------ |
| `NAME`      | (空)                                      | 一键派生池/输出名(见下)                    |
| `K`         | 18                                        | 聚类簇数                                   |
| `GROUP_BY`  | `subject-hand`                          | 聚类粒度:`subject-hand`/`hand`/`all` |
| `OUT`       | `out`                                   | 输出目录                                   |
| `POOL`      | `work_pool`                             | 池目录                                     |
| `N_GALLERY` | 3                                         | 动画画廊每类样本数                         |
| `DATA_ROOT` | `/data/cl_data/ai-infra/processed_data` | 批次根目录                                 |
| `PY`        | emg2pose conda python                     | 解释器                                     |

**池名/输出名怎么来的**:`POOL`/`OUT` 是**固定默认值** `work_pool`/`out`,**不**根据数据自动起名。
所以不同批次若都用默认值会互相覆盖。两种避免方式:

- 显式设:`POOL=work_pool_fgw OUT=out_fgw ./pulse.sh prep ...`
- 用 `NAME` 一键派生:`NAME=fgw ./pulse.sh prep ...` → 自动 `POOL=work_pool_fgw`、`OUT=out_fgw`(显式 POOL/OUT 优先于 NAME)。

例子:

```bash
NAME=fgw ./pulse.sh prep fgw0917_0502_left fgw0917_0504_right   # 池/输出都带 fgw 后缀
K=24 OUT=out_k24 ./pulse.sh cluster        # 换 k 跑到另一个输出目录
OUT=out_k24 ./pulse.sh qc                  # 对 k=24 出质检图
./pulse.sh cluster auto                    # 让 silhouette 自动选 k(多被试时更合理)
GROUP_BY=all ./pulse.sh cluster 18         # 所有数据合到一起聚类(见场景 E)
```

---

## 典型场景

### A. 复刻 pilot(单被试左右手)

```bash
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right
# 标注后:
./pulse.sh export
```

### B. 扩到多被试

直接把更多批次名加到 `prep` 后面;每个 `(被试,手)` 自动成一组、各自聚类:

```bash
./pulse.sh prep fgw0917_0502_left fgw0917_0504_right \
                wjh0111_0502_left wjh0111_0503_right
```

> 文件越多越慢(切分 ~1s/文件)。多被试建议用 `./pulse.sh cluster auto` 让每组自动选 k。
> 选批次:`ls $DATA_ROOT`;看某批次有效文件数:`ls $DATA_ROOT/<批次>/*__*__*.npz | wc -l`。

### C. 只想换 k 重看

`segments.csv` 不依赖 k,不用重切:

```bash
./pulse.sh cluster 24      # 重聚类
./pulse.sh qc              # 重出质检图
# 重新命名 labels.csv 后再 export
```

### D. 中途看进度

```bash
./pulse.sh status
```

### E. 对所有数据一起聚类(不分被试/手)

默认是**每个 (被试,手) 各跑一次 KMeans**(`GROUP_BY=subject-hand`)。想换粒度:

```bash
GROUP_BY=all  ./pulse.sh cluster 18    # 全部 segment 合成一组,跑一次 KMeans
GROUP_BY=hand ./pulse.sh cluster 18    # 只按左/右手分两组(跨被试合池)
```

整条链也能带:`GROUP_BY=all ./pulse.sh prep <批次...>`。

> ⚠️ **为什么默认不这么做**:不同被试手型/关节标定不同、左右手是镜像。合在一起聚类,KMeans 往往**先按"谁的手/哪只手"分**,而不是按手势分——簇可能变成"某人的手"而非"某个手势"。
> 全局聚类时:`labels_template.csv` 的 group 列会是 `all`(或 `left`/`right`);3D 手图在 `all` 模式下统一按左手渲染(右手簇会镜像不准,只作粗看)。先小规模看 `feature_maps` 的 silhouette 再决定是否当真用。

---

## 人工标注(唯一的人工环节)

`prep` 跑完后:

1. **看图决定名字**:

   - `out/clusters/<group>_hands.png` — 每簇的 3D 手姿态(主依据)
   - `out/hand_anim/index.html` — 每类前 N 个样本的动画(判断簇内是否一致)
2. **填表**:

   ```bash
   cp out/labels_template.csv out/labels.csv
   ```

   编辑 `out/labels.csv`,**只改 `label` 列**:- 起任意手势名(`fist` / `pinch_index` / `one` …)

   - **多行同名 → 合并**为一个手势(过聚类可恢复)
   - **留空 → 丢弃**该簇
3. `./pulse.sh export`

> **缺 labels.csv 会怎样**:直接 `./pulse.sh export` 而 `labels.csv` 不存在时,它**不会**用空模板(那样会导出 0 段),而是自动生成占位名 `label = <group>-<cluster_id>`(如 `fgw-0917-left-0`),即"每簇当成一个手势、按簇原样导出"——所有段都会落盘,方便先看结果,之后再改名/合并重导。
> (注意:`labels_template.csv` 的 label 列本身是**空**的;`cp` 模板后必须自己填,否则 export 0 段。占位名是 export 子命令现填的,不是模板自带的。)

---

## 产物位置

| 阶段    | 产物                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------ |
| segment | `out/segments.csv`、`out/overview/*.png`(EMG+包络+绿色段)                                                |
| cluster | `out/segments_clustered.csv`、`out/labels_template.csv`、`out/clusters/<group>{,_hands}.png`           |
| qc      | `out/feature_maps/<group>_features.png`(PCA/t-SNE/热图/silhouette)、`out/hand_anim/index.html`(动画画廊) |
| export  | `out/segments/<label>/*.npz`、`out/labeled_overview/*.png`                                               |

各 npz 字段、切片范围(=EMG 爆发段,不含保持期)等见 [docs/METHOD.md §7](docs/METHOD.md)。

---

## 单段可视化(可选,细看某一段)

```bash
PY=/home/chenglin/anaconda3/envs/emg2pose/bin/python
# 静态四面板(EMG/包络/关节角/3D手 start-apex-end)
$PY visualize_segment.py out/segments/<label>/<某.npz> -o /tmp/x.png
# 单段交互式 3D 动画(plotly html)
$PY animate_segment.py   out/segments/<label>/<某.npz>
```

---

## 常见踩坑

| 现象                                   | 原因 / 解决                                                                                                            |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: sklearn`       | 没走 pulse.sh(它已锁定正确的 emg2pose python)。或手动用 `$PY`                                                        |
| `OMP: Error #34 ...`                 | 没走 pulse.sh(它已设线程上限)。手动跑 cluster 要 `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` |
| pool 里冒出 `<date>_<time>` 那种组名 | 该批次文件名无 subject/hand,解析不了;pool 会告警,换批次或删掉                                                          |
| export 0 个                            | `labels.csv` 的 label 列全空,或表头被改坏                                                                            |
| 簇都长得一样                           | 段太少 / k 太大;减 k 或加更多文件                                                                                      |
| 3D 手图右手 IndexError                 | emg2pose 右手 init bug;代码已走 left FK+镜像,别改回 `side='right'`                                                   |

---

## 不想用 wrapper?(等价裸命令)

```bash
PY=/home/chenglin/anaconda3/envs/emg2pose/bin/python
cd /home/chenglin/FM_PULSE/spilt
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
