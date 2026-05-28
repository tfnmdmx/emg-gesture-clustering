# 方法说明 — EMG 手势切分与聚类

本文档详细解释切分/聚类的实现原理与数据规模,以当前 **fgw0917** 这次运行(k=18)为例给出真实数字。
代码:[segmentation.py](../emg_label/segmentation.py) / [features.py](../emg_label/features.py) /
[clustering.py](../emg_label/clustering.py) / [cluster.py](../cluster.py)。

---

## 0. 输入数据形态

每个 `.npz`:

| 字段             | shape   | 含义                                                       |
| ---------------- | ------- | ---------------------------------------------------------- |
| `emg`          | (T, 16) | 16 通道表面肌电,已滤波(40–850Hz 带通 + 50Hz 陷波),2000 Hz |
| `joint_angles` | (T, 20) | 20 维关节角,**弧度**,已对齐到 EMG 时间轴             |

逐样点同步(同一行的 emg 与 joint_angles 是同一时刻)。

**当前 fgw0917（/data/cl_data/ai-infra/processed_data/fgw0917_0502_left /data/cl_data/ai-infra/processed_data/fgw0917_0504_right） 规模**:

- 240 个文件,每个约 **119 秒** = 238,024 样点。
- 原始样点总数 ≈ **5712 万** 行(57.1 M)。
- 分成 2 个组:`fgw-0917-left`(120 文件)、`fgw-0917-right`(120 文件)。

---

## 1. 16 通道包络是怎么算的

切分不直接看 16 路原始信号,而是先把它们**压成一条一维"用力强度"曲线**,叫包络(envelope)。

### 1.1 公式

代码 [segmentation.py:7 `emg_envelope`](../emg_label/segmentation.py#L7):

```python
rect     = np.abs(emg).mean(axis=1)              # 步骤①②: 整流 + 跨通道平均 → (T,)
win      = round(smooth_ms/1000 * fs)            # 150ms × 2000Hz = 300 样点
activity = uniform_filter1d(rect, size=win)      # 步骤③: 滑动平均平滑 → (T,)
```

**三步**:

1. **整流(rectify)**:`abs(emg)`。
   原始 EMG 是围绕 0 上下振荡的交流信号(正负相间),直接平均会相互抵消≈0。取绝对值后,信号幅度变成恒正,反映"这一刻肌肉放电多强"。
2. **跨 16 通道平均**:`.mean(axis=1)`,`(T,16) → (T,)`。
   16 个电极贴在前臂不同位置,每路对不同肌群敏感。做某手势时通常多路同时活跃。取 16 路均值 = 把"整条前臂的总用力强度"汇成一条曲线。

   - 用**均值**而非求和,数值不随通道数变化;用全 16 路而非挑几路,是因为手势顺序未知、不知道哪几路重要,全用最稳妥。
3. **平滑(滑动平均)**:`uniform_filter1d(win=300)`。
   整流后的曲线仍然毛刺很多(肌电是随机放电脉冲串)。用 150ms 窗口(=300 样点)做滑动平均,把高频毛刺抹平,留下"动作整体起伏"的慢包络。

   - 150ms 是经验值:足够压住毛刺,又不会把 0.4s 量级的动作糊掉。`--smooth-ms` 可调。

得到的 `activity` 曲线:做手势时隆起成峰,放松时回落到基线 —— 这就是 `overview/*.png` 下半张那条黑线。

### 1.2 为什么这条曲线能用来切分

关键性质:**肌肉一放松,包络就回到基线**,跟手当时摆成什么姿势无关。
所以即使两个手势之间手没回到中性(比如从"比1"直接变"比2"),只要中间有一瞬放松,包络就会下凹,把两个动作分开。

---

## 2. 切分原理 —— 双阈值迟滞检测

目标:在 `activity` 曲线上找出每一段"动作"的 `[起点, 终点]`。

### 2.1 自动定阈值(Otsu)

代码 [segmentation.py:39 `auto_thresholds`](../emg_label/segmentation.py#L39):

把整条 `activity` 的取值画成直方图,它天然是**双峰**的——一大堆低值(静息)、一小撮高值(动作)。
**Otsu 法**自动在两峰之间找最佳分割谷 `t`(最大化类间方差),不用人手调阈值,也不在乎录制里动作占多大比例。

然后定**两个**阈值:

```python
enter = t                                   # 进入阈(高)
exit  = rest_center + 0.5*(t - rest_center) # 退出阈(低,在静息中位数和谷之间)
```

### 2.2 迟滞(hysteresis)扫描

代码 [segmentation.py:55 `hysteresis_segments`](../emg_label/segmentation.py#L55):一个状态机,逐样点扫:

```
不在动作中, 且 activity ≥ enter  →  开始一段, 记 start
在动作中,   且 activity < exit   →  结束这段, 收 (start, i)
```

**为什么要两个阈值(而不是一个)?**
若只用单阈值,信号在阈值附近抖动时会被反复"跨上跨下",一个动作被切成好几段。
迟滞:**进入要够高(enter)、退出要够低(exit)**,中间这段"缓冲区"里的抖动不会触发状态切换 → 一个动作=一段。这跟空调温控、施密特触发器是同一个原理。

### 2.3 后处理过滤

代码 [segmentation.py:72 `filter_segments`](../emg_label/segmentation.py#L72):

- **合并**:相邻两段间隔 < `min_rest_gap_s`(0.2s)→ 视为同一动作中途的短暂下凹,合并。
- **丢弃**:段长 < `min_action_s`(0.4s)→ 视为噪声毛刺,删掉。

### 2.4 为什么不用关节角度切分(重要)

另一个直觉方案:算"关节角到中性姿态的距离",距离大=在做动作。**这条路实测失败**:
手势之间手往往不回中性(连续做不同手势),关节角距离一直很大,会把多个连续手势粘成一个超长段(实测最长 26 秒)。
而 EMG 在每次放松时**一定**回基线,所以 EMG 包络能干净分隔。这是定下来不推翻的核心决策。

### 2.5 当前 fgw 切分结果

- 共切出 **6327 段**。
- `fgw-0917-left`:3612 段,平均 30.1 段/文件(18–38)。
- `fgw-0917-right`:2715 段,平均 22.6 段/文件(1–40)。
- 段时长:中位 0.96s,均值 1.18s,范围 0.4–6.8s。

---

## 3. 聚类

### 3.1 有监督还是无监督?——**完全无监督**

整个流程**没有任何标签参与训练**:

- 切分:纯信号处理(阈值检测),无学习。
- 聚类:**KMeans,无监督**。它只看"哪些段的姿态彼此像",自动把相似的归成一堆,**不知道任何手势的名字**。
- 选 k:用 silhouette(轮廓系数)这个**内部指标**(只衡量簇内紧/簇间散,不需要真值),也不是监督。

**唯一的人工介入在聚类之后**:人看每一簇的代表姿态(3D 手图),给簇**起名字**(`labels.csv`)。这是"事后命名",不是"事前训练标签"。
所以严格说这是 **无监督聚类 + 人工事后标注** 的半自动流程,不是分类器训练。

### 3.2 聚类用的不是原始信号,是"姿态特征"

每段不是直接拿 (段长×20) 的矩阵去聚——长度不一、且含移动过程。而是先压成**一个 20 维向量**,代表"这次手势定型时的姿态"。

代码 [features.py:7 `apex_pose_feature`](../emg_label/features.py#L7):

```python
# 保持窗口 = [本段 start, 下一段 start)   ← 段后的低肌电"保持期"
seg  = joint_angles[start : next_start]
dev  = 平滑( ‖seg - rest‖ )               # 每帧偏离中性姿态多远
apex = argmax(dev)                         # 偏离最大那帧 = 姿态最到位的瞬间
feature = median(seg[apex ± 50ms])         # 该帧附近 ±50ms 的中位姿态 → (20,)
```

- `rest = median(整段录制的 joint_angles)` ≈ 中性手姿态。
- **为什么取"段后保持期"而非段内?** EMG 段是"肌肉爆发=手正在移动到位"的过程;真正能区分手势的定型姿态出现在爆发**之后**的低肌电保持期。实测约 67% 手势的关节偏离峰值落在 EMG 段之后。
- **为什么取 median 而非单帧?** 抗抖动/抗个别坏帧。
- **这个特征只用原始 joint_angles,与 FK/emg2pose 无关** → 换精确 FK 不改变聚类结果。

### 3.3 是把所有切片一起聚,还是分块?——**分块(按"被试-手"分组),不是全局一锅**

这是你问的重点。**不是把 6327 段全丢进一个 KMeans**,而是:

```
按 group = "subject-hand" 分组,每组各自独立跑一次 KMeans
```

当前 fgw 分成 **2 组,各跑一次**:

| 组             | 段数 → 特征矩阵 X | KMeans         |
| -------------- | ------------------ | -------------- |
| fgw-0917-left  | (3612, 20)         | 独立聚成 18 簇 |
| fgw-0917-right | (2715, 20)         | 独立聚成 18 簇 |

**为什么分组而不混聚?**
不同被试的手型/关节标定不同;左右手是镜像关系。把它们混在一起,KMeans 会先按"谁的手/哪只手"分开,而不是按"什么手势"分开。按 (被试,手) 隔离后,组内才是同一只手做不同手势,聚类才在比手势。
代码 [cluster.py:32](../cluster.py#L32) 的 `for group, gdf in seg_df.groupby("group")` 就是这个分块循环。

> 多被试时,你这次的 240 文件→2 组;若把更多被试 pool 进来,会自动变成更多组,每组各自聚类。`--k 18` 目前对所有组用同一个 k。

### 3.4 KMeans + 选 k

代码 [clustering.py:8 `select_k_and_cluster`](../emg_label/clustering.py#L8):

```python
Xz = zscore(X)                              # 20 维各自标准化(去量纲)
# 你传了 --k 18 → 直接 k=18;不传则在 [k_min,k_max] 扫 silhouette 自动选
labels = KMeans(n_clusters=18, n_init=10).fit_predict(Xz)
```

- **zscore**:20 个关节角量程不同(有的关节活动范围大、有的小)。不标准化的话 KMeans 的欧氏距离会被大量程关节主导。标准化后每维等权。
- **KMeans 干什么**:在 20 维空间里找 18 个簇心,把每段分到最近的簇心,反复迭代到稳定。`n_init=10` 跑 10 个随机初值取最好,避免落到坏的局部最优。
- **k 怎么定**:你这次手动定 18。不指定时用 **silhouette** 在 12–30 范围逐个 k 试、取分数最高的(分数 = 簇内紧致且簇间分离的程度)。
  - **为什么 k 宁大勿小**:过度聚类可恢复(同一手势被拆成两簇 → 标注时填相同名字就合并);欠聚类不可恢复(两个不同手势并进一簇 → 没法再分开)。实测姿态可分性上限约 17 簇,真实手势约 24 个,部分姿态太相似。

### 3.5 聚类后产物

- 每簇算 20 维质心 `centroid` + 段数 `count`。
- `clusters/<group>_hands.png`:把质心喂 **emg2pose 正向运动学** 渲染成 3D 手骨架(供人命名;FK 只用于画图,不进聚类)。
- `labels_template.csv`:每 `(group, cluster_id)` 一行,等人填 `label`。

**当前 fgw 聚类结果**:

- fgw-0917-left:18 簇,簇大小 59–364(均值 201)。
- fgw-0917-right:18 簇,簇大小 73–234(均值 151)。

---

## 4. 数据规模流转一览(fgw0917, k=18)

```
240 文件 × 238k 样点      原始: ~5712 万行 (emg 16ch + joint 20维)
        │  切分(每文件独立)
        ▼
6327 段                   每段一个 [start,end];仅记录索引,不复制数据
        │  特征(每段 → apex 保持姿态)
        ▼
2 个特征矩阵              left (3612,20) / right (2715,20)
        │  分组 KMeans(每组独立, k=18)
        ▼
2×18 = 36 簇             每簇一个 20 维质心
        │  人工命名 + 导出(每段切片落盘)
        ▼
36 个标签目录, 6327 npz   每个 = 一段 emg/joint_angles 切片 + 溯源元数据
```

---

## 5. 聚类之后:质量检查(特征图)

聚类是无监督的,命名前先确认"这 k 个簇是否真的可分、k 选得是否合理"。
工具:[plot_cluster_features.py](../plot_cluster_features.py),用**与 cluster.py 完全一致**的 apex 特征,逐组画四面板图到 `<out>/feature_maps/<group>_features.png`。

```bash
OMP_NUM_THREADS=4 python plot_cluster_features.py work_pool_fgw --out out_fgw
# 特征缓存到 out_fgw/feature_cache/<group>.npz,重跑秒出;改 k 后加 --force;嫌慢加 --no-tsne
```

四个面板:

1. **PCA 2D 散点**:20 维线性降到 2D,点=段、色=簇、星=质心。标题里的百分比是前两个主成分占的方差;若只占 ~57%,平面上簇会重叠——是投影损失,不代表不可分。
2. **t-SNE 2D 散点**:非线性降维,**更能反映真实分离度**。簇若一团团分开,说明在原 20 维空间确实可分。
3. **质心热图**(k 簇 × 20 关节,z-score):红=该关节比平均更伸、蓝=更屈。两行相似 = 两簇姿态接近(可能该合并);某列在各簇间反差大 = 该关节是区分手势的主力。
4. **簇大小 + 轮廓系数**:彩柱=样本数,灰柱=该簇平均 silhouette(越高越紧致独立;红虚线=0,负值说明该簇样本更靠近别的簇)。

**关键数字 — 整体 silhouette**:衡量整体聚类质量(-1~1,>0.3 算有合理结构)。
当前 k=18:`fgw-0917-left=0.389`、`fgw-0917-right=0.374`。
**怎么用它选 k**:改 `--k` 重跑 cluster.py 再跑本脚本,看整体 silhouette 升降;但记住"欠聚类不可恢复",所以即使大 k 的 silhouette 略低,也宁可偏大(靠命名时合并)。

---

## 6. 人工标注:给簇起名(labels.csv)

这是流程里**唯一**的人工环节,也是无监督聚类转成"带手势名数据集"的关键一步。

### 6.1 模板与填写

cluster.py 已生成 `out/labels_template.csv`,每 `(group, cluster_id)` 一行:

```
group,cluster_id,count,label
fgw-0917-left,0,281,          ← label 列待填
fgw-0917-left,1,292,
...
```

操作:

```bash
cp out_fgw/labels_template.csv out_fgw/labels.csv
# 编辑 labels.csv,只改 label 列;不要动 group/cluster_id/count
```

### 6.2 看什么来决定名字

对照三类可视化(同一 `cluster_id` 在三处一一对应):

- `out/clusters/<group>_hands.png` — 每簇质心的 3D 手姿态(**主依据**)。
- `out/clusters/<group>.png` — 每簇质心的 20 维条形图。
- `out/hand_anim/index.html` — 每簇前几个样本的动画(判断簇内是否一致,见 §8)。

### 6.3 命名语义(三条规则,由 export.py 决定)

代码 [export.py:35](../export.py#L35) 把 `(group, cluster_id) → label` 建成映射,导出时按 label 归档:

1. **起任意名**:`fist`/`pinch_index`/`one`/`open` 等,自由。
2. **同名合并**:同一手势被拆成多个簇 → 多行填**相同**名字 → 导出时落进同一目录,自动合并。这就是"过聚类可恢复"的兑现方式。
3. **空白丢弃**:label 留空的簇,导出阶段**跳过**(代码里 `if not label: continue`)。可先只标确定的,存疑簇留空,之后再补。

> 注意:命名是**事后**给簇贴名,不是给模型喂监督标签;整个聚类没用到任何标签。

---

## 7. 导出:落地带标签的单段数据集(export.py)

```bash
python export.py work_pool_fgw --out out_fgw          # 默认读 out_fgw/labels.csv
# 或 --labels /path/to/labels.csv
```

### 7.1 做了什么

代码 [export.py:45](../export.py#L45) 按源文件逐个处理(用完即释放,内存恒定):

```python
for fname in 源文件:
    for 该文件每段:
        label = lab_map[(group, cluster_id)]
        if not label: continue                  # 空标签跳过
        np.savez(segments/<label>/<命名>.npz,
                 emg[s:e], joint_angles[s:e],   # 切片 = EMG 爆发段 [s,e)
                 label, source_file, group, seg_idx,
                 start_sample, end_sample, fs, cluster_id)
    画 labeled_overview/<file>.png               # 只画已标注的段
```

### 7.2 产物结构

```
out_fgw/
├── segments/<label>/                            训练用数据集
│   └── <label>__<subj>-<hand>__<原stem>__seg<i>.npz
└── labeled_overview/<stem>.png                  每源文件一张核对图(仅画已标注段)
```

单个 npz 内容:

| 键                                                                         | 内容                                                |
| -------------------------------------------------------------------------- | --------------------------------------------------- |
| `emg`                                                                    | (seg_len,16) 该段 EMG 切片                          |
| `joint_angles`                                                           | (seg_len,20) 同长关节角(弧度)                       |
| `label` / `cluster_id` / `group`                                     | 手势名 / 原簇号 / 被试-手                           |
| `source_file` / `seg_idx` / `start_sample` / `end_sample` / `fs` | 完整溯源:来自哪个原始文件的哪一段、对应原始样点区间 |

文件名本身就编码了 `标签/被试-手/源文件/段号`,不读内容也能溯源。

### 7.3 切片范围(重要约定)

导出的是 **EMG 爆发段 `[start, end)`**(overview 里的绿色span),**不含**后面的保持期。

- 这是先前确认的范围定义。
- 聚类特征虽然在"爆发段 + 后续保持期"窗口里取 apex,但**导出只切爆发段**——两者范围不同,别混淆。
- 若训练需要保持期姿态,改 [export.py](../export.py) 里的 `s,e` 切片范围(把 `e` 往后延到下一段起点)。

### 7.4 自检

- 终端末行 `Exported N labeled segments`。**N=0** → labels.csv 的 label 列全空,或表头/group/cluster_id 对不上 segments_clustered.csv。
- `ls out_fgw/segments/` 看手势目录;`ls out_fgw/segments/<label> | wc -l` 是该手势样本数。
- 抽查 `labeled_overview/*.png`:标签文字应压在对应 EMG 爆发上。

---

## 8. 可视化核对工具(命名前/后都可用)

三个脚本,从粗到细核对切分与聚类质量。

### 8.1 单段静态图 — [visualize_segment.py](../visualize_segment.py)

一段 npz → 一张 PNG,四面板:16 通道 EMG(堆叠)、EMG 包络、20 维关节角、3 帧 3D 手姿(start/apex/end)。

```bash
python visualize_segment.py out_fgw/segments/left-0/<某.npz> -o /tmp/x.png
```

用途:细看某一段的信号与姿态对不对。注意它的 apex 只在爆发段内搜,与聚类的 apex(含保持期)不同(§3.2)。

### 8.2 单段 3D 动画 — [animate_segment.py](../animate_segment.py)

一段 npz → 交互式 plotly HTML(可旋转/缩放/拖时间轴/播放),看手势成型。
输出路径:`<out_root>/<subdir>/<label>/<stem>.html`(out_root 自动从 npz 路径推导,subdir 默认 `hand_anim`)。

```bash
python animate_segment.py out_fgw/segments/left-0/<某.npz> --subdir hand_anim --max-frames 80 --fps 30
```

五指按色区分,坐标轴固定成立方体(等比例)。2540 帧默认下采样到 80 帧。

### 8.3 批量画廊 + 索引页 — [build_anim_gallery.py](../build_anim_gallery.py)

对每个标签取前 N 段各生成动画,并写一个总索引 `<out>/hand_anim/index.html`(按组分表,每类列出 N 个样本链接 + 时长 + 来源)。

```bash
python build_anim_gallery.py --out-root out_fgw --n 3 --clean
# 打开 out_fgw/hand_anim/index.html 逐类核对
```

**核对要点**:同一类的 N 个样本动画应是同一个手势。若某类 N 个差别大 → 该簇不纯(k 可能偏小,或该手势姿态分散),提示标注时需谨慎或回头调 k。

---

## 9. 完整工作流与迭代

```
segment.py ──► cluster.py ──► plot_cluster_features.py   (看 silhouette / t-SNE 判 k)
                   │                build_anim_gallery.py  (逐类看动画判簇纯度)
                   ▼
            〔人工填 labels.csv:命名 / 同名合并 / 空白丢弃〕
                   ▼
              export.py ──► segments/<label>/*.npz + labeled_overview/*.png
```

**迭代准则**:

- 切分不满意 → 调 segment.py 的阈值/时长参数,重跑全链。
- 簇不满意(太碎/太糊)→ 改 cluster.py 的 `--k` 重跑 cluster + 两个检查脚本;`segments.csv` 不变无需重切。
- k 宁大勿小:欠聚类(两手势并一簇)不可恢复;过聚类靠命名时同名合并即可恢复。
- 命名只影响 export,改 labels.csv 后单独重跑 export.py 即可。
