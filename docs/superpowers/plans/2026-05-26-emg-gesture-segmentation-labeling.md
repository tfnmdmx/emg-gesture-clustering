# EMG 手势数据 动作切分与标注 实现计划

> ⚠️ **历史归档 — 已被取代,勿照此实现。** 这是项目最初的 TDD 实现计划,与当前代码已严重不符,仅作历史记录保留。具体偏差:
> - **切分**:本计划的 `segmentation.estimate_rest_baseline / activity_signal / auto_thresholds / segment_file`(关节角度偏离静息基线的迟滞分割)**已不是 live 路径**。当前 `segment.py`(`segment_recording` @ `emg_label/action_segmentation.py:280`)以 EMG 包络(`segmentation.emg_envelope` + `segment_emg`)为骨架做切分,关节角速度(pose-speed)迟滞 + EMG 分割并段。`segmentation.py` 现仅含 `emg_envelope / auto_thresholds / hysteresis_segments / filter_segments / hold_windows / segment_emg`,不含 `estimate_rest_baseline / activity_signal / segment_file`。
> - **Config**:本计划的 `plateau_frac / baseline_iters / baseline_keep_frac` 字段已从 `emg_label/config.py` 删除。
> - **特征**:本计划的 `features.plateau_feature`(段中部 50% 中位数)已被 `features.apex_pose_feature`(保持窗口顶点附近中位数)取代;`features.py` 还新增了共享提取器 `feature_by_seg`、`apex_index`、`per_subject_center/zscore`。
> - **聚类落地**:live 聚类是 `cluster.py`(聚 apex 静态姿态特征,经 `pulse.sh cluster` 调用);`cluster_traj.py`(轨迹特征)是独立实验脚本,未接入 `pulse.sh`。
> - **CSV / 产物**:本计划的 `segments.csv` 列(`start_sample/end_sample`)与 `labels.csv` 命名与当前不符,详见已实现的设计文档 `../specs/2026-05-26-emg-gesture-segmentation-labeling-design.md` 第 0 节及 `docs/METHOD.md`。
>
> 下文 Task 中的具体函数签名、Config 字段、CLI 代码块均为**当时**的设计草稿,不代表当前实现。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从一批 npz(动作-静息交替)中自动切分动作段、按 (被试,手) 分组聚类、人工批量命名后,把每个动作段导出为带标签的单独 npz 并出可视化图。

**Architecture:** 三阶段 CLI(segment / cluster / export)+ 一个职责单一模块的包 `emg_label`(config / io_utils / segmentation / features / clustering / plotting)。阶段间用 CSV 衔接,各阶段可独立重跑。聚类只用关节角度姿态特征;EMG 仅用于输出切片与可视化。

**Tech Stack:** Python 3.13、numpy 2.4、scipy 1.17、scikit-learn 1.8、matplotlib 3.10、pandas 3.0、pytest。

> **版本管理已按用户要求关闭**:本计划不含 git 操作。每个任务最后一步是"跑全部测试做检查点"。

---

## 文件结构

```
spilt/
  emg_label/
    __init__.py
    config.py          # Config dataclass（默认参数）
    io_utils.py        # parse_file_info / group_files / load_npz
    segmentation.py    # 基线估计 / 活动信号 / 阈值 / 迟滞分割 / 形态过滤 / segment_file
    features.py        # plateau_feature / zscore
    clustering.py      # select_k_and_cluster（silhouette + KMeans）
    plotting.py        # plot_overview / plot_cluster_preview
  segment.py           # 阶段1 CLI
  cluster.py           # 阶段2 CLI
  export.py            # 阶段3 CLI
  tests/
    __init__.py
    test_io_utils.py
    test_segmentation.py
    test_features.py
    test_clustering.py
```

阶段间数据流:
- `segment.py` → `out/segments.csv`(source_file, group, seg_idx, start_sample, end_sample, duration_s) + `out/overview/{stem}.png`
- `cluster.py` → `out/segments_clustered.csv`(加 cluster_id 列)+ `out/labels_template.csv`(group, cluster_id, count, label="")+ `out/clusters/{group}.png`
- 人工:填 `labels_template.csv` 的 label 列 → 另存 `out/labels.csv`
- `export.py` → `out/segments/{label}/{label}__{subject}-{hand}__{stem}__seg{idx}.npz` + `out/labeled_overview/{stem}.png`

---

## Task 1: 项目骨架与测试环境

**Files:**
- Create: `emg_label/__init__.py`（空文件）
- Create: `tests/__init__.py`（空文件）
- Create: `emg_label/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 确认 pytest 可用**

Run: `python -m pytest --version`
Expected: 打印版本号。若报 `No module named pytest`,先运行 `python -m pip install pytest`。

- [ ] **Step 2: 写失败测试**

`tests/test_config.py`:
```python
from emg_label.config import Config


def test_config_defaults():
    c = Config()
    assert c.fs == 2000
    assert c.smooth_ms == 150.0
    assert c.min_action_s == 0.4
    assert c.min_rest_gap_s == 0.2
    assert c.plateau_frac == 0.5
    assert c.k_min == 12
    assert c.k_max == 30
    assert c.out_dir == "out"
    assert c.enter_thresh is None
    assert c.exit_thresh is None


def test_config_override():
    c = Config(fs=1000, k_max=25)
    assert c.fs == 1000
    assert c.k_max == 25
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'emg_label.config'`

- [ ] **Step 4: 创建空 `emg_label/__init__.py` 和 `tests/__init__.py`,写 `emg_label/config.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    fs: int = 2000
    smooth_ms: float = 150.0
    min_action_s: float = 0.4
    min_rest_gap_s: float = 0.2
    plateau_frac: float = 0.5
    k_min: int = 12
    k_max: int = 30
    out_dir: str = "out"
    enter_thresh: float | None = None
    exit_thresh: float | None = None
    baseline_iters: int = 2
    baseline_keep_frac: float = 0.5
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 6: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 2: 文件名解析与分组 (io_utils)

**Files:**
- Create: `emg_label/io_utils.py`
- Test: `tests/test_io_utils.py`

- [ ] **Step 1: 写失败测试**

`tests/test_io_utils.py`:
```python
import warnings

from emg_label.io_utils import parse_file_info, group_files


def test_parse_standard_name():
    info = parse_file_info("/data/fgw-0917__20260502-left__20260502_115218.npz")
    assert info.subject == "fgw-0917"
    assert info.hand == "left"
    assert info.group == "fgw-0917-left"
    assert info.stem == "fgw-0917__20260502-left__20260502_115218"
    assert info.parsed is True


def test_parse_right_hand():
    info = parse_file_info("abc__20260101-right__t.npz")
    assert info.hand == "right"
    assert info.group == "abc-right"


def test_parse_unparseable_name_becomes_own_group():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = parse_file_info("/data/weird_file.npz")
    assert info.parsed is False
    assert info.subject is None
    assert info.hand is None
    assert info.group == "weird_file"


def test_group_files_buckets_by_subject_hand():
    paths = [
        "s1__d-left__t1.npz",
        "s1__d-left__t2.npz",
        "s1__d-right__t1.npz",
        "s2__d-left__t1.npz",
    ]
    groups = group_files(paths)
    assert set(groups.keys()) == {"s1-left", "s1-right", "s2-left"}
    assert len(groups["s1-left"]) == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_io_utils.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'emg_label.io_utils'`

- [ ] **Step 3: 写实现**

`emg_label/io_utils.py`:
```python
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class FileInfo:
    path: str
    stem: str
    subject: str | None
    hand: str | None
    group: str
    parsed: bool


def parse_file_info(path: str) -> FileInfo:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("__")
    if len(parts) >= 3 and "-" in parts[1]:
        subject = parts[0]
        hand = parts[1].rsplit("-", 1)[-1]
        return FileInfo(path, stem, subject, hand, f"{subject}-{hand}", True)
    warnings.warn(f"Cannot parse filename, treating as own group: {stem}")
    return FileInfo(path, stem, None, None, stem, False)


def group_files(paths: list[str]) -> dict[str, list[FileInfo]]:
    groups: dict[str, list[FileInfo]] = {}
    for p in paths:
        info = parse_file_info(p)
        groups.setdefault(info.group, []).append(info)
    return groups


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    emg = np.asarray(d["emg"], dtype=np.float32)
    ja = np.asarray(d["joint_angles"], dtype=np.float32)
    if emg.ndim != 2 or ja.ndim != 2:
        raise ValueError(f"Expected 2D emg/joint_angles in {path}")
    if emg.shape[0] != ja.shape[0]:
        raise ValueError(
            f"emg/joint_angles length mismatch in {path}: "
            f"{emg.shape[0]} vs {ja.shape[0]}"
        )
    return emg, ja
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_io_utils.py -v`
Expected: 4 passed

- [ ] **Step 5: 加 load_npz 测试(用临时文件)**

追加到 `tests/test_io_utils.py`:
```python
import numpy as np
import pytest
from emg_label.io_utils import load_npz


def test_load_npz_roundtrip(tmp_path):
    p = tmp_path / "x.npz"
    emg = np.zeros((100, 16), dtype=np.float32)
    ja = np.ones((100, 20), dtype=np.float32)
    np.savez(p, emg=emg, joint_angles=ja)
    e, j = load_npz(str(p))
    assert e.shape == (100, 16)
    assert j.shape == (100, 20)


def test_load_npz_length_mismatch(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, emg=np.zeros((100, 16), np.float32), joint_angles=np.zeros((90, 20), np.float32))
    with pytest.raises(ValueError):
        load_npz(str(p))
```

- [ ] **Step 6: 跑测试确认通过**

Run: `python -m pytest tests/test_io_utils.py -v`
Expected: 6 passed

- [ ] **Step 7: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 3: 静息基线估计 (segmentation)

**Files:**
- Create: `emg_label/segmentation.py`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: 写失败测试**

`tests/test_segmentation.py`:
```python
import numpy as np

from emg_label.segmentation import estimate_rest_baseline


def test_baseline_recovers_rest_pose_with_actions():
    rng = np.random.default_rng(0)
    rest = np.array([0.1, 0.2, 0.3, 0.4])
    n = 2000
    X = rest + rng.normal(0, 0.01, size=(n, 4))
    # inject action bursts that move far from rest in ~30% of samples
    X[500:800] += np.array([1.0, -1.0, 1.0, -1.0])
    X[1200:1500] += np.array([0.8, 0.8, -0.8, -0.8])
    baseline = estimate_rest_baseline(X)
    assert np.allclose(baseline, rest, atol=0.05)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: FAIL,`ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3: 写实现(创建文件并加 estimate_rest_baseline)**

`emg_label/segmentation.py`:
```python
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def estimate_rest_baseline(joint_angles, n_iter: int = 2, keep_frac: float = 0.5):
    X = np.asarray(joint_angles, dtype=float)
    baseline = np.median(X, axis=0)
    for _ in range(n_iter):
        dist = np.linalg.norm(X - baseline, axis=1)
        thr = np.quantile(dist, keep_frac)
        mask = dist <= thr
        if not mask.any():
            break
        baseline = np.median(X[mask], axis=0)
    return baseline
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: 1 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 4: 活动信号与自动阈值 (segmentation)

**Files:**
- Modify: `emg_label/segmentation.py`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_segmentation.py`:
```python
from emg_label.segmentation import activity_signal, auto_thresholds


def test_activity_signal_high_during_action():
    rest = np.array([0.0, 0.0, 0.0])
    X = np.tile(rest, (1000, 1)).astype(float)
    X[400:600] += np.array([1.0, 1.0, 1.0])
    act = activity_signal(X, rest, fs=1000, smooth_ms=20.0)
    assert act.shape == (1000,)
    assert act[500] > act[100]          # action > rest
    assert act[100] < 0.1               # rest near zero


def test_auto_thresholds_enter_above_exit():
    rng = np.random.default_rng(1)
    act = np.abs(rng.normal(0, 0.1, size=5000))
    act[1000:1200] += 2.0
    enter, exit_thr = auto_thresholds(act)
    assert enter > exit_thr
    assert exit_thr > np.median(act)


def test_auto_thresholds_robust_to_high_duty_cycle():
    # 50% of samples are "action" at ~1.5: median sits between the two modes,
    # so a median+MAD heuristic fails. Otsu must still land between them.
    act = np.concatenate([np.full(2500, 0.02), np.full(2500, 1.5)])
    enter, exit_thr = auto_thresholds(act)
    assert 0.02 < exit_thr < enter < 1.5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: FAIL,`ImportError: cannot import name 'activity_signal'`

- [ ] **Step 3: 写实现(追加到 segmentation.py)**

```python
def activity_signal(joint_angles, baseline, fs: int, smooth_ms: float = 150.0):
    X = np.asarray(joint_angles, dtype=float)
    dist = np.linalg.norm(X - np.asarray(baseline, dtype=float), axis=1)
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    return uniform_filter1d(dist, size=win, mode="nearest")


def _otsu_threshold(x, nbins: int = 256) -> float:
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return hi
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = hist.astype(float) / hist.sum()
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.argmax(sigma_b2))])


def auto_thresholds(activity):
    """Bimodal (Otsu) split between rest and action activity levels.

    Robust to how much of the recording is action: enter = Otsu valley,
    exit = halfway between the rest-mode center and the valley (so the
    detector exits cleanly back into rest without flickering).
    """
    a = np.asarray(activity, dtype=float)
    t = _otsu_threshold(a)
    rest = a[a <= t]
    rest_center = float(np.median(rest)) if rest.size else float(a.min())
    enter = float(t)
    exit_thr = float(rest_center + 0.5 * (t - rest_center))
    return enter, exit_thr
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: 4 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 5: 迟滞分割与形态过滤 (segmentation)

**Files:**
- Modify: `emg_label/segmentation.py`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_segmentation.py`:
```python
from emg_label.segmentation import hysteresis_segments, filter_segments


def test_hysteresis_basic():
    a = np.array([0, 0, 5, 5, 5, 0, 0, 5, 5, 0], dtype=float)
    segs = hysteresis_segments(a, enter_thr=3.0, exit_thr=1.0)
    assert segs == [(2, 5), (7, 9)]


def test_hysteresis_open_segment_at_end():
    a = np.array([0, 5, 5], dtype=float)
    segs = hysteresis_segments(a, enter_thr=3.0, exit_thr=1.0)
    assert segs == [(1, 3)]


def test_filter_drops_short_segments():
    # fs=1000 -> min_action 0.4s = 400 samples
    segs = [(0, 100), (1000, 1600)]
    out = filter_segments(segs, fs=1000, min_action_s=0.4, min_rest_gap_s=0.2)
    assert out == [(1000, 1600)]


def test_filter_merges_close_segments():
    # gap of 50 samples < min_rest_gap 0.2s(=200) -> merge
    segs = [(0, 500), (550, 1000)]
    out = filter_segments(segs, fs=1000, min_action_s=0.1, min_rest_gap_s=0.2)
    assert out == [(0, 1000)]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: FAIL,`ImportError: cannot import name 'hysteresis_segments'`

- [ ] **Step 3: 写实现(追加到 segmentation.py)**

```python
def hysteresis_segments(activity, enter_thr: float, exit_thr: float):
    a = np.asarray(activity, dtype=float)
    segments: list[tuple[int, int]] = []
    in_action = False
    start = 0
    for i, v in enumerate(a):
        if not in_action and v >= enter_thr:
            in_action = True
            start = i
        elif in_action and v < exit_thr:
            segments.append((start, i))  # end exclusive
            in_action = False
    if in_action:
        segments.append((start, len(a)))
    return segments


def filter_segments(segments, fs: int, min_action_s: float = 0.4,
                    min_rest_gap_s: float = 0.2):
    if not segments:
        return []
    min_action = int(round(min_action_s * fs))
    min_gap = int(round(min_rest_gap_s * fs))
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) >= min_action]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: 8 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 6: segment_file 编排 (segmentation)

**Files:**
- Modify: `emg_label/segmentation.py`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_segmentation.py`:
```python
from emg_label.config import Config
from emg_label.segmentation import segment_file


def _synthetic_action_rest(fs=1000):
    # rest-dominant (25% duty): 12s total, three 1s actions at 2s, 5s, 8s
    rest = np.array([0.1, 0.2, 0.0])
    n = 12 * fs
    rng = np.random.default_rng(2)
    X = rest + rng.normal(0, 0.005, size=(n, 3))
    for c in (2, 5, 8):
        X[c * fs:(c * fs + fs)] += np.array([1.0, -1.0, 0.5])
    return X


def test_segment_file_finds_three_actions():
    X = _synthetic_action_rest(fs=1000)
    cfg = Config(fs=1000, min_action_s=0.4, min_rest_gap_s=0.2, smooth_ms=50.0)
    segs, act, baseline, enter, exit_thr = segment_file(X, cfg)
    assert len(segs) == 3
    # each segment roughly within its 1s action window (±200ms tolerance)
    centers = [(s + e) / 2 / 1000 for s, e in segs]
    assert np.allclose(sorted(centers), [2.5, 5.5, 8.5], atol=0.2)
    assert enter > exit_thr
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_segmentation.py::test_segment_file_finds_three_actions -v`
Expected: FAIL,`ImportError: cannot import name 'segment_file'`

- [ ] **Step 3: 写实现(追加到 segmentation.py)**

```python
def segment_file(joint_angles, config):
    baseline = estimate_rest_baseline(
        joint_angles, config.baseline_iters, config.baseline_keep_frac
    )
    act = activity_signal(joint_angles, baseline, config.fs, config.smooth_ms)
    if config.enter_thresh is not None and config.exit_thresh is not None:
        enter, exit_thr = config.enter_thresh, config.exit_thresh
    else:
        enter, exit_thr = auto_thresholds(act)
    raw = hysteresis_segments(act, enter, exit_thr)
    segs = filter_segments(
        raw, config.fs, config.min_action_s, config.min_rest_gap_s
    )
    return segs, act, baseline, enter, exit_thr
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: 9 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 7: 姿态特征与归一化 (features)

**Files:**
- Create: `emg_label/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: 写失败测试**

`tests/test_features.py`:
```python
import numpy as np

from emg_label.features import plateau_feature, zscore


def test_plateau_feature_ignores_transition_edges():
    # 100 frames: edges are transition garbage, middle is the held pose
    X = np.zeros((100, 3))
    X[:20] = 9.0          # ramp-in garbage
    X[80:] = -9.0         # ramp-out garbage
    X[20:80] = np.array([1.0, 2.0, 3.0])  # held pose
    feat = plateau_feature(X, 0, 100, frac=0.5)
    assert np.allclose(feat, [1.0, 2.0, 3.0])


def test_plateau_feature_short_segment():
    X = np.array([[1.0, 1.0]])
    feat = plateau_feature(X, 0, 1, frac=0.5)
    assert np.allclose(feat, [1.0, 1.0])


def test_zscore_unit_variance():
    X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    Xz, mean, std = zscore(X)
    assert np.allclose(Xz.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xz.std(axis=0), 1.0, atol=1e-9)


def test_zscore_constant_column_safe():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Xz, mean, std = zscore(X)
    assert np.all(np.isfinite(Xz))
    assert np.allclose(Xz[:, 0], 0.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'emg_label.features'`

- [ ] **Step 3: 写实现**

`emg_label/features.py`:
```python
from __future__ import annotations

import numpy as np


def plateau_feature(joint_angles, start: int, end: int, frac: float = 0.5):
    X = np.asarray(joint_angles, dtype=float)[start:end]
    n = len(X)
    if n == 0:
        raise ValueError("empty segment")
    if n <= 2:
        return np.median(X, axis=0)
    half = frac / 2.0
    lo = int(np.floor((0.5 - half) * n))
    hi = int(np.ceil((0.5 + half) * n))
    lo = max(0, lo)
    hi = min(n, max(lo + 1, hi))
    return np.median(X[lo:hi], axis=0)


def zscore(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std_safe, mean, std
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_features.py -v`
Expected: 4 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 8: 聚类与 k 选择 (clustering)

**Files:**
- Create: `emg_label/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: 写失败测试**

`tests/test_clustering.py`:
```python
import numpy as np

from emg_label.clustering import select_k_and_cluster


def _three_blobs(per=15):
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 10], [0, 10]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.3, size=(per, 2)) for c in centers])
    return X


def test_selects_correct_k_for_three_blobs():
    X = _three_blobs()
    labels, best_k = select_k_and_cluster(X, k_min=2, k_max=8)
    assert best_k == 3
    # each true blob maps to a single cluster
    for i in range(3):
        block = labels[i * 15:(i + 1) * 15]
        assert len(set(block)) == 1


def test_handles_single_sample():
    X = np.array([[1.0, 2.0]])
    labels, best_k = select_k_and_cluster(X, k_min=2, k_max=8)
    assert best_k == 1
    assert labels.tolist() == [0]


def test_clamps_k_to_sample_count():
    X = _three_blobs(per=2)  # 6 samples, k_max=30 must clamp
    labels, best_k = select_k_and_cluster(X, k_min=12, k_max=30)
    assert best_k <= 5
    assert len(labels) == 6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'emg_label.clustering'`

- [ ] **Step 3: 写实现**

`emg_label/clustering.py`:
```python
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def select_k_and_cluster(X, k_min: int, k_max: int, random_state: int = 0):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), 1
    hi = min(k_max, n - 1)
    lo = max(2, min(k_min, hi))
    best_k = lo
    best_score = -np.inf
    best_labels = None
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    if best_labels is None:
        km = KMeans(n_clusters=lo, n_init=10, random_state=random_state)
        best_labels = km.fit_predict(X)
        best_k = lo
    return best_labels, best_k
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: 3 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 9: 绘图模块 (plotting) — 冒烟测试

**Files:**
- Create: `emg_label/plotting.py`
- Test: `tests/test_plotting.py`

绘图内容靠人工核对,这里只验证函数能跑通并生成非空 PNG。

- [ ] **Step 1: 写失败测试**

`tests/test_plotting.py`:
```python
import os

import numpy as np

from emg_label.plotting import plot_overview, plot_cluster_preview


def test_plot_overview_creates_png(tmp_path):
    n = 2000
    emg = np.random.default_rng(0).normal(0, 1, size=(n, 16))
    ja = np.zeros((n, 20))
    activity = np.abs(np.random.default_rng(1).normal(0, 0.1, size=n))
    segs = [(400, 600), (1000, 1300)]
    out = tmp_path / "ov.png"
    plot_overview(emg, ja, activity, segs, fs=1000, out_path=str(out),
                  enter_thr=0.5, exit_thr=0.3, labels=["a", "b"])
    assert os.path.getsize(out) > 0


def test_plot_cluster_preview_creates_png(tmp_path):
    centroids = [np.random.default_rng(i).normal(0, 1, size=20) for i in range(5)]
    counts = [3, 5, 2, 8, 4]
    ids = [0, 1, 2, 3, 4]
    out = tmp_path / "cl.png"
    plot_cluster_preview(centroids, counts, ids, str(out))
    assert os.path.getsize(out) > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_plotting.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'emg_label.plotting'`

- [ ] **Step 3: 写实现**

`emg_label/plotting.py`:
```python
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_overview(emg, joint_angles, activity, segments, fs, out_path,
                  enter_thr=None, exit_thr=None, labels=None):
    n = len(activity)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    env = np.abs(np.asarray(emg)).mean(axis=1)
    axes[0].plot(t, env, lw=0.5)
    axes[0].set_ylabel("EMG |mean|")
    axes[1].plot(t, np.asarray(activity), lw=0.8, color="k")
    axes[1].set_ylabel("joint activity")
    axes[1].set_xlabel("time (s)")
    if enter_thr is not None:
        axes[1].axhline(enter_thr, color="r", ls="--", lw=0.8)
    if exit_thr is not None:
        axes[1].axhline(exit_thr, color="orange", ls="--", lw=0.8)
    ymax = axes[1].get_ylim()[1]
    for idx, (s, e) in enumerate(segments):
        for ax in axes:
            ax.axvspan(s / fs, e / fs, color="green", alpha=0.15)
        lab = str(labels[idx]) if labels is not None else str(idx)
        axes[1].text((s + e) / 2 / fs, ymax * 0.9, lab, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_cluster_preview(centroids, counts, ids, out_path):
    k = len(centroids)
    ncol = 4
    nrow = max(1, int(np.ceil(k / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow),
                             squeeze=False)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i < k:
            ax.bar(range(len(centroids[i])), centroids[i])
            ax.set_title(f"cluster {ids[i]} (n={counts[i]})", fontsize=9)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_plotting.py -v`
Expected: 2 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 10: 阶段1 CLI `segment.py`

**Files:**
- Create: `segment.py`

无单元测试(CLI 编排),靠 Task 13 端到端验证。

- [ ] **Step 1: 写 `segment.py`**

```python
from __future__ import annotations

import argparse
import csv
import glob
import os

from emg_label import io_utils, plotting, segmentation
from emg_label.config import Config


def main():
    ap = argparse.ArgumentParser(description="Stage 1: segment action vs rest")
    ap.add_argument("input_dir", help="folder containing .npz files")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--min-action-s", type=float, default=0.4)
    ap.add_argument("--min-rest-gap-s", type=float, default=0.2)
    ap.add_argument("--smooth-ms", type=float, default=150.0)
    args = ap.parse_args()

    cfg = Config(
        fs=args.fs, out_dir=args.out, min_action_s=args.min_action_s,
        min_rest_gap_s=args.min_rest_gap_s, smooth_ms=args.smooth_ms,
    )
    os.makedirs(os.path.join(cfg.out_dir, "overview"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    if not paths:
        print(f"No .npz files found in {args.input_dir}")
        return

    rows = []
    for path in paths:
        info = io_utils.parse_file_info(path)
        emg, ja = io_utils.load_npz(path)
        segs, act, _baseline, enter, exit_thr = segmentation.segment_file(ja, cfg)
        for seg_idx, (s, e) in enumerate(segs):
            rows.append({
                "source_file": os.path.basename(path),
                "group": info.group,
                "seg_idx": seg_idx,
                "start_sample": s,
                "end_sample": e,
                "duration_s": round((e - s) / cfg.fs, 4),
            })
        png = os.path.join(cfg.out_dir, "overview", info.stem + ".png")
        plotting.plot_overview(emg, ja, act, segs, cfg.fs, png, enter, exit_thr)
        print(f"{os.path.basename(path)}: {len(segs)} segments")

    csv_path = os.path.join(cfg.out_dir, "segments.csv")
    fields = ["source_file", "group", "seg_idx", "start_sample",
              "end_sample", "duration_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} segments)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('segment.py', encoding='utf-8').read())"`
Expected: 无输出(无语法错误)

- [ ] **Step 3: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过(本任务未加测试,确认没破坏既有测试)

---

## Task 11: 阶段2 CLI `cluster.py`

**Files:**
- Create: `cluster.py`

- [ ] **Step 1: 写 `cluster.py`**

```python
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from emg_label import clustering, features, io_utils, plotting
from emg_label.config import Config


def main():
    ap = argparse.ArgumentParser(description="Stage 2: cluster segments per group")
    ap.add_argument("input_dir", help="same folder of .npz used in stage 1")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--k-min", type=int, default=12)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--plateau-frac", type=float, default=0.5)
    args = ap.parse_args()

    cfg = Config(fs=args.fs, out_dir=args.out, k_min=args.k_min,
                 k_max=args.k_max, plateau_frac=args.plateau_frac)
    seg_df = pd.read_csv(os.path.join(cfg.out_dir, "segments.csv"))
    os.makedirs(os.path.join(cfg.out_dir, "clusters"), exist_ok=True)

    ja_cache: dict[str, np.ndarray] = {}

    def get_ja(fname: str) -> np.ndarray:
        if fname not in ja_cache:
            _, ja = io_utils.load_npz(os.path.join(args.input_dir, fname))
            ja_cache[fname] = ja
        return ja_cache[fname]

    seg_df["cluster_id"] = -1
    template_rows = []

    for group, gdf in seg_df.groupby("group"):
        feats, idxs = [], []
        for ridx, row in gdf.iterrows():
            ja = get_ja(row["source_file"])
            feats.append(features.plateau_feature(
                ja, int(row["start_sample"]), int(row["end_sample"]),
                cfg.plateau_frac))
            idxs.append(ridx)
        X = np.array(feats)
        Xz, _mean, _std = features.zscore(X)
        labels, best_k = clustering.select_k_and_cluster(Xz, cfg.k_min, cfg.k_max)
        for ridx, lab in zip(idxs, labels):
            seg_df.at[ridx, "cluster_id"] = int(lab)

        centroids, counts, ids = [], [], []
        for c in sorted(set(int(x) for x in labels)):
            mask = labels == c
            centroids.append(X[mask].mean(axis=0))
            counts.append(int(mask.sum()))
            ids.append(c)
            template_rows.append({"group": group, "cluster_id": c,
                                  "count": int(mask.sum()), "label": ""})
        png = os.path.join(cfg.out_dir, "clusters", f"{group}.png")
        plotting.plot_cluster_preview(centroids, counts, ids, png)
        print(f"group {group}: {len(gdf)} segments -> k={best_k}")

    seg_df.to_csv(os.path.join(cfg.out_dir, "segments_clustered.csv"), index=False)
    pd.DataFrame(template_rows).to_csv(
        os.path.join(cfg.out_dir, "labels_template.csv"), index=False)
    print("Wrote segments_clustered.csv and labels_template.csv")
    print(f"-> Fill the 'label' column in {cfg.out_dir}/labels_template.csv "
          f"and save as {cfg.out_dir}/labels.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('cluster.py', encoding='utf-8').read())"`
Expected: 无输出

- [ ] **Step 3: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 12: 阶段3 CLI `export.py`

**Files:**
- Create: `export.py`

- [ ] **Step 1: 写 `export.py`**

```python
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from emg_label import io_utils, plotting, segmentation
from emg_label.config import Config


def _clean_label(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


def main():
    ap = argparse.ArgumentParser(description="Stage 3: export labeled segments")
    ap.add_argument("input_dir", help="same folder of .npz used in stage 1")
    ap.add_argument("--out", default="out")
    ap.add_argument("--labels", default=None,
                    help="labels.csv path (default <out>/labels.csv)")
    ap.add_argument("--fs", type=int, default=2000)
    args = ap.parse_args()

    cfg = Config(fs=args.fs, out_dir=args.out)
    seg_df = pd.read_csv(os.path.join(cfg.out_dir, "segments_clustered.csv"))
    labels_path = args.labels or os.path.join(cfg.out_dir, "labels.csv")
    lab_df = pd.read_csv(labels_path)
    lab_map = {(r["group"], int(r["cluster_id"])): _clean_label(r["label"])
               for _, r in lab_df.iterrows()}

    seg_dir = os.path.join(cfg.out_dir, "segments")
    lov_dir = os.path.join(cfg.out_dir, "labeled_overview")
    os.makedirs(lov_dir, exist_ok=True)

    emg_cache: dict[str, np.ndarray] = {}
    ja_cache: dict[str, np.ndarray] = {}

    def load(fname: str):
        if fname not in emg_cache:
            emg_cache[fname], ja_cache[fname] = io_utils.load_npz(
                os.path.join(args.input_dir, fname))
        return emg_cache[fname], ja_cache[fname]

    by_file: dict[str, list] = {}
    n_exported = 0
    for _, row in seg_df.iterrows():
        fname = row["source_file"]
        group = row["group"]
        cid = int(row["cluster_id"])
        label = lab_map.get((group, cid), "")
        if not label:
            continue
        emg, ja = load(fname)
        s, e = int(row["start_sample"]), int(row["end_sample"])
        info = io_utils.parse_file_info(os.path.join(args.input_dir, fname))
        subj = info.subject or "NA"
        hand = info.hand or "NA"
        out_d = os.path.join(seg_dir, label)
        os.makedirs(out_d, exist_ok=True)
        out_name = (f"{label}__{subj}-{hand}__{info.stem}"
                    f"__seg{int(row['seg_idx'])}.npz")
        np.savez(
            os.path.join(out_d, out_name),
            emg=emg[s:e], joint_angles=ja[s:e],
            label=label, source_file=fname, group=group,
            seg_idx=int(row["seg_idx"]), start_sample=s, end_sample=e,
            fs=cfg.fs, cluster_id=cid,
        )
        by_file.setdefault(fname, []).append((s, e, label))
        n_exported += 1

    # labeled overview per file (recompute activity for context)
    for fname, segs in by_file.items():
        emg, ja = load(fname)
        baseline = segmentation.estimate_rest_baseline(ja)
        act = segmentation.activity_signal(ja, baseline, cfg.fs, cfg.smooth_ms)
        info = io_utils.parse_file_info(os.path.join(args.input_dir, fname))
        spans = [(s, e) for s, e, _ in segs]
        labels = [lab for _, _, lab in segs]
        png = os.path.join(lov_dir, info.stem + ".png")
        plotting.plot_overview(emg, ja, act, spans, cfg.fs, png, labels=labels)

    print(f"Exported {n_exported} labeled segments to {seg_dir}")
    print(f"Labeled overviews in {lov_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('export.py', encoding='utf-8').read())"`
Expected: 无输出

- [ ] **Step 3: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 13: 真实数据端到端验证(人工)

**Files:** 无新建,只运行。

- [ ] **Step 1: 跑阶段1**

Run: `python segment.py . --out out`
Expected: 打印 `fgw-0917__...: N segments` 与 `Wrote out/segments.csv (...)`。
人工核对:`out/overview/*.png` 中绿色动作区间是否落在关节活动信号的隆起处;N 是否接近"二十多个手势 × 2-3 次"的量级。

- [ ] **Step 2: 跑阶段2**

Run: `python cluster.py . --out out`
Expected: 打印 `group fgw-0917-left: N segments -> k=...`;生成 `out/labels_template.csv`、`out/clusters/fgw-0917-left.png`、`out/segments_clustered.csv`。
人工核对:聚类预览图里同一 cluster 的质心姿态是否各不相同、是否大致对应不同手势。

- [ ] **Step 3: 人工填标签**

把 `out/labels_template.csv` 的 `label` 列按预览图填上手势名,另存为 `out/labels.csv`。

- [ ] **Step 4: 跑阶段3**

Run: `python export.py . --out out`
Expected: 打印 `Exported M labeled segments`;`out/segments/{label}/*.npz` 生成;`out/labeled_overview/*.png` 上每段叠加了标签。
人工抽查一个导出的 npz:
Run: `python -c "import numpy as np,glob; f=glob.glob('out/segments/*/*.npz')[0]; d=np.load(f,allow_pickle=True); print(f); print({k: (d[k].shape if hasattr(d[k],'shape') and d[k].ndim else d[k]) for k in d})"`
Expected: 打印该段的 emg/joint_angles 形状与 label/source_file 等元数据。

- [ ] **Step 5: 最终检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## 完成标准

- 单元测试全绿(io_utils / segmentation / features / clustering / plotting)。
- 三个 CLI 在真实 npz 上跑通,产出 segments.csv、聚类预览、带标签 npz、标注总览图。
- 人工核对切分与聚类预览合理。
