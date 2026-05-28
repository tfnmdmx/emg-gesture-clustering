# 3D 手部姿态预览 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为聚类阶段额外渲染“每个 cluster 质心姿态的 3D 手骨架”拼图 PNG,替代不直观的条形图,便于看图给手势命名。

**Architecture:** 新增自包含正向运动学模块 `emg_label/hand3d.py`(关节角度→21 个 3D 关键点,仅用 numpy),`plotting.py` 增加 3D 手拼图函数,`cluster.py` 聚类后额外输出 `{group}_hands.png`。无新依赖(只用已装的 numpy/matplotlib)。

**Tech Stack:** Python 3.13、numpy、matplotlib(mpl_toolkits.mplot3d)、pytest。

> **版本管理已按用户要求关闭**:本计划不含 git 操作。每个任务最后一步是“跑全部测试做检查点”。

---

## 文件结构

```
emg_label/
  hand3d.py     # 新:LANDMARK_NAMES / BONE_CONNECTIONS / PALM_CONNECTIONS / angles_to_landmarks / draw_hand
  plotting.py   # 改:新增 plot_cluster_hands(...)
cluster.py      # 改:聚类后额外输出 {group}_hands.png
tests/
  test_hand3d.py    # 新
  test_plotting.py  # 改:加 plot_cluster_hands 冒烟测试
```

关键点顺序与骨骼连接严格沿用用户 `visualize_hand_3d.py`(便于将来替换为 emg2pose 精确 FK)。

---

## Task 1: hand3d 正向运动学

**Files:**
- Create: `emg_label/hand3d.py`
- Test: `tests/test_hand3d.py`

- [ ] **Step 1: 写失败测试**

`tests/test_hand3d.py`:
```python
import numpy as np

from emg_label.hand3d import angles_to_landmarks, LANDMARK_NAMES, BONE_CONNECTIONS


def test_landmark_layout_constants():
    assert len(LANDMARK_NAMES) == 21
    assert LANDMARK_NAMES[5] == "WRIST"
    assert LANDMARK_NAMES[1] == "INDEX_TIP"
    # index bone chain: wrist -> prox -> int -> dist -> tip
    assert [5, 8, 9, 10, 1] in BONE_CONNECTIONS


def test_shape_and_finite():
    lm = angles_to_landmarks(np.zeros(20), side="left")
    assert lm.shape == (21, 3)
    assert np.all(np.isfinite(lm))


def test_zero_angles_fingers_extended():
    # straight fingers: index tip (1) reaches further along +y than its knuckle (8)
    lm = angles_to_landmarks(np.zeros(20), side="left")
    assert lm[1][1] > lm[8][1]


def test_mcp_flexion_curls_finger_toward_palm():
    straight = np.zeros(20)
    flexed = np.zeros(20)
    flexed[5] = 1.4  # index MCP flexion channel
    lm0 = angles_to_landmarks(straight, side="left")
    lm1 = angles_to_landmarks(flexed, side="left")
    wrist = lm0[5]
    d0 = np.linalg.norm(lm0[1] - wrist)  # index-tip distance from wrist, straight
    d1 = np.linalg.norm(lm1[1] - wrist)  # flexed
    assert d1 < d0                 # curled -> fingertip closer to wrist
    assert lm1[1][2] < lm0[1][2]   # curled toward palm (-z)


def test_side_mirrors_x_only():
    a = np.zeros(20)
    a[5] = 0.8
    left = angles_to_landmarks(a, side="left")
    right = angles_to_landmarks(a, side="right")
    assert np.allclose(left[:, 0], -right[:, 0])      # x mirrored
    assert np.allclose(left[:, 1:], right[:, 1:])     # y,z identical
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_hand3d.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'emg_label.hand3d'`

- [ ] **Step 3: 写实现**

`emg_label/hand3d.py`:
```python
from __future__ import annotations

import numpy as np

LANDMARK_NAMES = [
    "THUMB_TIP", "INDEX_TIP", "MIDDLE_TIP", "RING_TIP", "PINKY_TIP",
    "WRIST",
    "THUMB_INT", "THUMB_DIST",
    "INDEX_PROX", "INDEX_INT", "INDEX_DIST",
    "MIDDLE_PROX", "MIDDLE_INT", "MIDDLE_DIST",
    "RING_PROX", "RING_INT", "RING_DIST",
    "PINKY_PROX", "PINKY_INT", "PINKY_DIST",
    "PALM",
]

BONE_CONNECTIONS = [
    [5, 6, 7, 0],         # thumb:  wrist -> int -> dist -> tip
    [5, 8, 9, 10, 1],     # index:  wrist -> prox -> int -> dist -> tip
    [5, 11, 12, 13, 2],   # middle
    [5, 14, 15, 16, 3],   # ring
    [5, 17, 18, 19, 4],   # pinky
]

PALM_CONNECTIONS = [
    [20, 5], [20, 8], [20, 11], [20, 14], [20, 17], [20, 7],
]

# Calibration constants (sign of flexion / abduction). Adjusted in Task 4
# against the real rest pose so the neutral hand renders correctly.
FLEX_SIGN = 1.0
ABD_SIGN = 1.0

# Per finger: (knuckle_x, knuckle_y, (Lprox, Lint, Ldist),
#              (abd_ch, mcp_ch, pip_ch, dip_ch),
#              prox_idx, int_idx, dist_idx, tip_idx)
_FINGERS = [
    ("index",  0.45, 1.00, (0.42, 0.26, 0.20), (4, 5, 6, 7),    8, 9, 10, 1),
    ("middle", 0.15, 1.05, (0.46, 0.30, 0.22), (8, 9, 10, 11),  11, 12, 13, 2),
    ("ring",  -0.15, 1.00, (0.42, 0.28, 0.21), (12, 13, 14, 15), 14, 15, 16, 3),
    ("pinky", -0.45, 0.90, (0.34, 0.20, 0.16), (16, 17, 18, 19), 17, 18, 19, 4),
]


def _rot_x(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def _rot_z(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _finger_chain(angles, kx, ky, lengths, chans):
    abd_c, mcp_c, pip_c, dip_c = chans
    abd = ABD_SIGN * float(angles[abd_c])
    a1 = FLEX_SIGN * float(angles[mcp_c])
    a2 = a1 + FLEX_SIGN * float(angles[pip_c])
    a3 = a2 + FLEX_SIGN * float(angles[dip_c])
    lp, li, ld = lengths
    # straight finger points +y; positive flexion curls toward -z (palm)
    d1 = np.array([0.0, np.cos(a1), -np.sin(a1)])
    d2 = np.array([0.0, np.cos(a2), -np.sin(a2)])
    d3 = np.array([0.0, np.cos(a3), -np.sin(a3)])
    prox = np.array([0.0, 0.0, 0.0])
    j_int = prox + lp * d1
    j_dist = j_int + li * d2
    j_tip = j_dist + ld * d3
    base = np.array([kx, ky, 0.0])
    return [(_rot_z(p, abd) + base) for p in (prox, j_int, j_dist, j_tip)]


def _thumb_chain(angles):
    # Approximate thumb: rooted near the wrist on the +x (thumb) side.
    # ch0 in-plane (opposition), ch1 out-of-plane, ch2 MCP flex, ch3 IP flex.
    base = np.array([0.60, 0.20, 0.05])          # THUMB_INT (landmark 6)
    inplane = ABD_SIGN * float(angles[0])
    out = float(angles[1])
    mcp = FLEX_SIGN * float(angles[2])
    ip = mcp + FLEX_SIGN * float(angles[3])
    d = np.array([0.7, 0.7, 0.0])
    d = d / np.linalg.norm(d)
    d = _rot_z(d, 0.5 * inplane)
    d = _rot_x(d, -0.4 * out)
    d1 = _rot_x(d, -mcp)
    dist = base + 0.34 * d1                       # THUMB_DIST (landmark 7)
    d2 = _rot_x(d1, -(ip - mcp))
    tip = dist + 0.28 * d2                        # THUMB_TIP (landmark 0)
    return base, dist, tip


def angles_to_landmarks(angles20, side="left"):
    """Approximate forward kinematics: 20 joint angles (rad) -> (21, 3) landmarks.

    Self-contained stick-figure hand. Landmark order matches LANDMARK_NAMES.
    ``side`` mirrors the x-axis ("left" vs "right"). Geometry is approximate
    (for gesture recognition during labeling), not anatomically exact.
    """
    a = np.asarray(angles20, dtype=float).reshape(-1)
    lm = np.zeros((21, 3))
    lm[5] = np.array([0.0, 0.0, 0.0])  # wrist at origin
    for _name, kx, ky, lengths, chans, pidx, iidx, didx, tidx in _FINGERS:
        prox, j_int, j_dist, j_tip = _finger_chain(a, kx, ky, lengths, chans)
        lm[pidx], lm[iidx], lm[didx], lm[tidx] = prox, j_int, j_dist, j_tip
    t_int, t_dist, t_tip = _thumb_chain(a)
    lm[6], lm[7], lm[0] = t_int, t_dist, t_tip
    lm[20] = (lm[5] + lm[11]) / 2.0   # palm center: wrist <-> middle knuckle
    if side == "left":
        lm = lm.copy()
        lm[:, 0] *= -1.0
    return lm
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_hand3d.py -v`
Expected: 5 passed

- [ ] **Step 5: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 2: 绘图 — draw_hand 与 plot_cluster_hands

**Files:**
- Modify: `emg_label/hand3d.py`(加 `draw_hand`)
- Modify: `emg_label/plotting.py`(加 `plot_cluster_hands`)
- Test: `tests/test_hand3d.py`(加 draw_hand 冒烟)、`tests/test_plotting.py`(加 montage 冒烟)

- [ ] **Step 1: 写失败测试(draw_hand)**

追加到 `tests/test_hand3d.py`:
```python
def test_draw_hand_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from emg_label.hand3d import draw_hand

    lm = angles_to_landmarks(np.zeros(20), side="left")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_hand(ax, lm)
    out = tmp_path / "hand.png"
    fig.savefig(out)
    plt.close(fig)
    import os
    assert os.path.getsize(out) > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_hand3d.py::test_draw_hand_smoke -v`
Expected: FAIL,`ImportError: cannot import name 'draw_hand'`

- [ ] **Step 3: 实现 draw_hand(追加到 `emg_label/hand3d.py`)**

```python
def draw_hand(ax, landmarks, color="C0"):
    """Draw a hand skeleton (joints + bones + palm lines) on a 3D axis."""
    L = np.asarray(landmarks)
    ax.scatter(L[:, 0], L[:, 1], L[:, 2], s=8, c=color)
    for bone in BONE_CONNECTIONS:
        ax.plot(L[bone, 0], L[bone, 1], L[bone, 2], c=color, lw=2)
    for conn in PALM_CONNECTIONS:
        ax.plot(L[conn, 0], L[conn, 1], L[conn, 2], c=color, lw=1, ls=":")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_hand3d.py -v`
Expected: 6 passed

- [ ] **Step 5: 写失败测试(plot_cluster_hands)**

追加到 `tests/test_plotting.py`:
```python
def test_plot_cluster_hands_creates_png(tmp_path):
    from emg_label.plotting import plot_cluster_hands
    rng = np.random.default_rng(0)
    centroids = [rng.normal(0.3, 0.3, size=20) for _ in range(5)]
    counts = [10, 20, 5, 8, 3]
    ids = [0, 1, 2, 3, 4]
    out = tmp_path / "hands.png"
    plot_cluster_hands(centroids, counts, ids, str(out), side="left")
    import os
    assert os.path.getsize(out) > 0
```

- [ ] **Step 6: 跑测试确认失败**

Run: `python -m pytest tests/test_plotting.py::test_plot_cluster_hands_creates_png -v`
Expected: FAIL,`ImportError: cannot import name 'plot_cluster_hands'`

- [ ] **Step 7: 实现 plot_cluster_hands(追加到 `emg_label/plotting.py`)**

在 `emg_label/plotting.py` 末尾追加(文件顶部已 `import numpy as np` 和 matplotlib Agg):
```python
def plot_cluster_hands(centroids, counts, ids, out_path, side="left"):
    from emg_label.hand3d import angles_to_landmarks, draw_hand

    all_lm = [angles_to_landmarks(c, side=side) for c in centroids]
    pts = np.concatenate(all_lm, axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    k = len(centroids)
    ncol = 4
    nrow = max(1, int(np.ceil(k / ncol)))
    fig = plt.figure(figsize=(3.2 * ncol, 3.0 * nrow))
    for i in range(k):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        draw_hand(ax, all_lm[i])
        ax.set_title(f"cluster {ids[i]} (n={counts[i]})", fontsize=9)
        ax.set_xlim(mn[0], mx[0])
        ax.set_ylim(mn[1], mx[1])
        ax.set_zlim(mn[2], mx[2])
        ax.view_init(elev=20, azim=-70)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
```

- [ ] **Step 8: 跑测试确认通过**

Run: `python -m pytest tests/test_plotting.py -v`
Expected: 3 passed

- [ ] **Step 9: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## Task 3: cluster.py 接入 3D 手拼图

**Files:**
- Modify: `cluster.py`

无单元测试(CLI 编排),靠 Task 4 端到端验证。

- [ ] **Step 1: 在写聚类预览 PNG 之后追加 3D 手拼图**

`cluster.py` 中当前这段(在 group 循环内):
```python
        png = os.path.join(cfg.out_dir, "clusters", f"{group}.png")
        plotting.plot_cluster_preview(centroids, counts, ids, png)
        print(f"group {group}: {len(gdf)} segments -> k={best_k}")
```
改为:
```python
        png = os.path.join(cfg.out_dir, "clusters", f"{group}.png")
        plotting.plot_cluster_preview(centroids, counts, ids, png)
        side = group.rsplit("-", 1)[-1]
        if side not in ("left", "right"):
            side = "left"
        hands_png = os.path.join(cfg.out_dir, "clusters", f"{group}_hands.png")
        plotting.plot_cluster_hands(centroids, counts, ids, hands_png, side=side)
        print(f"group {group}: {len(gdf)} segments -> k={best_k}")
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('cluster.py', encoding='utf-8').read())"`
Expected: 无输出

- [ ] **Step 3: 检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过(未加测试,确认没破坏既有测试)

---

## Task 4: 真实数据标定与验证(人工)

**Files:** 可能 Modify `emg_label/hand3d.py`(仅调 `FLEX_SIGN`/`ABD_SIGN` 或 `_thumb_chain` 增益常量)

- [ ] **Step 1: 渲染静息姿态单手,核对方向**

Run:
```
python -c "import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from emg_label import io_utils, hand3d; import glob; f=sorted(glob.glob('fgw0917_0502_left/*.npz'))[0]; _,ja=io_utils.load_npz(f); rest=np.median(ja,axis=0); lm=hand3d.angles_to_landmarks(rest, side='left'); fig=plt.figure(); ax=fig.add_subplot(111,projection='3d'); hand3d.draw_hand(ax,lm); ax.view_init(elev=20,azim=-70); fig.savefig('out/_rest_hand.png',dpi=110); print('wrote out/_rest_hand.png')"
```
Expected: 生成 `out/_rest_hand.png`。打开核对:应像中性、略放松、手指基本伸展、略分开的手;手指**不应向手背反向弯**,拇指应在侧面。

- [ ] **Step 2: 若方向不对则调标定常量**

只允许改 `emg_label/hand3d.py` 顶部的 `FLEX_SIGN`(屈曲方向反了就设为 `-1.0`)、`ABD_SIGN`(张开方向反了就翻转),以及必要时 `_thumb_chain` 内的增益(`0.5`、`0.4`)。改后重跑 Step 1,直到静息手看起来合理。每次改动后跑 `python -m pytest tests/test_hand3d.py -q` 确认几何测试仍通过(若 `FLEX_SIGN` 改为 -1,`test_mcp_flexion_curls_finger_toward_palm` 里 `lm1[1][2] < lm0[1][2]` 会变号 —— 此时把该断言改为对应方向,并在测试注释里说明标定结论)。

- [ ] **Step 3: 重跑聚类,生成正式 3D 拼图**

Run: `python cluster.py fgw0917_0502_left --out out --k 18`
Expected: 打印 `group fgw-0917-left: 3612 segments -> k=18`;生成 `out/clusters/fgw-0917-left_hands.png`(及原 `fgw-0917-left.png`)。

- [ ] **Step 4: 核对 3D 拼图**

打开 `out/clusters/fgw-0917-left_hands.png`。核对:不同 cluster 的手形可区分(如握拳=全屈、伸指=张开、对指=拇指与某指靠拢),统一视角下便于横向比较。若个别手势辨识仍困难,记录但不强求(近似 FK 局限)。

- [ ] **Step 5: 清理临时文件**

Run: `python -c "import os; p='out/_rest_hand.png'; os.remove(p) if os.path.exists(p) else None; print('cleaned')"`

- [ ] **Step 6: 最终检查点 — 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部通过

---

## 完成标准

- `emg_label/hand3d.py` 提供 `angles_to_landmarks`(21×3)与 `draw_hand`;几何单测通过。
- `cluster.py` 额外产出 `out/clusters/{group}_hands.png`(3D 手拼图),保留条形图。
- 静息姿态渲染合理(标定通过),3D 拼图中各 cluster 手形可区分用于命名。
- 全部测试通过。
