# 交互式打标网页 — 设计与实现文档

> 目标：把现有"静态 `index.html` 浏览 + 手动编辑 `clips.csv`"的真值集打标流程，
> 升级为一个**交互式网页**：左边 clip 队列，中间 overview 图 + 可旋转/可播放的 3D 手势，
> 右边填手势真值标签，**点一下即写回 CSV**。覆盖 RUNBOOK §A.5 那一步。

本文是设计 + 落地文档，代码块基本可直接复制使用。先读 [METHOD.md](METHOD.md) / [RUNBOOK.md](RUNBOOK.md) 了解上游产物。

---

## 0. 现状与痛点

当前真值集打标（RUNBOOK §A.5）的人工环节是：

1. `export_clips.py` 生成 `out_pose/clips_export/index.html`（静态画廊，每个 clip 6 帧关键帧 PNG）；
2. 人**另开文本编辑器**手动在 `out_pose/clips.csv` 的 `gesture_label` 列逐行填字符串。

痛点：

| 痛点 | 说明 |
|------|------|
| 看图与填表分离 | 浏览器看图、编辑器填表，来回切，行号对不上极易填错行 |
| 只有 6 帧静态图 | 快速/细微手势 6 帧不够辨识，缺一个能旋转、能逐帧播放的 3D 视图 |
| 没有 overview 上下文 | 看不到这个 clip 在整条录制时间轴上的位置、和相邻 burst/clip 的关系 |
| 无进度/无校验 | 不知道标了多少、哪些是"优先复核"（`matched_burst=-1`）、标签拼写无约束 |
| 手改 `clips.csv` 危险 | 脚本会重写 `clips.csv` 其它列，直接编辑容易被覆盖（RUNBOOK 已警告要先 `cp`） |

---

## 1. 方案选型（"是否有更好的方法"）

三种可选架构，结论：**推荐 C（本地轻服务端）**。

| 方案 | 怎么做 | 能写 CSV 吗 | 3D 交互 | 复杂度 | 结论 |
|------|--------|-------------|---------|--------|------|
| **A. 纯静态 HTML** | 在现有 `index.html` 上加 JS，标签存 `localStorage`，最后导出下载一个 csv | ✗ 只能"下载"到浏览器下载目录，无法原地写 `out_pose/`，多人/断点续标乱 | 可（Plotly.js）但 3D 坐标得预先烘进 HTML，体积爆炸 | 低 | 不推荐 |
| **B. 静态 + File System Access API** | 用浏览器新 API 让用户授权写本地文件 | △ 仅 Chrome/Edge，需每次授权，路径不可控，非 https 受限 | 同 A | 中 | 不推荐 |
| **C. 本地轻服务端**（推荐） | 一个本地 Python 进程读 `out_pose/`、按需算 3D、`POST` 写回 CSV，浏览器只当前端 | ✓ 服务端直接原子写 `labels.csv`，断点续标天然支持 | ✓ 3D 坐标**按需**算 + 缓存，前端 Plotly.js 渲染 | 中 | **推荐** |

**为什么是 C：**

- 写 CSV 是核心诉求，只有服务端能稳妥地原地写 `out_pose/labels.csv`（原子写、可加 labeler/时间戳、断点续标）。
- 3D 渲染可**复用现有模块**：优先 `io_utils.load_skeleton`（raw Manus 骨架，**无需 torch**），回退 `hand3d.angles_batch_to_landmarks`（FK，需 torch）——和 `export_clips.py` / `animate_segment.py` 完全同一套逻辑。
- 零额外重依赖：用 **Python 标准库 `http.server`** 就能写（项目里目前没装 flask/fastapi，标准库方案免安装、用 emg2pose 的 python 直接跑）。前端 3D 用 Plotly.js（一个 CDN/vendored js 文件）。

> 如果团队已惯用 Flask/FastAPI，可平替（路由一一对应），但标准库版的好处是**零 pip 安装**、和 `pulse.sh` 用同一个解释器即可跑。

---

## 2. 架构总览

```
浏览器 (webui/index.html + app.js + plotly.min.js)
   │  fetch JSON / POST 标签
   ▼
label_server.py   (stdlib http.server，用 emg2pose 的 python 跑)
   │  启动时建索引：clips.csv → {key: row}；读 labels.csv、recordings.csv
   │
   ├─ 读  out_pose/clips.csv            (打标单元 + QC 列)
   ├─ 读  out_pose/recordings.csv       (每录制 duration / lag_flag，用于 overview 上下文)
   ├─ 读  out_pose/shards/{stem}/overview.png   (三行 overview 图，直接当图片发)
   ├─ 读  out_pose/clips_export/{key}.png       (6 帧关键帧，快速参考；可选)
   ├─ 算  3D 帧 JSON  ← load_skeleton(源npz) 优先 / FK 回退，缓存到 pose_cache/{key}.json
   │
   └─ 写  out_pose/labels.csv           ← POST /api/label（原子写，断点续标）
          out_pose/clips_labeled.csv    ← GET /api/export（clips.csv ⨝ labels.csv）
```

**数据流（一次打标）：**

1. 前端 `GET /api/clips` 拿到全部 clip 的轻量元信息（QC + 当前标签 + key），渲染左侧队列。
2. 选中一个 clip → `GET /api/overview?stem=...`（图片）+ `GET /api/pose?key=...`（3D 帧 JSON）+ `GET /api/keyframes?key=...`（6 帧 png）。
3. 填标签、回车 → `POST /api/label {key, gesture_label}` → 服务端 upsert `labels.csv` → 前端把该行染绿、自动跳下一个。
4. 全部标完 → `GET /api/export` 生成 `clips_labeled.csv`（= 真值集）。

---

## 3. 数据契约

### 3.1 读取（上游产物，只读，不改）

| 文件 | 用途 |
|------|------|
| `out_pose/clips.csv` | 打标单元 + 子结构 sample 索引 + QC 列（schema 见 METHOD §9.2） |
| `out_pose/recordings.csv` | 每录制 `duration_s` / `lag_flag` / `n_clip_only`，给 overview 提供上下文与排序优先级 |
| `out_pose/shards/{stem}/overview.png` | 三行 overview 图（EMG 包络+burst / pose speed+clip / matched 段） |
| `out_pose/clips_export/{key}.png` | 6 帧关键帧（可选，作快速参考；没有就不显示） |
| 源 npz（`clips.csv.source_path`） | 算 3D：`load_skeleton` 取 `manus_*_skeleton`（无 torch）；缺则 FK |

**clip key**（前端唯一 id，与 `export_clips.py` 一致）：

```python
key = f"{io_utils.parse_file_info(source_path).stem}__c{int(clip_id):04d}"
```

### 3.2 写出（本工具产物）

**`out_pose/labels.csv`** — 打标结果，按 `(source_file, clip_id)` 唯一键 upsert。
**刻意不直接写 `clips.csv`**（脚本会重写它），改用独立文件，对上游重跑稳健：

| 列 | 含义 |
|----|------|
| `source_file` | 溯源（联合主键之一） |
| `clip_id` | 溯源（联合主键之一） |
| `gesture_label` | 人填的手势名；空串 = 未标/丢弃 |
| `invalid` | `1` = 人工标记为**无效切片**（过切产生的垃圾段），与 `gesture_label` 互斥；导出时排除 |
| `labeler` | 标注员（前端输入，便于多人协作/质检） |
| `labeled_at` | ISO 时间戳 |
| `note` | 备注（可选，比如"疑似翻手腕""信号噪声"） |

**导出（`GET /api/export`）** — 只导**已打标**的 clip（空标签按约定丢弃），分两层：
- `out_pose/clips_labeled/{subject}__{session}.csv` — **每个源文件夹一份**（文件夹 = `os.path.dirname(source_path)`，即左侧树的 `{subject}/{date-hand}` 会话目录），含该文件夹内全部已标 clip + `gesture_label`/`label_note`。
- `out_pose/clips_labeled.csv` — **最外层合并文件**（全部已标 clip）。

这就是 RUNBOOK §A.5 那份"真值集"，但按文件夹拆开 + 一份合并，避免一个 22 万行的巨型完整 CSV。`/api/export` 返回 `{n_labeled, n_folders, dir, merged}`。

**`out_pose/pose_cache/{key}.json`** — 3D 帧缓存，首次请求时算好落盘，后续直接发。

---

## 4. 后端 API 规格

| 方法 | 路径 | 入参 | 返回 |
|------|------|------|------|
| GET | `/` | — | `index.html` |
| GET | `/app.js` `/plotly.min.js` | — | 静态资源 |
| GET | `/api/recordings` | — | **录制列表**（每条录制一行 + n_clips/n_review/n_labeled，懒加载用） |
| GET | `/api/overview` | `stem` | `image/png`（overview.png） |
| GET | `/api/keyframes` | `key` | `image/png`（6 帧；无则 404） |
| GET | `/api/pose` | `key` | 单 clip 3D 帧 JSON（遗留，当前未用） |
| GET | `/api/recording` | `stem` | **整条录制**的 overview 曲线 + `frame_times` + s1..sN 段落（同步播放用，带缓存） |
| GET | `/api/handframe` | `stem, i` | 第 `i` 帧手部 PNG（matplotlib `draw_skeleton`，ipynb 同款；按需渲染+磁盘缓存） |
| GET | `/api/handstatus` | `stem` | `{done, total}` 后台预渲染进度 |
| POST | `/api/label` | `{key, gesture_label, labeler, note}` | `{ok, labeled, total}` |
| GET | `/api/export` | — | `{written, path, n_labeled}`，并落盘 `clips_labeled.csv` |

**`GET /api/clips` 返回**（一次性发全部，几千行小 JSON 没问题；>5k 再考虑虚拟列表）：

```json
{
  "fs": 2000,
  "labels_used": ["fist", "pinch_index", "one"],
  "clips": [
    {
      "key": "subj__sess__ts__c0003",
      "source_file": "….npz", "clip_id": 3,
      "stem": "subj__sess__ts",
      "group": "subj-left", "subject": "subj", "hand": "left",
      "duration_s": 1.42, "motion_duration_s": 0.46,
      "emg_rms": 31.2, "envelope_peak": 18.7,
      "pose_range": 1.83, "max_pose_speed": 9.1,
      "matched_emg_seg_idx": -1,
      "clip_start_s": 12.30, "clip_end_s": 13.72,
      "rec_duration_s": 64.0,
      "review": true,
      "gesture_label": "", "labeler": "", "note": ""
    }
  ]
}
```

`review = (matched_emg_seg_idx < 0)` → 前端标红、排到队列最前（EMG 漏切，优先复核）。

**`GET /api/pose?key=` 返回**（前端 Plotly.js 直接吃）：

```json
{
  "mode": "skeleton",          // 或 "fk"
  "fps": 30, "n_total": 2840, "step": 36,
  "frames": [[[x,y,z], …J个关节], … F帧],
  "bones": [[0,1,2,3,4], …],   // 连线（chain 索引）
  "palm":  [[0,5,10,15,20]],
  "colors": ["#CC6677", …],    // 每条 finger chain 一色
  "apex_frame": 14,            // 下采样后最接近 apex 的帧
  "structure": {               // 下采样后的帧区间，用于时间轴上色
    "static_in": [0, 5], "motion": [5, 11], "static_out": [11, 19]
  }
}
```

skeleton 模式 J=25（连接表来自 `emg_label/skeleton.py`），FK 模式 J=21（来自 `emg_label/hand3d.py`）。前端只按 `bones`/`palm`/`colors` 画线，对两种模式通用。

---

## 5. 后端实现（`label_server.py`，标准库）

放在仓库根目录，用 `$PY`（emg2pose 的 python）跑。完整可运行：

```python
#!/usr/bin/env python
"""Interactive labelling server for pose clips.

Run:  $PY label_server.py --out out_pose --port 8000
Then open http://127.0.0.1:8000

Reads out/clips.csv (+ recordings.csv, shard overviews, source npz for 3D),
writes out/labels.csv on each label, out/clips_labeled.csv on export.
Stdlib only -- no flask/fastapi. Reuses emg_label for 3D, same as export_clips.
"""
from __future__ import annotations

import argparse, json, os, threading, csv as _csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

from emg_label import io_utils
from emg_label.skeleton import (
    SKELETON_CONNECTIONS, SKELETON_PALM_CONNECTIONS,
    SKELETON_FINGER_COLORS, normalize_skeleton,
)

FS = 2000
POSE_MAX_FRAMES = 60          # downsample target for the 3D animation
WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")


def _clip_key(stem: str, clip_id) -> str:
    return f"{stem}__c{int(clip_id):04d}"


# ----------------------------------------------------------------------------
# State: load clips.csv once, build key->row index, load existing labels.
# A single lock guards label writes (server is the only writer).
# ----------------------------------------------------------------------------
class Store:
    def __init__(self, out_dir: str):
        self.out = out_dir
        self.lock = threading.Lock()
        self.clips_path = os.path.join(out_dir, "clips.csv")
        self.labels_path = os.path.join(out_dir, "labels.csv")
        self.pose_cache_dir = os.path.join(out_dir, "pose_cache")
        os.makedirs(self.pose_cache_dir, exist_ok=True)

        df = pd.read_csv(self.clips_path)
        # recording durations for timeline context
        rec_dur = {}
        rec_path = os.path.join(out_dir, "recordings.csv")
        if os.path.isfile(rec_path):
            r = pd.read_csv(rec_path)
            rec_dur = dict(zip(r["source_file"], r["duration_s"]))

        self.by_key: dict[str, dict] = {}
        self.order: list[str] = []
        self._stem_cache: dict[str, str] = {}
        for _, row in df.iterrows():
            sp = str(row["source_path"])
            stem = self._stem_of(sp)
            key = _clip_key(stem, row["clip_id"])
            rec = {
                "key": key, "stem": stem,
                "source_file": str(row["source_file"]), "source_path": sp,
                "clip_id": int(row["clip_id"]),
                "group": str(row.get("group", "")),
                "subject": str(row.get("subject", "")),
                "hand": str(row.get("hand", "")) or None,
                "clip_start_sample": int(row["clip_start_sample"]),
                "clip_end_sample": int(row["clip_end_sample"]),
                "static_in_start": int(row["static_in_start_sample"]),
                "static_in_end": int(row["static_in_end_sample"]),
                "motion_start": int(row["motion_start_sample"]),
                "motion_end": int(row["motion_end_sample"]),
                "static_out_start": int(row["static_out_start_sample"]),
                "static_out_end": int(row["static_out_end_sample"]),
                "apex": int(row["apex_sample"]),
                "duration_s": float(row["duration_s"]),
                "motion_duration_s": float(row["motion_duration_s"]),
                "emg_rms": float(row["emg_rms"]),
                "envelope_peak": float(row["envelope_peak"]),
                "pose_range": float(row["pose_range"]),
                "max_pose_speed": float(row["max_pose_speed"]),
                "matched_emg_seg_idx": int(row["matched_emg_seg_idx"]),
                "rec_duration_s": float(rec_dur.get(str(row["source_file"]), 0.0)),
            }
            self.by_key[key] = rec
            self.order.append(key)

        self.labels: dict[str, dict] = {}       # key -> label record
        self._load_labels()

    def _stem_of(self, source_path: str) -> str:
        s = self._stem_cache.get(source_path)
        if s is None:
            s = io_utils.parse_file_info(source_path).stem
            self._stem_cache[source_path] = s
        return s

    def _load_labels(self):
        if not os.path.isfile(self.labels_path):
            return
        ldf = pd.read_csv(self.labels_path, dtype=str).fillna("")
        for _, r in ldf.iterrows():
            key = _clip_key(self._stem_for_source(r["source_file"]),
                            int(float(r["clip_id"])))
            self.labels[key] = {
                "gesture_label": r.get("gesture_label", ""),
                "labeler": r.get("labeler", ""),
                "labeled_at": r.get("labeled_at", ""),
                "note": r.get("note", ""),
            }

    def _stem_for_source(self, source_file: str) -> str:
        # source_file -> stem via any clip with that source_file
        for rec in self.by_key.values():
            if rec["source_file"] == source_file:
                return rec["stem"]
        return source_file

    # --- labels list payload ------------------------------------------------
    def clips_payload(self) -> dict:
        used = sorted({l["gesture_label"] for l in self.labels.values()
                       if l["gesture_label"]})
        clips = []
        for key in self.order:
            r = self.by_key[key]
            lab = self.labels.get(key, {})
            clips.append({
                "key": key, "stem": r["stem"],
                "source_file": r["source_file"], "clip_id": r["clip_id"],
                "group": r["group"], "subject": r["subject"], "hand": r["hand"],
                "duration_s": r["duration_s"],
                "motion_duration_s": r["motion_duration_s"],
                "emg_rms": r["emg_rms"], "envelope_peak": r["envelope_peak"],
                "pose_range": r["pose_range"], "max_pose_speed": r["max_pose_speed"],
                "matched_emg_seg_idx": r["matched_emg_seg_idx"],
                "review": r["matched_emg_seg_idx"] < 0,
                "clip_start_s": r["clip_start_sample"] / FS,
                "clip_end_s": r["clip_end_sample"] / FS,
                "rec_duration_s": r["rec_duration_s"],
                "gesture_label": lab.get("gesture_label", ""),
                "labeler": lab.get("labeler", ""),
                "note": lab.get("note", ""),
            })
        return {"fs": FS, "labels_used": used, "clips": clips}

    # --- upsert one label + atomic flush ------------------------------------
    def set_label(self, key, gesture_label, labeler, note) -> dict:
        if key not in self.by_key:
            raise KeyError(key)
        with self.lock:
            self.labels[key] = {
                "gesture_label": gesture_label.strip(),
                "labeler": labeler.strip(),
                "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": note.strip(),
            }
            self._flush_labels()
        labeled = sum(1 for l in self.labels.values() if l["gesture_label"])
        return {"ok": True, "labeled": labeled, "total": len(self.order)}

    def _flush_labels(self):
        tmp = self.labels_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["source_file", "clip_id", "gesture_label",
                        "labeler", "labeled_at", "note"])
            for key, lab in self.labels.items():
                r = self.by_key[key]
                w.writerow([r["source_file"], r["clip_id"],
                            lab["gesture_label"], lab["labeler"],
                            lab["labeled_at"], lab["note"]])
        os.replace(tmp, self.labels_path)

    # --- export clips_labeled.csv (clips.csv left-join labels) --------------
    def export(self) -> dict:
        df = pd.read_csv(self.clips_path)
        glab, gnote = [], []
        for _, row in df.iterrows():
            key = _clip_key(self._stem_of(str(row["source_path"])), row["clip_id"])
            lab = self.labels.get(key, {})
            glab.append(lab.get("gesture_label", ""))
            gnote.append(lab.get("note", ""))
        df["gesture_label"] = glab
        df["label_note"] = gnote
        out = os.path.join(self.out, "clips_labeled.csv")
        tmp = out + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, out)
        n = sum(1 for x in glab if str(x).strip())
        return {"written": len(df), "n_labeled": n, "path": out}

    # --- 3D pose frames (skeleton-first, FK fallback), cached --------------
    def pose(self, key: str) -> dict:
        cache = os.path.join(self.pose_cache_dir, key + ".json")
        if os.path.isfile(cache):
            with open(cache) as f:
                return json.load(f)
        r = self.by_key[key]
        cs, ce = r["clip_start_sample"], r["clip_end_sample"]
        n_total = ce - cs
        step = max(1, n_total // POSE_MAX_FRAMES)
        idxs = list(range(0, n_total, step))[:POSE_MAX_FRAMES]

        side = r["hand"] or io_utils.parse_file_info(r["source_path"]).hand or "left"
        skel = io_utils.load_skeleton(r["source_path"], hand=side)
        if skel is not None:
            clip = skel[cs:ce][idxs]                  # (F,25,3)
            frames = normalize_skeleton(clip)
            bones = SKELETON_CONNECTIONS
            palm = SKELETON_PALM_CONNECTIONS
            colors = SKELETON_FINGER_COLORS
            mode = "skeleton"
        else:
            from emg_label.hand3d import (
                angles_batch_to_landmarks, BONE_CONNECTIONS, PALM_CONNECTIONS)
            _, ja = io_utils.load_npz(r["source_path"], hand=side)
            frames = angles_batch_to_landmarks(ja[cs:ce][idxs], side=side)  # (F,21,3)
            bones, palm = BONE_CONNECTIONS, PALM_CONNECTIONS
            colors = ["#CC6677", "#4477AA", "#228833", "#EE6677", "#AA3377"]
            mode = "fk"

        def to_frame(i):  # absolute sample -> downsampled frame idx
            return min(len(idxs) - 1, max(0, (i - cs) // step))
        out = {
            "mode": mode, "fps": 30, "n_total": n_total, "step": step,
            "frames": np.asarray(frames, dtype=float).round(2).tolist(),
            "bones": [list(map(int, b)) for b in bones],
            "palm": [list(map(int, p)) for p in palm],
            "colors": list(colors),
            "apex_frame": to_frame(r["apex"]),
            "structure": {
                "static_in": [to_frame(r["static_in_start"]), to_frame(r["static_in_end"])],
                "motion": [to_frame(r["motion_start"]), to_frame(r["motion_end"])],
                "static_out": [to_frame(r["static_out_start"]), to_frame(r["static_out_end"])],
            },
        }
        with open(cache, "w") as f:
            json.dump(out, f)
        return out


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------
def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quieter
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype):
            if not os.path.isfile(path):
                return self._send(404, {"error": "not found"})
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            p = u.path
            if p == "/":
                return self._file(os.path.join(WEBUI_DIR, "index.html"), "text/html")
            if p in ("/app.js", "/plotly.min.js"):
                ct = "text/javascript"
                return self._file(os.path.join(WEBUI_DIR, p.lstrip("/")), ct)
            if p == "/api/clips":
                return self._send(200, store.clips_payload())
            if p == "/api/overview":
                stem = q.get("stem", [""])[0]
                return self._file(
                    os.path.join(store.out, "shards", stem, "overview.png"),
                    "image/png")
            if p == "/api/keyframes":
                key = q.get("key", [""])[0]
                return self._file(
                    os.path.join(store.out, "clips_export", key + ".png"),
                    "image/png")
            if p == "/api/pose":
                key = q.get("key", [""])[0]
                try:
                    return self._send(200, store.pose(key))
                except Exception as ex:
                    return self._send(500, {"error": f"{type(ex).__name__}: {ex}"})
            if p == "/api/export":
                return self._send(200, store.export())
            return self._send(404, {"error": "no route"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/label":
                return self._send(404, {"error": "no route"})
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            try:
                res = store.set_label(
                    data["key"], data.get("gesture_label", ""),
                    data.get("labeler", ""), data.get("note", ""))
                return self._send(200, res)
            except KeyError as ex:
                return self._send(400, {"error": f"unknown key {ex}"})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_pose")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    store = Store(args.out)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Labelling {len(store.order)} clips from {args.out}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
```

**实现要点：**

- **复用、不重写**：3D 走 `io_utils.load_skeleton`（无 torch）→ 失败回退 `hand3d` FK，和 `export_clips.py` 同一策略；连接表/配色直接 import。
- **键一致**：`_clip_key` 与 `export_clips._clip_key` 同式，所以 `clips_export/{key}.png` 能直接拿来当 6 帧参考。
- **写安全**：单进程 + `threading.Lock`，每次 upsert 后**原子写**（tmp→`os.replace`）整张 `labels.csv`；几千行重写成本可忽略，省掉行内改写的坑。
- **断点续标**：启动读已存在的 `labels.csv` 进内存，刷新页面/重启进程都不丢。
- **缓存**：3D 帧 JSON 落 `pose_cache/{key}.json`，第二次秒开；下采样到 ≤60 帧控制体积。
- **只读上游**：绝不写 `clips.csv`；标签独立存 `labels.csv`，导出时才 join。

---

## 6. 前端实现（`webui/`）

三个文件：`webui/index.html`、`webui/app.js`、`webui/plotly.min.js`（从 https://cdn.plot.ly/plotly-2.35.2.min.js 下载放进来，或 `index.html` 里直接引 CDN）。

### 6.1 布局

```
┌────────────────────────────────────────────────────────────────┐
│ 顶栏：进度 [■■■□□ 123/512]  过滤[未标▾]  排序[复核优先▾]         │
│        labeler:[___]  [导出 clips_labeled.csv]                    │
├────────────────┬───────────────────────────────────────────────┤
│ 左：录制树      │ overview.png（整条录制）      │  3D 手（ipynb 同款）│
│ ▾ ax-0819/      │  EMG/pose-speed/matched      │   [手部图]          │
│   20260428-left │  橙=clip，蓝竖线=playhead      │   与 playhead 同步   │
│   ● ..114049 3/8│ transport:[▶][═══●══进度条══] 1.23/4.00s 1×        │
│   ● ..114308 0/4│ 段落条 [s1 fist][s2 ⚠][s3]…（绿=已标/灰=未标/红框） │
│ ▸ bx-0712/…     ├───────────────────────────────────────────────┤
│   (折叠)        │ （中间区可滚动）                                 │
│ (folder 进度     │                                                │
│  绿=标完/红=复核)│                                                │
├────────────────┴───────────────────────────────────────────────┤
│ 下方：meta  gesture_label:[____▾]  note:[___]  [保存并下一个⏎][跳过] │
│        热键 ⏎保存 · ↑↓切换段 · 空格播放 · 1-9 常用标签             │
└──────────────────────────────────────────────────────────────────┘
```
- **左侧按源文件夹折叠**：录制按 `os.path.dirname(source_path)`（即 `{subject}/{date-hand}` 会话目录）分组，folder 头可点击展开/折叠，显示该文件夹 `已标/总 clip` + 复核数；搜索时自动展开命中的文件夹。
- **3D 手在 overview 右侧**（`#viz` 横向：ovwrap 占主宽 + handwrap 固定 ~330px）。
- **标注栏在页面最下方**（`#bottom` 横条：meta + label/note + 保存/跳过 + 热键）。
```
```

### 6.2 `index.html`

```html
<!doctype html><meta charset=utf-8><title>Clip 打标</title>
<script src="/plotly.min.js"></script>
<style>
  body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;color:#222;height:100vh;display:flex;flex-direction:column}
  #top{display:flex;gap:16px;align-items:center;padding:8px 14px;background:#f4f4f4;border-bottom:1px solid #ddd;font-size:13px}
  #main{flex:1;display:flex;min-height:0}
  #list{width:300px;overflow:auto;border-right:1px solid #eee;font-size:12px}
  #list .row{padding:6px 10px;border-bottom:1px solid #f2f2f2;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #list .row.sel{background:#e8f0fe}
  #list .row.done{border-left:4px solid #2e7d32}
  #list .row.review{border-left:4px solid #c62828}
  #list .row.todo{border-left:4px solid #ccc}
  #center{flex:1;overflow:auto;padding:12px;display:flex;flex-direction:column;gap:10px}
  #overview img{max-width:100%;border:1px solid #ddd}
  #timeline{height:18px;background:#eee;position:relative;border:1px solid #ccc}
  #timeline .band{position:absolute;top:0;bottom:0;background:rgba(255,140,0,.6)}
  #plot{width:100%;height:460px}
  #keyframes img{max-width:100%;border:1px solid #eee}
  #side{width:300px;padding:12px;border-left:1px solid #eee;font-size:13px}
  #side input{width:100%;padding:6px;font-size:14px;margin:4px 0}
  .btn{padding:7px 12px;margin:4px 4px 4px 0;cursor:pointer}
  .chip{display:inline-block;background:#eee;border-radius:10px;padding:1px 8px;margin:1px;font-size:11px}
  .warn{color:#c62828;font-weight:600}
  .qc{color:#555;font-size:12px;line-height:1.6}
</style>
<div id=top>
  <span id=progress>0/0</span>
  <label>过滤 <select id=filter>
    <option value=all>全部</option>
    <option value=todo selected>未标</option>
    <option value=review>仅复核(burst=-1)</option>
    <option value=done>已标</option>
  </select></label>
  <label>排序 <select id=sort>
    <option value=review>复核优先</option>
    <option value=order>原序</option>
  </select></label>
  <label>labeler <input id=labeler style="width:90px" value=""></label>
  <button class=btn id=export>导出 clips_labeled.csv</button>
  <span id=msg></span>
</div>
<div id=main>
  <div id=list></div>
  <div id=center>
    <div id=overview></div>
    <div id=timeline><div class=band></div></div>
    <div id=plot></div>
    <div id=keyframes></div>
  </div>
  <div id=side>
    <div id=meta class=qc></div>
    <label>gesture_label</label>
    <input id=label list=labellist autocomplete=off>
    <datalist id=labellist></datalist>
    <label>note</label>
    <input id=note autocomplete=off>
    <div>
      <button class=btn id=save>保存并下一个 ⏎</button>
      <button class=btn id=skip>跳过(丢弃)</button>
    </div>
    <div id=hotkeys class=qc>⏎ 保存并下一个 · ↑↓ 切换 · 空格 播放/暂停 · 1-9 套用常用标签</div>
    <div id=common></div>
  </div>
</div>
<script src="/app.js"></script>
```

### 6.3 `app.js`

```javascript
let STATE = { clips: [], view: [], idx: 0, labels_used: [], pose: null, anim: null };

async function boot() {
  const d = await (await fetch('/api/clips')).json();
  STATE.fs = d.fs;
  STATE.clips = d.clips;
  STATE.labels_used = d.labels_used;
  refreshDatalist();
  applyFilter();
  document.getElementById('filter').onchange = applyFilter;
  document.getElementById('sort').onchange = applyFilter;
  document.getElementById('save').onclick = () => saveLabel(true);
  document.getElementById('skip').onclick = () => { setLabelInput(''); saveLabel(true); };
  document.getElementById('export').onclick = doExport;
  document.onkeydown = onKey;
}

function applyFilter() {
  const f = document.getElementById('filter').value;
  const s = document.getElementById('sort').value;
  let v = STATE.clips.filter(c => {
    if (f === 'todo') return !c.gesture_label;
    if (f === 'review') return c.review && !c.gesture_label;
    if (f === 'done') return !!c.gesture_label;
    return true;
  });
  if (s === 'review') v.sort((a, b) => (b.review - a.review));  // review first
  STATE.view = v;
  STATE.idx = Math.min(STATE.idx, Math.max(0, v.length - 1));
  renderList();
  if (v.length) select(STATE.idx);
  renderProgress();
}

function renderProgress() {
  const done = STATE.clips.filter(c => c.gesture_label).length;
  document.getElementById('progress').textContent = `${done}/${STATE.clips.length}`;
}

function renderList() {
  const el = document.getElementById('list');
  el.innerHTML = '';
  STATE.view.forEach((c, i) => {
    const cls = c.gesture_label ? 'done' : (c.review ? 'review' : 'todo');
    const div = document.createElement('div');
    div.className = `row ${cls}${i === STATE.idx ? ' sel' : ''}`;
    div.textContent = `c${String(c.clip_id).padStart(4,'0')} ${c.gesture_label || (c.review ? '⚠未' : '未')}`;
    div.title = c.key;
    div.onclick = () => select(i);
    el.appendChild(div);
  });
}

function refreshDatalist() {
  const dl = document.getElementById('labellist');
  dl.innerHTML = STATE.labels_used.map(l => `<option value="${l}">`).join('');
  document.getElementById('common').innerHTML =
    '常用: ' + STATE.labels_used.slice(0, 9)
      .map((l, i) => `<span class=chip>${i+1}:${l}</span>`).join('');
}

async function select(i) {
  if (i < 0 || i >= STATE.view.length) return;
  STATE.idx = i;
  const c = STATE.view[i];
  renderList();
  // meta + inputs
  document.getElementById('meta').innerHTML =
    `<b>${c.key}</b><br>group ${c.group} · hand ${c.hand||'?'}<br>`+
    `dur ${c.duration_s.toFixed(2)}s · motion ${c.motion_duration_s.toFixed(2)}s<br>`+
    `env_peak ${c.envelope_peak.toFixed(1)} · pose_range ${c.pose_range.toFixed(2)}<br>`+
    (c.review ? `<span class=warn>matched_burst = -1（优先复核）</span>` : `burst ${c.matched_emg_seg_idx}`);
  setLabelInput(c.gesture_label || '');
  document.getElementById('note').value = c.note || '';
  // overview + timeline
  document.getElementById('overview').innerHTML =
    `<img src="/api/overview?stem=${encodeURIComponent(c.stem)}">`;
  const band = document.querySelector('#timeline .band');
  const dur = c.rec_duration_s || (c.clip_end_s + 1);
  band.style.left = (100 * c.clip_start_s / dur) + '%';
  band.style.width = (100 * (c.clip_end_s - c.clip_start_s) / dur) + '%';
  // keyframes (optional)
  document.getElementById('keyframes').innerHTML =
    `<img src="/api/keyframes?key=${encodeURIComponent(c.key)}" `+
    `onerror="this.style.display='none'">`;
  // 3D
  loadPose(c.key);
  document.getElementById('label').focus();
}

async function loadPose(key) {
  const d = await (await fetch('/api/pose?key=' + encodeURIComponent(key))).json();
  if (d.error) { Plotly.purge('plot'); return; }
  STATE.pose = d;
  const traces = frameTraces(d, d.apex_frame);  // start at apex (most recognisable)
  const layout = {
    margin: {l:0,r:0,t:0,b:0}, showlegend: false,
    scene: { aspectmode: 'cube', xaxis:{visible:false}, yaxis:{visible:false}, zaxis:{visible:false},
             camera:{eye:{x:-0.25,y:-1.6,z:0.6}} },
  };
  Plotly.react('plot', traces, layout, {displaylogo:false});
}

function frameTraces(d, fi) {
  const f = d.frames[fi];                 // [[x,y,z],...J]
  const xs = f.map(p=>p[0]), ys = f.map(p=>p[1]), zs = f.map(p=>p[2]);
  const traces = [{ type:'scatter3d', mode:'markers', x:xs, y:ys, z:zs,
                    marker:{size:4,color:'#333'} }];
  d.bones.forEach((b, ci) => {
    traces.push({ type:'scatter3d', mode:'lines',
      x:b.map(j=>f[j][0]), y:b.map(j=>f[j][1]), z:b.map(j=>f[j][2]),
      line:{width:6, color:d.colors[ci % d.colors.length]} });
  });
  d.palm.forEach(p => {
    traces.push({ type:'scatter3d', mode:'lines',
      x:p.map(j=>f[j][0]), y:p.map(j=>f[j][1]), z:p.map(j=>f[j][2]),
      line:{width:2, color:'#888', dash:'dot'} });
  });
  return traces;
}

function playPose() {
  const d = STATE.pose; if (!d) return;
  if (STATE.anim) { clearInterval(STATE.anim); STATE.anim = null; return; }
  let fi = 0;
  STATE.anim = setInterval(() => {
    Plotly.react('plot', frameTraces(d, fi), document.getElementById('plot').layout);
    fi = (fi + 1) % d.frames.length;
  }, 1000 / d.fps);
}

function setLabelInput(v){ document.getElementById('label').value = v; }

async function saveLabel(advance) {
  const c = STATE.view[STATE.idx]; if (!c) return;
  const gesture_label = document.getElementById('label').value.trim();
  const note = document.getElementById('note').value.trim();
  const labeler = document.getElementById('labeler').value.trim();
  await fetch('/api/label', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ key:c.key, gesture_label, labeler, note }) });
  // update local mirror
  const orig = STATE.clips.find(x => x.key === c.key);
  orig.gesture_label = gesture_label; orig.note = note;
  c.gesture_label = gesture_label; c.note = note;
  if (gesture_label && !STATE.labels_used.includes(gesture_label)) {
    STATE.labels_used.push(gesture_label); STATE.labels_used.sort(); refreshDatalist();
  }
  renderProgress(); renderList();
  document.getElementById('msg').textContent = '已保存 ' + new Date().toLocaleTimeString();
  if (advance) nextTodo();
}

function nextTodo() {
  // next item in current view (prefer next unlabeled)
  for (let j = STATE.idx + 1; j < STATE.view.length; j++)
    if (!STATE.view[j].gesture_label) return select(j);
  if (STATE.idx + 1 < STATE.view.length) return select(STATE.idx + 1);
}

function onKey(e) {
  if (e.key === 'Enter') { e.preventDefault(); saveLabel(true); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); select(STATE.idx + 1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); select(STATE.idx - 1); }
  else if (e.key === ' ' && e.target.tagName !== 'INPUT') { e.preventDefault(); playPose(); }
  else if (/^[1-9]$/.test(e.key) && e.target.id !== 'note') {
    const l = STATE.labels_used[+e.key - 1];
    if (l && document.activeElement.id !== 'label') { setLabelInput(l); saveLabel(true); }
  }
}

async function doExport() {
  const r = await (await fetch('/api/export')).json();
  document.getElementById('msg').textContent =
    `导出 ${r.n_labeled}/${r.written} 行 → ${r.path}`;
}

boot();
```

---

## 6.5 同步播放 + 段落进度（已实现，覆盖原静态 overview）

需求：**进度条拖动/播放时，overview 与 3D 手势同步**；最下方显示 `s1 s2 …` 段落，绿=已标、灰=未标。

实现把"静态 overview.png + 单 clip 3D"升级为**录制级同步视图**：

- **后端 `GET /api/recording?stem=`**（`label_server.Store._recording_geometry`）：把整条录制
  - 下采样为 ≤600 帧 3D（skeleton 优先 / FK 回退，和单 clip 同源）；附 `frame_times`（每帧秒数）；
  - 顺带算 `pose_speed`（`pose_segmentation.pose_speed`）和 `emg_env`（`segmentation.emg_envelope`）两条曲线，下采样到 ≤2000 点 —— **几乎零额外成本**，因为录制已经 load 进来了；
  - 返回该录制全部 clip，按时间排序编号 `s1..sN`，带 `start_s/motion_*/apex_s/review` 几何；
  - 几何缓存到 `recording_cache/{stem}.json`；**标签每次请求实时叠加**（缓存与标签解耦）。
- **上方 = 录制的 `overview.png`**（`segment.py` 的 `plot_overview_dual` 三行图：EMG 包络+burst / pose-speed+clip / matched s 段）：直接由 `/api/overview?stem=` 发图，前端 `<img>` 显示，上面叠一根 CSS playhead 竖线 + 点击跳转。
- **overview 缺失时从中间数据重新生成**：默认优先用 `segment.py` 调好的静态 `overview.png`；**找不到时**（如 `--no-overview`、shard 命名对不上、新数据未切）后端用**同一个 `plot_overview_dual`** 从中间数据现生成——`env`/`pose_speed` 服务端算，burst 段取自 `segments.csv`，clip 段取自 `clips.csv`，阈值取自 `recordings.csv`，`clip_to_burst` 用 `matched_emg_seg_idx`——产物与切分时的图**逐像素同款**，缓存到 `overview_cache/{stem}.{png,json}`。现生成时**在渲染瞬间直接捕获坐标轴框** → playhead `ov_x0/ov_x1` 精确，无需检测。设 `OVERVIEW_REGEN=1`（如 `OVERVIEW_REGEN=1 ./pulse.sh label`）则**一律现生成**（忽略静态图，便于在服务端改样式而不必重切全量数据）。
- **playhead 对位靠从 PNG 检测数据轴左右边（静态图时）**：`plot_overview_dual` 用 `tight_layout`，其左右边距既随 y 轴刻度宽度变，又会在标签拥挤时**失败回退到 matplotlib 默认 0.125/0.9**——所以固定常量必然偏（实测旧常量 0.049 在起点偏 ~11px）。后端 `_overview_xrange` 直接读 overview.png 里**海军蓝 pose-speed 曲线**横跨的列范围 = 数据轴 `x0..x1`（图像分数），~1px 精度、对 tight_layout 成功/失败都自洽（60/60 真实 overview 全检出），写进 `/api/recording` 的 `ov_x0/ov_x1`，缓存于 geometry v4。前端优先用它，检测失败才回退常量 `OVX0=0.054/OVX1=0.989`。
  > 早先做过浏览器内 Plotly 曲线版，但用户要看 overview.png 本身（信息更全：burst/clip/阈值/matched 行）。已**移除 Plotly 依赖**（手部与 overview 都是图片）。后端仍算 `curves`（pose_speed/EMG）但前端当前不用，保留以备回退。若要 playhead 像素级精确，可让 `plot_overview_dual` 落一个 `overview.bbox.json` 边距 sidecar，前端优先用它。
- **3D 手 = emg2pose 手部网格(mesh)，前端 Plotly.js `Mesh3d` 渲染**：演进过——①最初 Plotly scatter3d 点线（Y 轴薄、看成侧面、看不清动作）→ ②服务端 matplotlib `draw_skeleton` 静态 PNG（清楚但点线、不能转）→ ③**现在用 `seg_vis_all.ipynb` 同款 `emg2pose.visualization` 的手部网格**：清晰的着色面 + **可旋转/缩放**(WebGL) + 与进度条同步。后端 `skin_vertices_np(profile, joint_angles[i])` 算 788 顶点(emg2pose 蒙皮，缓存 hand model 后 ~6ms/帧)；`/api/handmesh?stem=&i=` 发该帧顶点(int mm, ~12KB)落盘 `mesh_cache/{stem}/f{i}.json`，`/api/handfaces` 发三角面(1544，左右手相同，一次)。前端 `Plotly.newPlot` 一个 `Mesh3d`，scrub/播放时 `Plotly.restyle` 换顶点 x/y/z；`scene.uirevision` 保留用户旋转视角。左手用镜像 profile(`mirror_profile`，对应 ipynb 的 `flip`)。需 torch+emg2pose（环境已具备）。
- **关键单位坑(deg→rad)**：Manus `*_ergonomics` 是**度**(range ~±60)，emg2pose 蒙皮/FK 要**弧度**。直接喂度会把五指拧成一团。`_mesh_prep` 里当 `max|ja|>2π` 判定为度并 `×π/180`（弧度数据如 emg2pose 自带 joint_angles 不受影响）。缓存目录加版本 `mesh_cache_v2` 让旧的(扭曲)缓存失效。⚠️ 同一坑也存在于聚类路径的 `hand3d` FK（`plot_cluster_hands` 等也喂了度）——本次未改，如要正确的聚类 3D 手图需同样转弧度。
- **播放为什么之前"只在暂停时刷新"**：三个原因叠加，已全部修掉——
  1. **没有浏览器预载**：播放时每帧改 `<img>.src` 触发网络加载，下一帧的 src 在上一张加载完前就把它**取消**了，于是只有停下来（暂停）最后一张才加载完。→ 现在前端 `preloadHands` 把全部帧用 `Image()` 预载进**浏览器缓存**，播放时换 src 命中缓存、瞬时上屏、不取消。
  2. **后台预渲染线程抢 GIL**：原来 `/api/recording` 起一个线程连续 matplotlib 渲染，CPU 占满 GIL，把"当前帧"的按需请求饿死。→ 去掉后台线程，渲染改由前端预载请求驱动（受 `onload` 节流），`/api/handstatus` 改成数磁盘 PNG 数报进度。
  3. **60fps 重排**：`setTime` 每个 rAF 调一次 `Plotly.relayout` 把主线程占满、图片解码上不了屏。→ 播放循环按真实时间推进但**节流到 ~25fps 才重绘**。
- **同步引擎**（`app.js`）：单一时间 `STATE.t` 驱动 playhead（`#ovhead` CSS `left%`）+ 手部（`<img>.src` 按 `frame_times` 最近邻取帧，`curHandFrame` 去重）。拖动进度条/▶播放/点击 overview 都只改 `STATE.t` 后调 `setTime()`，天然同步。
- **首次加载**：首帧 ~2.6s（load_skeleton），随后 600 帧在后台预载渲染（~50–120ms/帧）。手部图顶部一条**进度条**（`#handprog`，蓝色填充 + 「渲染手部 N/600」），渲完变绿显示「**渲染完成 ✓**」约 1.5s 后淡出；已渲染帧可立即 scrub/播放，未渲染帧按需即时补。整条录制全部缓存后再开秒进。
- **左侧 3D 缓存标识**：每条录制名前一个小圆点——灰=无缓存 / 橙=部分 / 绿=已全部渲染；文件夹头显示 `🎬{已缓存录制数}`。后端按 `hand_frames/{stem}/` 的 PNG 数 vs 由时长推出的 `n_frames`（不读 npz）判定 `cached∈{0,1,2}`；前端随渲染进度（`markCached`）实时更新。
- **底部段落条 `s1..sN`**：绿=已标 / 灰=未标 / 红框=`burst=-1` 复核；蓝色外框=当前打标目标。点 chip → 选中该段并把 playhead 跳到它的 **clip 起点**（`start_s`，对齐 overview 里该段左边缘；早先跳 apex 落在末尾的持姿期，易误以为"跳到结尾"）。保存后该 chip 立即变绿。
- **overview 第三行（row3）显示全部 clip + burst**：上半=所有 clip（橙色，按时间 `s1..sN` 编号，与底部 chip **数量/编号一致**；matched 的更深更实，clip-only 的浅），下半=所有 burst（绿色 `b0..`）；段数 >40 时省略文字标号仅留色块。修了之前 row3 只画 matched 段导致与 chip 数量对不上的问题。改的是 `plot_overview_dual`（切分时也会用到），**要 `OVERVIEW_REGEN=1` 现生成或重跑 segment.py 才能在旧的静态图上看到新 row3**。

> 单 clip 的 `/api/pose`（apex 起、clip 内更细的 60 帧）保留作备用，当前主视图走 `/api/recording`。

### 6.5.1 手渲染对齐 ipynb 的要点（务必保留）

后端 `_hand_prep` / `_render_hand` 必须复刻 notebook（`render_segmentation_60s.ipynb`）的变换，否则手"看不出动作"：

1. **米 → 毫米**：raw Manus `manus_*_skeleton` 的 XYZ 是**米**（±0.14 m）。渲染前 `normalize_skeleton(skel) * 1000.0`（与 notebook 一致）。FK 回退路径（`angles_batch_to_landmarks`）已是 mm，**不要再 ×1000**。
2. **算 cube 前把 inf 变 nan**：raw 骨架含约 0.006% 的 `inf/nan`（遮挡丢帧）。整条录制做 `axis_limits` 时，`inf` 会让 `nanmin/nanmax` 返回 inf → cube span 退化成兜底的 1.0 → 手被缩成一点/裁掉。所以 `_hand_prep` 先 `arr[~isfinite]=nan` 再 `axis_limits(arr)`（`axis_limits` 忽略 nan、且对全坏有 span=1 兜底）。
3. **视角**：`ax.view_init(elev=30, azim=-60)`（matplotlib 3D 默认 = notebook）。WebGL 时代用 Plotly 相机看成手的薄边（Y 轴仅 ±~40mm）→ 动作投影几乎为零，这正是之前"看不出动作"的根因；matplotlib 这个 3/4 俯视角把五指铺开，动作清晰。

> EMG / pose-speed 曲线**保持本项目 overview 的实现**（Plotly 双轴曲线 + clip 色块 + playhead），不回退成 notebook 的 matplotlib 三联面板——notebook 只作 3D 手渲染的参照。
> 遗留的 `_frames_json()`（非有限坐标→`None`，JSON 合法化）仅服务于未用的 `/api/pose`，可保留备用。

## 7. 打标交互设计要点

| 设计 | 理由 |
|------|------|
| **复核优先排序** | `matched_emg_seg_idx == -1`（EMG 漏切）排队首并标红——METHOD §3.5 指出这批最该先看 |
| **过滤未标/已标** | 大量 clip 时只看待办，断点续标不重复 |
| **标签自动补全 + 1-9 热键** | 手势词表小且复用率高；`datalist` 防拼写漂移，数字键秒套常用标签 |
| **过切 + 人工剔除** | 切分降低门槛多切（`POSE_PCT`/`MIN_STATIC_S`/`MIN_MOTION_S` 调低，见下），打标时把垃圾段按 `x`/「标记无效」剔除 |
| **三态：未标/已标/无效** | 空标签=未看；填手势名=有效；`invalid=1`=看过且判为垃圾。三者在 chip/录制树/进度里区分（绿/✗灰/灰），"已看完"= 已标+无效 覆盖全部 clip |
| **空标签 = 丢弃** | 未标与无效都不进真值集；区别是"无效"代表你已审阅过、不必再看 |
| **3D 默认停在 apex 帧** | apex 是最到位的定型姿态（METHOD §6.2），一眼可辨；空格键播放看全过程 |
| **overview + 时间轴条** | 给出该 clip 在整条录制里的位置和邻段关系，判断是否切碎/粘连 |
| **6 帧 png 作快速参考** | 已有产物零成本复用；3D 没渲染成功时仍有图可看 |
| **自动保存 + 时间戳 + labeler** | 每次回车即落盘；多人协作可追溯谁标的 |

**overview 高亮**：overview.png 由 matplotlib `tight_layout` 生成，数据区像素框不固定，**不在图上做像素级覆盖**；改用图下方一条 JS 时间轴条（按 `clip_start_s / rec_duration_s` 精确定位），既准又与 matplotlib 解耦。图里本来就画了 `c{clip_id}` 文字标号可对照。
> 可选增强：让 `plot_overview_dual` 额外吐一个 `overview.bbox.json`（数据轴的 figure 分数坐标），前端就能在图上画精确高亮带。非必需。

---

## 8. 部署与运行

```bash
PY=/home/chenglin/anaconda3/envs/emg2pose/bin/python

# 1) 准备前端 plotly（二选一）
#    a. 下载 vendored：
curl -L https://cdn.plot.ly/plotly-2.35.2.min.js -o webui/plotly.min.js
#    b. 或把 index.html 里的 <script src="/plotly.min.js"> 改成 CDN 链接

# 2) 跑服务（先确保 out_pose/ 已由 segment.py + export_clips.py 生成）
$PY label_server.py --out out_pose --port 8000

# 3) 浏览器打开 http://127.0.0.1:8000，开始打标
# 4) 标完点"导出" → out_pose/clips_labeled.csv（即真值集）
```

**接入 `pulse.sh`**（加一个子命令，和现有风格一致）：

```bash
# 在 pulse.sh 的 case 分支里加：
  label)
    : "${RAW_OUT:=out_pose}" ; : "${PORT:=8000}"
    exec "$PY" label_server.py --out "$RAW_OUT" --port "$PORT" ;;
```

用法：`PORT=8000 ./pulse.sh label`。RUNBOOK §A.5 的"填表"那步即可替换为本网页。

**可选：提前批量渲染手部帧（首开即秒进）**

```bash
WORKERS=8 ./pulse.sh label-prewarm            # 渲染 out_pose/clips.csv 里全部录制
# 或子集：
python prewarm_hands.py --out out_pose --workers 8 --limit 50
python prewarm_hands.py --out out_pose --workers 8 --subject pgy-0226
```

每条录制独立一个进程（matplotlib 进程内单线程），并行跨录制；幂等（已渲染的跳过）。落盘 `out_pose/hand_frames/`，与按需渲染同一缓存。约 ~30s/录制（单进程），8 进程 ~4s/录制。

远程服务器场景：`ssh -L 8000:127.0.0.1:8000 user@host` 端口转发后本地浏览器访问，服务端仍 `--host 127.0.0.1` 不对外暴露。

---

## 9. 与现有流程衔接

```
segment.py ──► clips.csv / recordings.csv / shards/*/overview.png
export_clips.py ──► clips_export/{key}.npz + {key}.png        (6 帧参考)
        │
        ▼
label_server.py（本工具）── 浏览器交互打标
        │  POST /api/label → labels.csv（增量、可断点）
        ▼  GET /api/export
clips_labeled.csv  ＝ 真值集（替代 RUNBOOK §A.5 手填那份）
```

`clips_labeled.csv` 列 = `clips.csv` 全列 + 填好的 `gesture_label` + `label_note`，下游用法与原约定一致（空标签丢弃、同名合并）。

---

## 10. 边界与性能

| 项 | 处理 |
|----|------|
| clip 数很多（22 万+，5300 录制） | **按录制懒加载**（已实现）：`Store` 用 pandas groupby 建每条录制的汇总（无 22 万行 Python 循环，启动 ~2s）；`/api/recordings` 只发录制级汇总（~1.2MB）；某条录制的 clip 仅在 `/api/recording?stem=` 选中时从 df 切片构建。左侧列录制（带 `n_labeled/n_clips`、复核数、搜索/过滤），选中才载入其 clip（底部 s 段落条）。指向顶层 `out_pose2` 即可标全部，不卡。 |
| 手部帧缓存大小 | 单帧 PNG ~13 KB → **单条录制 600 帧 ≈ 8 MB（磁盘 ~10 MB，含小文件块开销）**；落盘 `hand_frames/{stem}/f*.png`，**跨重启复用**，第二次打开秒进。100 条 ≈ 0.8 GB、1000 条 ≈ 8 GB、全量 ~9330 条 ≈ 73 GB（磁盘通常够，瓶颈是渲染时间）。 |
| 首次打开慢（每条一次性） | 首帧 ~2.6s（`load_skeleton`）+ 后台逐帧渲染 ~50ms/帧；要"首开即秒进"用 `prewarm_hands.py` 提前批量渲染（见下）。 |
| 源 npz 在慢 NFS | skeleton 只读 `manus_*_skeleton`+时间戳；缓存后不再读源 |
| 无 torch | 自动走 skeleton 路径（raw 数据都有）；processed 数据无 skeleton 才需 torch 走 FK |
| 上游重跑改了 clips.csv | 标签按 `(source_file, clip_id)` 存，clip 数变了也能 re-join；重启服务重读 |
| 并发写 | 单进程 + Lock + 原子替换；不建议多实例同写一个 `out_pose` |
| 空标签语义 | 导出时空 `gesture_label` 原样保留（空=丢弃）；下游按约定过滤 |

---

## 11. 测试

最小冒烟（沿用 `tests/` 风格，pytest）：

```python
# tests/test_label_server.py
def test_clip_key_matches_export():
    from label_server import _clip_key
    assert _clip_key("subj__sess__ts", 3) == "subj__sess__ts__c0003"

def test_store_roundtrip(tmp_path):
    # 造一个最小 clips.csv（含 source_path 指向一个含 skeleton 的小 npz），
    # Store(tmp).set_label(key,'fist',...) 后断言 labels.csv 写出且 export() join 正确。
    ...
```

手动验证清单：

- [ ] `GET /api/clips` 行数 == `clips.csv` 行数，`review` 标记与 `matched_emg_seg_idx<0` 一致
- [ ] 选中 clip → overview 显示、时间轴条位置正确、3D 可旋转、空格可播放
- [ ] 回车保存 → `labels.csv` 立刻出现该行；刷新页面标签仍在（断点续标）
- [ ] 导出 → `clips_labeled.csv` 行数 == clips.csv，`gesture_label` 已填
- [ ] 无 torch 环境（raw 数据）3D 仍出图（skeleton 路径）

---

## 12. 实施步骤清单

1. `webui/` 放 `index.html` / `app.js`，下载 `plotly.min.js`（§6、§8）。
2. 仓库根加 `label_server.py`（§5）。
3. 确保 `out_pose/` 已有 `clips.csv` / `recordings.csv` / `shards/*/overview.png`（`./pulse.sh raw ...`），`clips_export/*.png` 可选。
4. `$PY label_server.py --out out_pose` 起服务，浏览器自测（§11 清单）。
5. `pulse.sh` 加 `label)` 子命令（§8）。
6. 标完 `GET /api/export` → `clips_labeled.csv`，接回真值集流程。
7. （可选）加 `test_label_server.py`、`pose_cache` 预热脚本、overview bbox 精确高亮。

---

## 13. 后续可选增强

- **质检视图**：按 `gesture_label` 分组看同标签的 3D 缩略图墙，抓离群/错标。
- **多人分工**：`/api/clips?assignee=` 按 labeler 切片；冲突检测（同 clip 多人标不一致时高亮）。
- **键盘流**：完全脱离鼠标（↑↓选段、数字键贴标签、回车跳下一个）。
- **EMG/角度曲线**：在 3D 下方加该 clip 的 16 通道 EMG + 20 关节角小图（复用 `visualize_segment.py` 思路），辅助判断。
- **撤销/历史**：`labels.csv` 已带时间戳，可加一个 `/api/undo` 回退上一次。
```
