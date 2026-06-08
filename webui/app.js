// Two-level navigation: left panel lists RECORDINGS; selecting one lazily
// loads its clips (the bottom s1..sN chips). Labelling acts on the current clip.
let STATE = {
  recordings: [], recView: [], recIdx: 0,
  folderPrefix: '',     // common path prefix stripped from folder labels
  expanded: new Set(),  // expanded folder keys in the left tree
  rec: null,            // loaded recording payload (segments=clips + frame grid)
  cur: null,            // currently selected clip (segment) = labelling target
  labels_used: [], common_labels: [],   // labels_used = autocomplete; common_labels = curated quick-set palette
  total_clips: 0, total_labeled: 0, total_invalid: 0, total_invalid_rec: 0,
  t: 0, playing: false, raf: null,
  curHandFrame: -1, handPoll: null, handEpoch: 0,
  faces: null, wantFrame: 0, meshReady: false,   // 3D hand mesh state
  handVerts: [], handN: 0,   // in-memory per-frame vertices (preloaded -> jitter-free playback)
};

const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
// Folder label = path with the prefix shared by ALL recordings stripped, so the
// common data root (e.g. .../processed_data, == DATA_ROOT) doesn't repeat on
// every row. Computed once from the recordings list (STATE.folderPrefix); falls
// back to the last path segment when nothing distinctive remains.
function commonFolderPrefix(folders) {
  const uniq = [...new Set(folders)].filter(Boolean);
  if (uniq.length < 2) return '';            // one folder: nothing shared to strip
  const segs = uniq.map(f => f.split('/'));
  let i = 0;
  while (segs.every(p => i < p.length - 1 && p[i] === segs[0][i])) i++;  // keep ≥1 tail seg
  return segs[0].slice(0, i).join('/');
}
const folderShort = f => {
  const pre = STATE.folderPrefix;
  if (pre && f.startsWith(pre)) {
    const s = f.slice(pre.length).replace(/^\/+/, '');
    if (s) return s;
  }
  return f.split('/').filter(Boolean).slice(-2).join('/') || f;  // fallback: last 2 segs
};
const recShort = stem => { const p = stem.split('__'); return p.length > 1 ? p[p.length - 1] : stem; };
// Source npz path trimmed to start at the data-root segment (e.g.
// "processed_data/<batch>/<file>.npz"): strip everything ABOVE the shared
// folder prefix's last segment, so the boring mount path doesn't show.
function pathFromRoot(p) {
  const pre = STATE.folderPrefix;            // e.g. /data/cl_data/ai-infra/processed_data
  if (!pre) return p;
  const parent = pre.slice(0, pre.lastIndexOf('/'));   // /data/cl_data/ai-infra
  return (parent && p.startsWith(parent + '/')) ? p.slice(parent.length + 1) : p;
}

// Fallback overview data-axis edges (image fraction) when the server couldn't
// detect them from the PNG; the per-recording ov_x0/ov_x1 are exact and preferred.
const OVX0 = 0.054, OVX1 = 0.989;
const ovX0 = () => (STATE.rec && STATE.rec.ov_x0 != null ? STATE.rec.ov_x0 : OVX0);
const ovX1 = () => (STATE.rec && STATE.rec.ov_x1 != null ? STATE.rec.ov_x1 : OVX1);
const recDur = () => (STATE.rec ? STATE.rec.duration_s : 0);

// Bind helper: never let a missing/renamed element abort the whole boot (which
// would leave a blank page). Logs which id was missing instead.
function on(id, ev, fn) {
  const el = document.getElementById(id);
  if (el) el[ev] = fn;
  else console.warn('missing element #' + id);
}

// Plotly loads deferred (4.5MB, non-blocking) so the tree paints immediately;
// the 3D hand mesh awaits it here rather than blocking the whole page.
function whenPlotly(timeoutMs = 20000) {
  if (window.Plotly) return Promise.resolve();
  // Resolve once Plotly is ready; reject after a timeout so a missing/blocked
  // plotly.min.js (no local vendor AND no CDN reachable) surfaces an error in
  // the 3D panel instead of spinning forever.
  return new Promise((res, rej) => {
    const t0 = Date.now();
    const t = setInterval(() => {
      if (window.Plotly) { clearInterval(t); res(); }
      else if (Date.now() - t0 > timeoutMs) {
        clearInterval(t);
        rej(new Error('Plotly failed to load (no local webui/plotly.min.js and CDN unreachable)'));
      }
    }, 50);
  });
}

async function boot() {
  try {
    const d = await (await fetch('/api/recordings')).json();
    STATE.recordings = d.recordings || [];
    STATE.folderPrefix = commonFolderPrefix(STATE.recordings.map(r => r.folder));
    STATE.labels_used = d.labels_used || [];
    STATE.common_labels = d.common_labels || [];
    STATE.total_clips = d.total_clips || 0;
    STATE.total_labeled = d.total_labeled || 0;
    STATE.total_invalid = d.total_invalid || 0;
    STATE.total_invalid_rec = d.total_invalid_rec || 0;
    refreshDatalist();
    renderCommon();
    on('common', 'onclick', onCommonClick);
    on('filter', 'onchange', applyRecFilter);
    on('sort', 'onchange', applyRecFilter);
    on('search', 'oninput', applyRecFilter);
    on('save', 'onclick', () => saveLabel(true));
    on('invalid', 'onclick', () => markInvalid());
    on('recinvalid', 'onclick', () => toggleRecordingInvalid());
    on('droprec', 'onclick', () => dropRecording());
    on('export', 'onclick', doExport);
    on('play', 'onclick', togglePlay);
    on('scrub', 'oninput', e => { stop(); setTime(e.target.value / 1000 * recDur()); });
    on('list', 'onclick', e => {
      const fh = e.target.closest('.folder');
      if (fh) {
        const f = fh.dataset.folder;
        STATE.expanded.has(f) ? STATE.expanded.delete(f) : STATE.expanded.add(f);
        return renderRecList();
      }
      const row = e.target.closest('.row');
      if (row) selectRecordingIdx(+row.dataset.i);
    });
    on('segchips', 'onclick', e => {
      const c = e.target.closest('.seg');
      if (c && STATE.rec) selectClip(STATE.rec.segments[+c.dataset.i]);
    });
    document.onkeydown = onKey;
    applyRecFilter();
    renderProgress();
  } catch (e) {
    console.error('boot failed', e);
    document.body.insertAdjacentHTML('afterbegin',
      '<div style="background:#fee;color:#900;padding:10px;font:13px monospace;' +
      'white-space:pre-wrap">boot 失败: ' + ((e && e.stack) || e) + '</div>');
  }
}

// ---- recordings list ------------------------------------------------------
function applyRecFilter() {
  const f = document.getElementById('filter').value;
  const s = document.getElementById('sort').value;
  const q = (document.getElementById('search').value || '').toLowerCase();
  let v = STATE.recordings.filter(r => {
    if (q && !(r.stem.toLowerCase().includes(q) ||
               (r.subject || '').toLowerCase().includes(q))) return false;
    const done = recDone(r);                 // labelled+invalid clips, or whole rec invalid
    if (f === 'todo') return !done;
    if (f === 'review') return r.n_review > 0 && !done;
    if (f === 'done') return done;
    return true;
  });
  const pending = r => (r.n_review > 0 && !recDone(r)) ? 1 : 0;
  if (s === 'review') v.sort((a, b) => pending(b) - pending(a));
  STATE.recView = v;
  // searching: expand all matching folders so results are visible
  if (q) v.forEach(r => STATE.expanded.add(r.folder));
  renderRecList();
  const curIdx = STATE.rec ? v.findIndex(r => r.stem === STATE.rec.stem) : -1;
  if (curIdx >= 0) { STATE.recIdx = curIdx; renderRecList(); }
  else if (v.length) selectRecordingIdx(0);
}

const reviewed = r => (r.n_labeled || 0) + (r.n_invalid || 0);  // triaged clips
// A recording is "done" when every clip is triaged OR the whole recording is
// marked invalid (excluded from labelling/clustering).
const recDone = r => !!r.rec_invalid || reviewed(r) >= r.n_clips;

// Left tree: recordings grouped by source folder; folders collapsible.
function renderRecList() {
  const folders = [], byFolder = new Map();
  STATE.recView.forEach((r, i) => {
    if (!byFolder.has(r.folder)) { byFolder.set(r.folder, []); folders.push(r.folder); }
    byFolder.get(r.folder).push(i);
  });
  const sum = (idxs, fn) => idxs.reduce((a, i) => a + fn(STATE.recView[i]), 0);
  const html = folders.map(folder => {
    const idxs = byFolder.get(folder);
    const nLab = sum(idxs, r => r.n_labeled), nClip = sum(idxs, r => r.n_clips);
    const nInv = sum(idxs, r => r.n_invalid || 0), nRev = sum(idxs, r => r.n_review);
    const nRecInv = idxs.filter(i => STATE.recView[i].rec_invalid).length;
    const nCached = idxs.filter(i => STATE.recView[i].cached >= 2).length;
    const exp = STATE.expanded.has(folder);
    const allDone = idxs.every(i => recDone(STATE.recView[i]));
    const fcls = allDone ? 'fdone' : (nRev > 0 ? 'freview' : '');
    let h = `<div class="folder ${fcls}" data-folder="${esc(folder)}">` +
      `${exp ? '▾' : '▸'} ${esc(folderShort(folder))} ` +
      `<small>${idxs.length}录制 · ${nLab}/${nClip}${nInv ? ` ✗${nInv}` : ''}${nRecInv ? ` ⊘${nRecInv}` : ''}${nRev ? ` · ⚠${nRev}` : ''}` +
      `${nCached ? ` · <span class=cdone>🎬${nCached}</span>` : ''}</small></div>`;
    if (exp) h += idxs.map(i => {
      const r = STATE.recView[i];
      const cls = r.rec_invalid ? 'recinvalid'
        : (reviewed(r) >= r.n_clips ? 'done' : (r.n_review > 0 ? 'review' : 'todo'));
      const sel = STATE.rec && STATE.rec.stem === r.stem ? ' sel' : '';
      const dot = `<span class="cdot c${r.cached || 0}" title="3D缓存: ${['无','部分','完成'][r.cached || 0]}"></span>`;
      return `<div class="row ${cls}${sel}" data-i="${i}" title="${esc(r.stem)}">` +
        `${dot}${esc(recShort(r.stem))} <small>${r.rec_invalid ? '⊘整条无效'
          : `${r.n_labeled}/${r.n_clips}${r.n_invalid ? ` ✗${r.n_invalid}` : ''}${r.n_review ? ` ⚠${r.n_review}` : ''}`}</small></div>`;
    }).join('');
    return h;
  }).join('');
  document.getElementById('list').innerHTML = html;
}

async function selectRecordingIdx(i) {
  if (i < 0 || i >= STATE.recView.length) return;
  STATE.recIdx = i;
  const r = STATE.recView[i];
  STATE.expanded.add(r.folder);            // keep the selected recording visible
  if (STATE.rec && STATE.rec.stem === r.stem) { renderRecList(); return; }
  await loadRecording(r.stem);
  if (!STATE.rec) return;
  renderRecList();
  const segs = STATE.rec.segments;
  const seg = segs.find(s => !s.gesture_label && !s.invalid) || segs[0];
  if (seg) selectClip(seg);
}

// labels_used drives the <datalist> autocomplete on the gesture_label input.
function refreshDatalist() {
  document.getElementById('labellist').innerHTML =
    STATE.labels_used.map(l => `<option value="${esc(l)}">`).join('');
}

// common_labels = the curated quick-set palette: clickable chips (click =
// apply+save+next, same as the 1-9 number keys for the first nine). Each chip
// has an × to drop it; "★加入常用" adds the current gesture_label input.
function renderCommon() {
  const chips = STATE.common_labels.map((l, i) => {
    const num = i < 9 ? `<b>${i + 1}</b> ` : '';   // first nine double as hotkeys
    return `<span class=commonchip data-label="${esc(l)}" title="点击套用并保存">`
      + `${num}${esc(l)}<span class=cx data-rm="${esc(l)}" title="移出常用">×</span></span>`;
  }).join('');
  const n = STATE.common_labels.length, full = n >= 20;
  document.getElementById('common').innerHTML =
    `<span class=clab>常用 (${n}/20):</span> `
    + (chips || '<span class=qc>（空，输入标签后点"加入常用"）</span> ')
    + `<button type=button class="btn addcommon" id=addcommon ${full ? 'disabled' : ''}`
    + ` title="把当前 gesture_label 输入框的标签加入常用">★ 加入常用</button>`;
}

function applyCommonLabel(l) {
  if (!l) return;
  setLabelInput(l);
  saveLabel(true);          // apply + save + advance, like the number keys
}

async function saveCommon(labels) {
  const res = await (await fetch('/api/common_labels', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels })
  })).json();
  STATE.common_labels = res.common_labels || [];
  renderCommon();
}

function addCommon(l) {
  l = (l || '').trim();
  const msg = document.getElementById('msg');
  if (!l) { msg.textContent = '先在 gesture_label 输入框填入要加入的标签'; return; }
  if (STATE.common_labels.includes(l)) { msg.textContent = `"${l}" 已在常用`; return; }
  if (STATE.common_labels.length >= 20) { msg.textContent = '常用已满 20 个，先移出一些'; return; }
  saveCommon([...STATE.common_labels, l]);
}

function removeCommon(l) { saveCommon(STATE.common_labels.filter(x => x !== l)); }

// One delegated handler survives the innerHTML re-render in renderCommon().
function onCommonClick(e) {
  const rm = e.target.closest('.cx');
  if (rm) { e.stopPropagation(); return removeCommon(rm.dataset.rm); }
  if (e.target.closest('#addcommon')) {
    return addCommon(document.getElementById('label').value);
  }
  const chip = e.target.closest('.commonchip');
  if (chip) return applyCommonLabel(chip.dataset.label);
}

function renderProgress() {
  document.getElementById('progress').textContent =
    `已标 ${STATE.total_labeled} · 无效段 ${STATE.total_invalid}` +
    `${STATE.total_invalid_rec ? ` · ⊘录制 ${STATE.total_invalid_rec}` : ''} / ${STATE.total_clips}`;
}

// ---- recording: overview img + hand frames --------------------------------
async function loadRecording(stem) {
  stop();
  const d = await (await fetch('/api/recording?stem=' + encodeURIComponent(stem))).json();
  if (d.error) { document.getElementById('msg').textContent = 'recording: ' + d.error; STATE.rec = null; return; }
  STATE.rec = d;
  buildOverview(d);
  buildHand(d);
  renderChips();
  renderOvInfo();
  updateRecInvalidBtn();
}

// Recording-level line above the overview. Line 1 = original npz path (from the
// data root). Line 2 = this recording's segmentation QC / thresholds (lag, EMG &
// pose-speed thresholds, seg_version) -- see docs/字段说明.md. All fields come
// from the recordings-list entry (recordings table), so they're static per
// recording; hover each chip for the raw column name + meaning.
function renderOvInfo() {
  const el = document.getElementById('ovinfo');
  if (!el) return;
  if (!STATE.rec) { el.innerHTML = ''; return; }
  const d = STATE.rec;
  const rec = STATE.recordings.find(r => r.stem === d.stem) || {};
  const path = rec.source_path ? pathFromRoot(rec.source_path) : d.stem;
  const num = (v, n) => (v == null ? '—' : (+v).toFixed(n));
  const lag = rec.emg_pose_lag_s == null ? 'n/a' : num(rec.emg_pose_lag_s, 2) + 's';
  const qc = [
    `<span title="emg_pose_lag_s（lag_flag）：EMG包络与关节速度的互相关延迟（秒），正=EMG领先动作；本类准周期数据上 lag_flag 偏乐观，仅作诊断">延迟 ${lag}${rec.lag_flag ? `（${esc(rec.lag_flag)}）` : ''}</span>`,
    `<span title="emg_pose_corr：上面延迟处的相关系数（越高越可信）">相关 ${num(rec.emg_pose_corr, 2)}</span>`,
    `<span title="enter_thresh / exit_thresh：EMG包络的开 / 关阈值">EMG阈值 进/出 ${num(rec.enter_thresh, 1)}/${num(rec.exit_thresh, 1)}</span>`,
    `<span title="pose_thresh / pose_exit_thresh：姿态速度的进入 / 退出阈值（rad/s）">姿态阈值 进/出 ${num(rec.pose_thresh, 2)}/${num(rec.pose_exit_thresh, 2)}</span>`,
  ];
  if (rec.seg_version) qc.push(`<span title="seg_version：切分参数版本（8位哈希）">切分版本 ${esc(rec.seg_version)}</span>`);
  el.innerHTML =
    `<div class=ovpath title="${esc(rec.source_path || '')}">${esc(path)}</div>` +
    `<div class=ovbits>${qc.join(' · ')}</div>`;
}

// Version the URL by duration+frame count so a re-segmented/regenerated overview
// (e.g. a corrected duration) never gets served stale from the browser cache.
const OV = (stem, ver) => '/api/overview?stem=' + encodeURIComponent(stem) +
  (ver != null ? '&v=' + encodeURIComponent(ver) : '');
function buildOverview(d) {
  const img = document.getElementById('ovimg');
  img.onerror = () => { img.removeAttribute('src'); document.getElementById('msg').textContent = 'overview.png 缺失（--no-overview?）'; };
  img.src = OV(d.stem, `${d.duration_s}_${d.n_frames}`);
  document.getElementById('ovwrap').onclick = e => {
    const r = img.getBoundingClientRect();
    if (!r.width) return;
    const f = (e.clientX - r.left) / r.width;
    stop(); setTime((f - ovX0()) / (ovX1() - ovX0()) * recDur());
  };
}

function placePlayhead(t) {
  document.getElementById('ovhead').style.left =
    (ovX0() + (t / recDur()) * (ovX1() - ovX0())) * 100 + '%';
}

// Hand = emg2pose hand MESH as a rotatable Plotly Mesh3d. The server sends just
// the 788 vertices (int mm) per frame; faces (i,j,k) are fetched once. The
// scrubber drives which frame's vertices get restyled in; rotation is kept
// across frames via scene.uirevision.
function handUrl(stem, fi) { return `/api/handmesh?stem=${encodeURIComponent(stem)}&i=${fi}`; }

async function fetchVerts(stem, fi) {
  const m = await (await fetch(handUrl(stem, fi))).json();
  return m.v;                            // [[x,y,z], ...788]
}

function cube(xs, ys, zs) {
  const lo = [Math.min(...xs), Math.min(...ys), Math.min(...zs)];
  const hi = [Math.max(...xs), Math.max(...ys), Math.max(...zs)];
  const mid = lo.map((v, k) => (v + hi[k]) / 2);
  let half = Math.max(...lo.map((v, k) => hi[k] - v)) / 2 * 1.4;
  if (!(half > 0)) half = 100;
  return [mid.map(v => v - half), mid.map(v => v + half)];
}

async function buildHand(d) {
  STATE.curHandFrame = -1;
  STATE.handEpoch += 1;
  STATE.wantFrame = 0;
  STATE.meshReady = false;
  STATE.handVerts = new Array(d.n_frames || 0);  // fresh in-memory frame buffer
  STATE.handN = d.n_frames || 0;
  document.getElementById('handprog').classList.remove('on', 'done');
  if (!STATE.faces) STATE.faces = await (await fetch('/api/handfaces')).json();
  const epoch = STATE.handEpoch;
  const v = await fetchVerts(d.stem, 0);
  if (epoch !== STATE.handEpoch) return;         // recording switched mid-fetch
  STATE.handVerts[0] = v;
  const xs = v.map(p => p[0]), ys = v.map(p => p[1]), zs = v.map(p => p[2]);
  const [lo, hi] = cube(xs, ys, zs);
  const f = STATE.faces;
  const trace = {
    type: 'mesh3d', x: xs, y: ys, z: zs, i: f.i, j: f.j, k: f.k,
    color: 'lightpink', opacity: 1, flatshading: false,
    lighting: { ambient: 0.55, diffuse: 0.7, specular: 0.25, roughness: 0.6, fresnel: 0.1 },
    lightposition: { x: 120, y: 220, z: 320 },
  };
  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
    scene: {
      aspectmode: 'cube', uirevision: 'hand',
      xaxis: { visible: false, range: [lo[0], hi[0]] },
      yaxis: { visible: false, range: [lo[1], hi[1]] },
      zaxis: { visible: false, range: [lo[2], hi[2]] },
      camera: { eye: { x: 1.5, y: -1.5, z: 0.9 } },
    },
  };
  try {
    await whenPlotly();                            // deferred 4.5MB lib may still be loading
  } catch (e) {
    const el = document.getElementById('hand');
    if (el) el.innerHTML = "<div style='padding:12px;color:#a00;font-size:12px'>" +
      "3D 手部渲染不可用：未找到 webui/plotly.min.js 且 CDN 不可达。" +
      "请下载 plotly-2.35.2.min.js 放到 webui/ 或联网后重试。</div>";
    return;
  }
  if (epoch !== STATE.handEpoch) return;
  Plotly.newPlot('hand', [trace], layout, { displaylogo: false, responsive: true });
  STATE.meshReady = true; STATE.curHandFrame = 0;
  preloadHands(d.stem, d.n_frames, STATE.handEpoch);
  pollHandStatus(d.stem);
}

// reflect 3D-cache state on the recording in the left tree (0/1/2)
function markCached(stem, level) {
  const r = STATE.recordings.find(x => x.stem === stem);
  if (r && (r.cached || 0) < level) { r.cached = level; renderRecList(); }
}

// Pull every frame's vertices into STATE.handVerts so playback restyles from
// memory (no per-frame fetch -> no network jitter / dropped frames).
function preloadHands(stem, n, epoch) {
  let next = 0;
  const PAR = 6;
  const one = () => {
    if (epoch !== STATE.handEpoch || next >= n) return;
    const fi = next++;
    if (STATE.handVerts[fi]) return one();        // already have it (e.g. frame 0)
    fetch(handUrl(stem, fi))
      .then(r => r.json())
      .then(m => { if (epoch === STATE.handEpoch) STATE.handVerts[fi] = m.v; })
      .catch(() => {})
      .finally(one);
  };
  for (let k = 0; k < PAR; k++) one();
}

async function updateHand(fi) {
  if (!STATE.rec || !STATE.meshReady || fi === STATE.curHandFrame) return;
  STATE.wantFrame = fi;
  const stem = STATE.rec.stem;
  let v = STATE.handVerts[fi];
  if (!v) {                              // not preloaded yet -> fetch on demand
    v = await fetchVerts(stem, fi);
    if (!STATE.rec || stem !== STATE.rec.stem || fi !== STATE.wantFrame) return;  // stale
    STATE.handVerts[fi] = v;
  }
  STATE.curHandFrame = fi;
  Plotly.restyle('hand', { x: [v.map(p => p[0])], y: [v.map(p => p[1])], z: [v.map(p => p[2])] });
}

function pollHandStatus(stem) {
  if (STATE.handPoll) clearInterval(STATE.handPoll);
  const prog = document.getElementById('handprog');
  const fill = document.getElementById('handprogfill');
  const lab = document.getElementById('handproglabel');
  const tick = async () => {
    if (!STATE.rec || STATE.rec.stem !== stem) {
      clearInterval(STATE.handPoll); STATE.handPoll = null; prog.classList.remove('on'); return;
    }
    const total = STATE.rec.n_frames || 0;
    const s = await (await fetch('/api/handstatus?stem=' + encodeURIComponent(stem))).json();
    if (total && s.done < total) {
      prog.classList.add('on'); prog.classList.remove('done');
      fill.style.width = (100 * s.done / total) + '%';
      lab.textContent = `渲染手部 ${s.done}/${total}`;
      markCached(stem, 1);
    } else {                                   // fully rendered
      prog.classList.add('on', 'done'); fill.style.width = '100%';
      lab.textContent = '渲染完成 ✓';
      markCached(stem, 2);
      clearInterval(STATE.handPoll); STATE.handPoll = null;
      setTimeout(() => { if (STATE.rec && STATE.rec.stem === stem) prog.classList.remove('on'); }, 1500);
    }
  };
  STATE.handPoll = setInterval(tick, 800);
  tick();                                      // immediate first check
}

function frameIndexForTime(t) {
  const ft = STATE.rec.frame_times, F = ft.length;
  if (F <= 1) return 0;
  if (t <= ft[0]) return 0;
  if (t >= ft[F - 1]) return F - 1;
  let lo = 0, hi = F - 1;
  while (lo + 1 < hi) { const mid = (lo + hi) >> 1; if (ft[mid] <= t) lo = mid; else hi = mid; }
  return (t - ft[lo] <= ft[hi] - t) ? lo : hi;
}

// ---- transport ------------------------------------------------------------
function setTime(t) {
  if (!STATE.rec) return;
  t = Math.max(0, Math.min(recDur(), t));
  STATE.t = t;
  document.getElementById('scrub').value = Math.round(t / recDur() * 1000) || 0;
  document.getElementById('tread').textContent = `${t.toFixed(2)} / ${recDur().toFixed(2)}s`;
  placePlayhead(t);
  updateHand(frameIndexForTime(t));
  highlightCurSeg(t);
}

function highlightCurSeg(t) {
  document.querySelectorAll('#segchips .seg').forEach(el => {
    const s = STATE.rec.segments[+el.dataset.i];
    el.classList.toggle('cur', STATE.cur ? s.key === STATE.cur.key
      : (t >= s.start_s && t <= s.end_s));
  });
}

function togglePlay() { STATE.playing ? stop() : play(); }
function play() {
  if (!STATE.rec) return;
  STATE.playing = true; document.getElementById('play').textContent = '⏸';
  if (STATE.t >= recDur() - 1e-3) STATE.t = 0;
  const speed = +document.getElementById('pbspeed').value;
  let last = performance.now(), lastDraw = 0;
  const tick = () => {
    if (!STATE.playing) return;
    const now = performance.now();
    STATE.t += (now - last) / 1000 * speed; last = now;
    if (STATE.t >= recDur()) { setTime(recDur()); stop(); return; }
    if (now - lastDraw >= 40) { setTime(STATE.t); lastDraw = now; }  // ~25fps
    STATE.raf = requestAnimationFrame(tick);
  };
  STATE.raf = requestAnimationFrame(tick);
}
function stop() {
  STATE.playing = false; document.getElementById('play').textContent = '▶';
  if (STATE.raf) cancelAnimationFrame(STATE.raf);
  STATE.raf = null;
}

// ---- clips (chips) + labelling --------------------------------------------
function renderChips() {
  if (!STATE.rec) return;
  document.getElementById('segchips').innerHTML = STATE.rec.segments.map((s, i) => {
    const state = s.invalid ? 'invalid' : (s.gesture_label ? 'done' : 'todo');
    const cls = `seg ${state}${s.review ? ' review' : ''}` +
      `${STATE.cur && STATE.cur.key === s.key ? ' cur' : ''}`;
    const txt = s.invalid ? `s${s.s} ✗` : `s${s.s}${s.gesture_label ? ' ' + s.gesture_label : ''}`;
    return `<div class="${cls}" data-i="${i}" title="${s.key}${s.review ? ' · burst=-1 复核' : ''}">${txt}</div>`;
  }).join('');
}

function selectClip(seg) {
  STATE.cur = seg;
  document.getElementById('meta').innerHTML =
    `<b>s${seg.s} · c${String(seg.clip_id).padStart(4, '0')}</b>` +
    `${seg.invalid ? ' <span class=warn>[已标记无效]</span>' : ''}<br>` +
    `dur ${seg.duration_s.toFixed(2)}s · motion ${seg.motion_duration_s.toFixed(2)}s<br>` +
    `env_peak ${seg.envelope_peak.toFixed(1)} · pose_range ${seg.pose_range.toFixed(2)}<br>` +
    (seg.review ? `<span class=warn>matched_burst = -1（优先复核）</span>` : `burst ${seg.matched_emg_seg_idx}`);
  setLabelInput(seg.gesture_label || '');
  document.getElementById('note').value = seg.note || '';
  renderChips();
  setTime(seg.start_s);     // jump to the clip's start (its left edge in overview)
  document.getElementById('label').focus();
}

function setLabelInput(v) { document.getElementById('label').value = v; }

// shared writer for both "save gesture label" and "mark invalid"
async function postLabel(gesture_label, invalid, advance) {
  const seg = STATE.cur; if (!seg) return;
  const note = document.getElementById('note').value.trim();
  const labeler = document.getElementById('labeler').value.trim();
  const res = await (await fetch('/api/label', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key: seg.key, gesture_label, invalid, labeler, note })
  })).json();
  seg.gesture_label = invalid ? '' : gesture_label; seg.invalid = invalid; seg.note = note;
  if (res.total_labeled != null) STATE.total_labeled = res.total_labeled;
  if (res.total_invalid != null) STATE.total_invalid = res.total_invalid;
  const rec = STATE.recordings.find(r => r.stem === res.stem);
  if (rec) {
    if (res.stem_labeled != null) rec.n_labeled = res.stem_labeled;
    if (res.stem_invalid != null) rec.n_invalid = res.stem_invalid;
  }
  if (!invalid && gesture_label && !STATE.labels_used.includes(gesture_label)) {
    STATE.labels_used.push(gesture_label); STATE.labels_used.sort(); refreshDatalist();
  }
  renderChips(); renderRecList(); renderProgress();
  document.getElementById('msg').textContent =
    (invalid ? '已标记无效 ' : '已保存 ') + new Date().toLocaleTimeString();
  if (advance) nextTodo();
}

function saveLabel(advance) { postLabel(document.getElementById('label').value.trim(), false, advance); }
function markInvalid() { postLabel('', true, true); }

// --- whole-recording invalid (excluded from labelling/clustering) ----------
function recInvalidState() {
  const rec = STATE.rec && STATE.recordings.find(r => r.stem === STATE.rec.stem);
  return !!(rec && rec.rec_invalid);
}
function updateRecInvalidBtn() {
  const btn = document.getElementById('recinvalid');
  if (!btn) return;
  const inv = recInvalidState();
  btn.textContent = inv ? '取消整条无效 (X)' : '整条录制无效 (X)';
  btn.style.background = inv ? '#444' : '';
  btn.style.color = inv ? '#fff' : '';
  btn.disabled = !STATE.rec;
}
async function markRecordingInvalid(toInvalid) {
  if (!STATE.rec) return;
  const stem = STATE.rec.stem;
  const labeler = document.getElementById('labeler').value.trim();
  const res = await (await fetch('/api/invalid_recording', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stem, invalid: toInvalid, labeler })
  })).json();
  const rec = STATE.recordings.find(r => r.stem === stem);
  if (rec) rec.rec_invalid = res.rec_invalid;
  if (res.total_invalid_rec != null) STATE.total_invalid_rec = res.total_invalid_rec;
  updateRecInvalidBtn(); renderRecList(); renderProgress();
  document.getElementById('msg').textContent =
    (res.rec_invalid ? '整条录制已标记无效，跳过 ' : '已取消整条无效 ') + new Date().toLocaleTimeString();
  if (res.rec_invalid) nextRecording();   // marked bad -> move on
}
function toggleRecordingInvalid() { markRecordingInvalid(!recInvalidState()); }

async function dropRecording() {
  if (!STATE.rec) return;
  const stem = STATE.rec.stem;
  if (!confirm(`永久删除录制 ${stem}？\n会删除其 shard + 索引行并 tombstone（重切分不会复活）。不可撤销。`))
    return;
  const res = await (await fetch('/api/drop_recording', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stem })
  })).json();
  if (!res.ok) { document.getElementById('msg').textContent = '删除失败: ' + (res.error || ''); return; }
  STATE.recordings = STATE.recordings.filter(r => r.stem !== stem);
  document.getElementById('msg').textContent = `已永久删除 ${stem} `+ new Date().toLocaleTimeString();
  renderRecList(); renderProgress(); nextRecording();
}

function nextRecording() {
  for (let k = STATE.recIdx + 1; k < STATE.recView.length; k++)
    if (!recDone(STATE.recView[k])) return selectRecordingIdx(k);
  document.getElementById('msg').textContent = '可见录制都看完了 🎉';
}

function nextTodo() {
  const segs = STATE.rec ? STATE.rec.segments : [];
  const ci = STATE.cur ? segs.findIndex(s => s.key === STATE.cur.key) : -1;
  for (let j = ci + 1; j < segs.length; j++)        // next un-reviewed clip
    if (!segs[j].gesture_label && !segs[j].invalid) return selectClip(segs[j]);
  nextRecording();                                  // recording done -> next
}

function navClip(d) {
  if (!STATE.rec) return;
  const segs = STATE.rec.segments;
  const ci = STATE.cur ? segs.findIndex(s => s.key === STATE.cur.key) : -1;
  const j = ci + d;
  if (j >= 0 && j < segs.length) selectClip(segs[j]);
}

function onKey(e) {
  if (e.key === 'Enter') { e.preventDefault(); saveLabel(true); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); navClip(1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); navClip(-1); }
  else if (e.key === ' ' && e.target.tagName !== 'INPUT') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'X' && document.activeElement.id !== 'label' && document.activeElement.id !== 'note') {
    e.preventDefault(); toggleRecordingInvalid();      // Shift+X: whole recording
  }
  else if (e.key === 'x' && document.activeElement.id !== 'label' && document.activeElement.id !== 'note') {
    e.preventDefault(); markInvalid();                 // x: this clip
  }
  else if (/^[1-9]$/.test(e.key) && e.target.id !== 'note' && document.activeElement.id !== 'label') {
    applyCommonLabel(STATE.common_labels[+e.key - 1]);   // 1-9 = first nine common labels
  }
}

async function doExport() {
  document.getElementById('msg').textContent = '导出中…';
  const r = await (await fetch('/api/export')).json();
  document.getElementById('msg').textContent =
    `导出 ${r.n_labeled} 条标注 → ${r.n_folders} 个分文件夹 CSV (${r.dir}/) + 合并 ${r.merged}`;
}

boot();
