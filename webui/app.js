// Two-level navigation: left panel lists RECORDINGS; selecting one lazily
// loads its clips (the bottom s1..sN chips). Labelling acts on the current clip.
let STATE = {
  recordings: [], recView: [], recIdx: 0,
  expanded: new Set(),  // expanded folder keys in the left tree
  rec: null,            // loaded recording payload (segments=clips + frame grid)
  cur: null,            // currently selected clip (segment) = labelling target
  labels_used: [], total_clips: 0, total_labeled: 0,
  t: 0, playing: false, raf: null,
  curHandFrame: -1, handPoll: null, handEpoch: 0,
  faces: null, wantFrame: 0, meshReady: false,   // 3D hand mesh state
};

const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const folderShort = f => f.split('/').filter(Boolean).slice(-2).join('/') || f;
const recShort = stem => { const p = stem.split('__'); return p.length > 1 ? p[p.length - 1] : stem; };

// Fallback overview data-axis edges (image fraction) when the server couldn't
// detect them from the PNG; the per-recording ov_x0/ov_x1 are exact and preferred.
const OVX0 = 0.054, OVX1 = 0.989;
const ovX0 = () => (STATE.rec && STATE.rec.ov_x0 != null ? STATE.rec.ov_x0 : OVX0);
const ovX1 = () => (STATE.rec && STATE.rec.ov_x1 != null ? STATE.rec.ov_x1 : OVX1);
const recDur = () => (STATE.rec ? STATE.rec.duration_s : 0);

async function boot() {
  const d = await (await fetch('/api/recordings')).json();
  STATE.recordings = d.recordings;
  STATE.labels_used = d.labels_used;
  STATE.total_clips = d.total_clips;
  STATE.total_labeled = d.total_labeled;
  refreshDatalist();
  document.getElementById('filter').onchange = applyRecFilter;
  document.getElementById('sort').onchange = applyRecFilter;
  document.getElementById('search').oninput = applyRecFilter;
  document.getElementById('save').onclick = () => saveLabel(true);
  document.getElementById('skip').onclick = () => { setLabelInput(''); saveLabel(true); };
  document.getElementById('export').onclick = doExport;
  document.getElementById('play').onclick = togglePlay;
  document.getElementById('scrub').oninput = e => { stop(); setTime(e.target.value / 1000 * recDur()); };
  document.getElementById('list').onclick = e => {
    const fh = e.target.closest('.folder');
    if (fh) {
      const f = fh.dataset.folder;
      STATE.expanded.has(f) ? STATE.expanded.delete(f) : STATE.expanded.add(f);
      return renderRecList();
    }
    const row = e.target.closest('.row');
    if (row) selectRecordingIdx(+row.dataset.i);
  };
  document.getElementById('segchips').onclick = e => {
    const c = e.target.closest('.seg');
    if (c && STATE.rec) selectClip(STATE.rec.segments[+c.dataset.i]);
  };
  document.onkeydown = onKey;
  applyRecFilter();
  renderProgress();
}

// ---- recordings list ------------------------------------------------------
function applyRecFilter() {
  const f = document.getElementById('filter').value;
  const s = document.getElementById('sort').value;
  const q = (document.getElementById('search').value || '').toLowerCase();
  let v = STATE.recordings.filter(r => {
    if (q && !(r.stem.toLowerCase().includes(q) ||
               (r.subject || '').toLowerCase().includes(q))) return false;
    const done = r.n_labeled >= r.n_clips;
    if (f === 'todo') return !done;
    if (f === 'review') return r.n_review > 0 && !done;
    if (f === 'done') return done;
    return true;
  });
  const pending = r => (r.n_review > 0 && r.n_labeled < r.n_clips) ? 1 : 0;
  if (s === 'review') v.sort((a, b) => pending(b) - pending(a));
  STATE.recView = v;
  // searching: expand all matching folders so results are visible
  if (q) v.forEach(r => STATE.expanded.add(r.folder));
  renderRecList();
  const curIdx = STATE.rec ? v.findIndex(r => r.stem === STATE.rec.stem) : -1;
  if (curIdx >= 0) { STATE.recIdx = curIdx; renderRecList(); }
  else if (v.length) selectRecordingIdx(0);
}

// Left tree: recordings grouped by source folder; folders collapsible.
function renderRecList() {
  const folders = [], byFolder = new Map();
  STATE.recView.forEach((r, i) => {
    if (!byFolder.has(r.folder)) { byFolder.set(r.folder, []); folders.push(r.folder); }
    byFolder.get(r.folder).push(i);
  });
  const sum = (idxs, k) => idxs.reduce((a, i) => a + STATE.recView[i][k], 0);
  const html = folders.map(folder => {
    const idxs = byFolder.get(folder);
    const nLab = sum(idxs, 'n_labeled'), nClip = sum(idxs, 'n_clips'), nRev = sum(idxs, 'n_review');
    const nCached = idxs.filter(i => STATE.recView[i].cached >= 2).length;
    const exp = STATE.expanded.has(folder);
    const fcls = nLab >= nClip ? 'fdone' : (nRev > 0 ? 'freview' : '');
    let h = `<div class="folder ${fcls}" data-folder="${esc(folder)}">` +
      `${exp ? '▾' : '▸'} ${esc(folderShort(folder))} ` +
      `<small>${idxs.length}录制 · ${nLab}/${nClip}${nRev ? ` · ⚠${nRev}` : ''}` +
      `${nCached ? ` · <span class=cdone>🎬${nCached}</span>` : ''}</small></div>`;
    if (exp) h += idxs.map(i => {
      const r = STATE.recView[i];
      const cls = r.n_labeled >= r.n_clips ? 'done' : (r.n_review > 0 ? 'review' : 'todo');
      const sel = STATE.rec && STATE.rec.stem === r.stem ? ' sel' : '';
      const dot = `<span class="cdot c${r.cached || 0}" title="3D缓存: ${['无','部分','完成'][r.cached || 0]}"></span>`;
      return `<div class="row ${cls}${sel}" data-i="${i}" title="${esc(r.stem)}">` +
        `${dot}${esc(recShort(r.stem))} <small>${r.n_labeled}/${r.n_clips}` +
        `${r.n_review ? ` ⚠${r.n_review}` : ''}</small></div>`;
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
  const seg = segs.find(s => !s.gesture_label) || segs[0];
  if (seg) selectClip(seg);
}

function refreshDatalist() {
  document.getElementById('labellist').innerHTML =
    STATE.labels_used.map(l => `<option value="${l}">`).join('');
  document.getElementById('common').innerHTML =
    '常用: ' + STATE.labels_used.slice(0, 9)
      .map((l, i) => `<span class=chip>${i + 1}:${l}</span>`).join('');
}

function renderProgress() {
  document.getElementById('progress').textContent =
    `${STATE.total_labeled}/${STATE.total_clips}`;
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
}

const OV = (stem) => '/api/overview?stem=' + encodeURIComponent(stem);
function buildOverview(d) {
  const img = document.getElementById('ovimg');
  img.onerror = () => { img.removeAttribute('src'); document.getElementById('msg').textContent = 'overview.png 缺失（--no-overview?）'; };
  img.src = OV(d.stem);
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
  document.getElementById('handprog').classList.remove('on', 'done');
  if (!STATE.faces) STATE.faces = await (await fetch('/api/handfaces')).json();
  const epoch = STATE.handEpoch;
  const v = await fetchVerts(d.stem, 0);
  if (epoch !== STATE.handEpoch) return;         // recording switched mid-fetch
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

function preloadHands(stem, n, epoch) {
  let next = 0;
  const PAR = 3;                         // warm browser cache for smooth playback
  const load = () => {
    if (epoch !== STATE.handEpoch || next >= n) return;
    fetch(handUrl(stem, next++)).then(r => r.text()).then(load, load);
  };
  for (let k = 0; k < PAR; k++) load();
}

async function updateHand(fi) {
  if (!STATE.rec || !STATE.meshReady || fi === STATE.curHandFrame) return;
  STATE.wantFrame = fi;
  const stem = STATE.rec.stem;
  const v = await fetchVerts(stem, fi);
  if (!STATE.rec || stem !== STATE.rec.stem || fi !== STATE.wantFrame) return;  // stale
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
    const done = !!s.gesture_label;
    const cls = `seg ${done ? 'done' : 'todo'}${s.review ? ' review' : ''}` +
      `${STATE.cur && STATE.cur.key === s.key ? ' cur' : ''}`;
    return `<div class="${cls}" data-i="${i}" title="${s.key}${s.review ? ' · burst=-1 复核' : ''}">` +
      `s${s.s}${s.gesture_label ? ' ' + s.gesture_label : ''}</div>`;
  }).join('');
}

function selectClip(seg) {
  STATE.cur = seg;
  document.getElementById('meta').innerHTML =
    `<b>s${seg.s} · c${String(seg.clip_id).padStart(4, '0')}</b><br>` +
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

async function saveLabel(advance) {
  const seg = STATE.cur; if (!seg) return;
  const gesture_label = document.getElementById('label').value.trim();
  const note = document.getElementById('note').value.trim();
  const labeler = document.getElementById('labeler').value.trim();
  const res = await (await fetch('/api/label', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key: seg.key, gesture_label, labeler, note })
  })).json();
  seg.gesture_label = gesture_label; seg.note = note;
  if (res.total_labeled != null) STATE.total_labeled = res.total_labeled;
  const rec = STATE.recordings.find(r => r.stem === res.stem);
  if (rec && res.stem_labeled != null) rec.n_labeled = res.stem_labeled;
  if (gesture_label && !STATE.labels_used.includes(gesture_label)) {
    STATE.labels_used.push(gesture_label); STATE.labels_used.sort(); refreshDatalist();
  }
  renderChips(); renderRecList(); renderProgress();
  document.getElementById('msg').textContent = '已保存 ' + new Date().toLocaleTimeString();
  if (advance) nextTodo();
}

function nextTodo() {
  const segs = STATE.rec ? STATE.rec.segments : [];
  const ci = STATE.cur ? segs.findIndex(s => s.key === STATE.cur.key) : -1;
  for (let j = ci + 1; j < segs.length; j++)
    if (!segs[j].gesture_label) return selectClip(segs[j]);
  // recording finished -> jump to next recording (in view) that has unlabeled
  for (let k = STATE.recIdx + 1; k < STATE.recView.length; k++)
    if (STATE.recView[k].n_labeled < STATE.recView[k].n_clips)
      return selectRecordingIdx(k);
  document.getElementById('msg').textContent = '可见录制都标完了 🎉';
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
  else if (/^[1-9]$/.test(e.key) && e.target.id !== 'note' && document.activeElement.id !== 'label') {
    const l = STATE.labels_used[+e.key - 1];
    if (l) { setLabelInput(l); saveLabel(true); }
  }
}

async function doExport() {
  document.getElementById('msg').textContent = '导出中…';
  const r = await (await fetch('/api/export')).json();
  document.getElementById('msg').textContent =
    `导出 ${r.n_labeled} 条标注 → ${r.n_folders} 个分文件夹 CSV (${r.dir}/) + 合并 ${r.merged}`;
}

boot();
