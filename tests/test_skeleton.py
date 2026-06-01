import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from emg_label.skeleton import (axis_limits, draw_skeleton,
                                normalize_skeleton)


def _synthetic_skeleton(n_frames=1):
    """A flat fan of 5 finger chains anchored at the wrist (joint 0 of each)."""
    rng = np.random.default_rng(42)
    skel = np.zeros((n_frames, 25, 3), dtype=np.float32)
    for chain in range(5):
        # Spread each chain along +y, with the first joint at the palm.
        for j in range(5):
            idx = chain * 5 + j
            skel[:, idx, 0] = chain * 0.02     # finger-spread along x
            skel[:, idx, 1] = j * 0.03         # finger length along y
            skel[:, idx, 2] = 0.0
    skel += rng.normal(0, 0.001, skel.shape)
    return skel


def test_normalize_centres_on_roots():
    skel = _synthetic_skeleton(3) + np.array([[10.0, -5.0, 7.0]])
    normed = normalize_skeleton(skel)
    # After normalising, the per-frame mean of the 5 root joints is ~origin.
    roots = normed[:, [0, 5, 10, 15, 20], :]
    assert np.allclose(roots.mean(axis=1), 0.0, atol=1e-5)


def test_axis_limits_cubic_and_bounded():
    skel = _synthetic_skeleton(2)
    lims = axis_limits(skel)
    assert len(lims) == 3
    spans = [hi - lo for lo, hi in lims]
    # Cubic axes -> all three spans equal.
    assert np.allclose(spans, spans[0], atol=1e-5)
    # Skeleton bounds must lie inside the cube.
    pts = skel.reshape(-1, 3)
    for ax_i in range(3):
        lo, hi = lims[ax_i]
        assert lo <= pts[:, ax_i].min() and pts[:, ax_i].max() <= hi


def test_draw_skeleton_smoke(tmp_path):
    # No torch needed: smoke-test that the renderer can draw without raising.
    skel = _synthetic_skeleton(1)[0]
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    draw_skeleton(ax, skel)
    out = tmp_path / "smoke.png"
    fig.savefig(out, dpi=60)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000
