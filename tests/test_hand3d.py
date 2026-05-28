import numpy as np
import pytest

from emg_label.hand3d import LANDMARK_NAMES, BONE_CONNECTIONS, PALM_CONNECTIONS

# The FK now lives in emg2pose, which needs torch. Skip cleanly if not present.
pytest.importorskip("torch")

from emg_label.hand3d import angles_to_landmarks, angles_batch_to_landmarks  # noqa: E402


def test_landmark_layout_constants():
    assert len(LANDMARK_NAMES) == 21
    assert LANDMARK_NAMES[5] == "WRIST"
    assert LANDMARK_NAMES[1] == "INDEX_TIP"
    assert [5, 8, 9, 10, 1] in BONE_CONNECTIONS
    assert [20, 5] in PALM_CONNECTIONS


def test_shape_and_finite_left():
    lm = angles_to_landmarks(np.zeros(20), side="left")
    assert lm.shape == (21, 3)
    assert np.all(np.isfinite(lm))


def test_shape_and_finite_right():
    lm = angles_to_landmarks(np.zeros(20), side="right")
    assert lm.shape == (21, 3)
    assert np.all(np.isfinite(lm))


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 20)).astype(np.float32) * 0.3
    batch = angles_batch_to_landmarks(A, side="left")
    assert batch.shape == (4, 21, 3)
    for i in range(4):
        one = angles_to_landmarks(A[i], side="left")
        assert np.allclose(batch[i], one, atol=1e-4)


def test_mcp_flexion_shortens_index_tip_to_wrist():
    """Flexing the index MCP should bring the index tip closer to the wrist.

    Channel layout for emg2pose: index DOFs are [4..7] with one of them being
    MCP flexion. We sweep each of the four candidates and require at least one
    to reduce wrist-to-tip distance vs the zero-pose -- this is an FK-agnostic
    sanity check that flexion actually flexes.
    """
    base = angles_to_landmarks(np.zeros(20), side="left")
    wrist = base[5]
    d_base = np.linalg.norm(base[1] - wrist)
    reduced = False
    for ch in (4, 5, 6, 7):
        a = np.zeros(20, dtype=np.float32)
        a[ch] = 1.2
        lm = angles_to_landmarks(a, side="left")
        if np.linalg.norm(lm[1] - wrist) < d_base * 0.95:
            reduced = True
            break
    assert reduced, "no index MCP-like channel reduced wrist-to-tip distance"


def test_side_changes_geometry():
    """Left vs right should differ (emg2pose mirrors internally)."""
    a = np.zeros(20, dtype=np.float32)
    a[5] = 0.6
    left = angles_to_landmarks(a, side="left")
    right = angles_to_landmarks(a, side="right")
    assert not np.allclose(left, right)


def test_draw_hand_smoke(tmp_path):
    import os
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
    assert os.path.getsize(out) > 0
