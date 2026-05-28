from emg_label.config import Config


def test_config_defaults():
    c = Config()
    assert c.fs == 2000
    assert c.smooth_ms == 150.0
    assert c.min_action_s == 0.4
    assert c.min_rest_gap_s == 0.2
    assert c.k_min == 12
    assert c.k_max == 30
    assert c.out_dir == "out"
    assert c.enter_thresh is None
    assert c.exit_thresh is None


def test_config_override():
    c = Config(fs=1000, k_max=25)
    assert c.fs == 1000
    assert c.k_max == 25
