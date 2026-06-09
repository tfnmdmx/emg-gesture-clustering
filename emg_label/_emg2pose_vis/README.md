# Vendored emg2pose FK helper

`calculate_hand_error.py` is copied verbatim from emg2pose's
`scripts/visualize_3d/` so this repo's 3D-hand cluster gallery
(`emg_label/hand3d.py`) is self-contained and no longer needs a
`$EMG2POSE_VIS_DIR` pointing at a loose checkout.

It still requires the **`emg2pose` package + torch** to be importable
(it does `import emg2pose.kinematics / constants / UmeTrack`); vendoring
only removes the loose-path dependency, not the package dependency.

Source: emg2pose (Meta). Subject to emg2pose's license (CC-BY-NC). Used
here for non-commercial research only. Override the vendored copy at
runtime with `$EMG2POSE_VIS_DIR` if you need a different version.
