from __future__ import annotations

"""Frequency-domain EMG curve plot: raw (before) vs processed (after).

The processing step is just a filter, so the difference is clearest in the
frequency domain. This picks a few channels and a short segment of signal and
draws each channel's power spectrum (frequency on the x-axis) for raw vs
processed -- the filtering shows up directly as the bands where the processed
curve drops below the raw curve.

Usage:
    python plot_emg_spectrum.py <raw.npz> <processed.npz> [-o out.png]
        [--channels 0,7,14] [--start S] [--duration 6]

If --channels is omitted, the busiest channels in the segment are picked.
If --start is omitted, the most active window is found automatically.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

RAW_C, PROC_C = "#d62728", "#1f77b4"   # red = before, blue = after


def _segment(emg: np.ndarray, fs: int, start_s, dur_s: float):
    T = len(emg)
    W = max(1, int(dur_s * fs))
    if start_s is None:                         # auto: most active window
        e = np.abs(emg).mean(axis=1)
        c = np.convolve(e, np.ones(W) / W, mode="same")
        s = int(np.clip(int(np.argmax(c)) - W // 2, 0, max(0, T - W)))
    else:
        s = int(np.clip(round(start_s * fs), 0, max(0, T - 1)))
    return s, min(T, s + W)


def plot_spectrum(raw_path: str, proc_path: str, out_path: str,
                  channels=None, fs: int = 2000, start_s=None, dur_s: float = 6.0):
    raw = np.asarray(np.load(raw_path, allow_pickle=True)["emg"], float)
    proc = np.asarray(np.load(proc_path, allow_pickle=True)["emg"], float)

    s, e = _segment(raw, fs, start_s, dur_s)
    if channels is None:                        # busiest channels in the segment
        order = np.argsort(raw[s:e].std(axis=0))[::-1]
        channels = sorted(int(c) for c in order[:3])

    nper = min(4096, e - s)
    nch = len(channels)
    fig, axs = plt.subplots(nch, 1, figsize=(10, 2.3 * nch + 1.0),
                            sharex=True, squeeze=False)
    for i, ch in enumerate(channels):
        ax = axs[i, 0]
        f, Pr = welch(raw[s:e, ch], fs=fs, nperseg=nper)
        _, Pp = welch(proc[s:e, ch], fs=fs, nperseg=nper)
        ax.semilogy(f, Pr, color=RAW_C, lw=1.1, label="raw (before)")
        ax.semilogy(f, Pp, color=PROC_C, lw=1.1, label="processed (after)")
        ax.set_ylabel(f"ch{ch}\nPSD")
        ax.grid(True, which="both", alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    axs[-1, 0].set_xlabel("frequency (Hz)")
    axs[-1, 0].set_xlim(0, fs / 2)
    fig.suptitle(f"{os.path.basename(raw_path)}  |  EMG spectrum: raw vs processed "
                 f"(filtering)  |  segment {s/fs:.1f}–{e/fs:.1f}s", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (channels={channels}, seg {s/fs:.1f}-{e/fs:.1f}s)")


def main():
    ap = argparse.ArgumentParser(
        description="Raw vs processed EMG power spectrum (frequency x-axis)")
    ap.add_argument("raw", help="raw (before) recording .npz")
    ap.add_argument("processed", help="processed (after) recording .npz")
    ap.add_argument("-o", "--out", default="emg_spectrum.png", help="output .png")
    ap.add_argument("--channels", default=None,
                    help="comma-separated channel indices (default: busiest 3)")
    ap.add_argument("--start", type=float, default=None,
                    help="segment start in seconds (default: auto active window)")
    ap.add_argument("--duration", type=float, default=6.0,
                    help="segment length in seconds (default 6)")
    ap.add_argument("--fs", type=int, default=2000, help="sample rate (default 2000)")
    args = ap.parse_args()

    chans = None
    if args.channels:
        chans = [int(c) for c in args.channels.split(",") if c.strip() != ""]
    plot_spectrum(args.raw, args.processed, args.out, channels=chans,
                  fs=args.fs, start_s=args.start, dur_s=args.duration)


if __name__ == "__main__":
    main()
