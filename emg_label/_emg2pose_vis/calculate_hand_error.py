#!/usr/bin/env python3
"""
Virtual Hand Position Error Calculator
Computes position errors (in mm) from joint angle errors using emg2pose kinematics.

Usage:
    # From numpy arrays (real-time inference)
    pred_angles = np.random.randn(20)  # Predicted joint angles (radians)
    target_angles = np.random.randn(20)  # Ground truth joint angles (radians)
    errors = calculate_hand_errors(pred_angles, target_angles)

    # From npz file (offline evaluation)
    errors = calculate_hand_errors_from_npz("path/to/file.npz")
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List
import argparse

# Add emg2pose to path
sys.path.insert(0, str(Path(__file__).parent / "emg2pose"))

from emg2pose.kinematics import (
    load_default_hand_model,
    TorchHandModel,
    forward_kinematics as emg2pose_forward_kinematics,
)
from emg2pose.constants import LANDMARKS, FINGERS, NO_MOVEMENT_LANDMARKS
from emg2pose.UmeTrack.lib.common.hand import mirrored_hand_model, HandModel


class HandPositionErrorCalculator:
    """
    Calculate position errors (in mm) from joint angle predictions.
    Uses emg2pose's forward kinematics to convert angles to 3D positions.
    """

    # Landmark indices for fingertips
    FINGERTIP_INDICES = [0, 1, 2, 3, 4]  # thumb, index, middle, ring, pinky
    WRIST_INDEX = 5

    # Per-finger landmark indices (including intermediate joints)
    FINGER_LANDMARKS = {
        "thumb": [0, 6, 7],           # fingertip, intermediate, distal
        "index": [1, 8, 9, 10],       # fingertip, proximal, intermediate, distal
        "middle": [2, 11, 12, 13],
        "ring": [3, 14, 15, 16],
        "pinky": [4, 17, 18, 19],
    }

    def __init__(self, device: str = "cpu", side: str = "left"):
        """
        Initialize the calculator with emg2pose hand model.

        Args:
            device: torch device ('cpu' or 'cuda')
            side: 'left' or 'right' hand (default: 'left')
                  The emg2pose model is trained on left hand data.
                  For right hand, the model will be mirrored.
        """
        self.device = torch.device(device)
        self.side = side.lower()
        hand_model = load_default_hand_model()

        # Mirror the hand model for right hand
        # (emg2pose treats right hand as mirrored left hand)
        if self.side == "right":
            # Create a mask for mirroring (X axis)
            to_mirror = torch.tensor([
                False, False,  # wrist (not used)
                True, True, True, True,  # thumb
                True, True, True, True,  # index
                True, True, True, True,  # middle
                True, True, True, True,  # ring
                True, True, True, True,  # pinky
            ])
            hand_model = mirrored_hand_model(hand_model, to_mirror)

        self.hand_model = TorchHandModel(hand_model)
        self.hand_model.to(self.device)

    def angles_to_positions(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Convert joint angles to 3D landmark positions.

        Args:
            joint_angles: (..., 20) array of joint angles in radians

        Returns:
            (..., 21, 3) array of landmark positions in millimeters
        """
        # Ensure numpy input
        is_single = joint_angles.ndim == 1
        if is_single:
            joint_angles = joint_angles[np.newaxis, :]

        # Convert to tensor
        angles_tensor = torch.from_numpy(joint_angles.astype(np.float32))
        angles_tensor = angles_tensor.to(self.device)

        # Reshape: (N, 20) -> (N, 22) by appending 2 wrist zeros at the end
        # Hand model DOF order: [0-3]=thumb, [4-7]=index, ..., [16-19]=pinky, [20-21]=wrist
        # This matches the training code (kinematics.py forward_kinematics)
        N = angles_tensor.shape[0]

        # Add null wrist angles at the END (joints 20-21 are wrist DOFs)
        wrist_zeros = torch.zeros(N, 2, dtype=angles_tensor.dtype, device=angles_tensor.device)
        angles_tensor = torch.cat([angles_tensor, wrist_zeros], dim=1)  # (N, 22)

        # Run forward kinematics using the lower-level function
        # This expects (..., 22) format and returns (..., 21, 3)
        # Output is in millimeters (not meters!)
        from emg2pose.kinematics import _batched_forward_kinematics
        with torch.no_grad():
            positions = _batched_forward_kinematics(
                angles_tensor,
                self.hand_model.to_hand_model(),
                degrees=False
            )

        # Convert back to numpy: (..., 21, 3)
        positions = positions.cpu().numpy()

        if is_single:
            positions = positions[0]

        return positions

    def calculate_errors_aligned(
        self,
        pred_angles: np.ndarray,
        target_angles: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate position errors between already-aligned predicted and target joint angles.

        Both inputs must have the same length and be temporally aligned (same sampling rate).

        Args:
            pred_angles: (N, 20) predicted joint angles in radians
            target_angles: (N, 20) ground truth joint angles in radians (same length, already aligned)

        Returns:
            Dictionary with error metrics in millimeters
        """
        assert len(pred_angles) == len(target_angles), (
            f"pred and target must have same length, got {len(pred_angles)} vs {len(target_angles)}"
        )

        # Convert to positions (output is already in millimeters)
        pred_pos = self.angles_to_positions(pred_angles)  # (N, 21, 3) in mm
        target_pos = self.angles_to_positions(target_angles)  # (N, 21, 3) in mm

        # Compute Euclidean distance errors (millimeters)
        errors_mm = np.linalg.norm(pred_pos - target_pos, axis=-1)  # (N, 21)

        # Compute metrics
        metrics = {}

        # Overall statistics
        metrics["mean_error_mm"] = float(np.mean(errors_mm))
        metrics["median_error_mm"] = float(np.median(errors_mm))
        metrics["std_error_mm"] = float(np.std(errors_mm))
        metrics["max_error_mm"] = float(np.max(errors_mm))

        # Fingertip errors (5 fingertips)
        fingertip_errors = errors_mm[..., self.FINGERTIP_INDICES]
        metrics["fingertip_mean_mm"] = float(np.mean(fingertip_errors))
        metrics["fingertip_median_mm"] = float(np.median(fingertip_errors))

        # Individual fingertip errors
        fingertip_names = ["thumb", "index", "middle", "ring", "pinky"]
        for i, name in enumerate(fingertip_names):
            metrics[f"{name}_fingertip_mm"] = float(np.mean(errors_mm[..., i]))

        # Per-finger errors (all landmarks of each finger)
        for finger_name, indices in self.FINGER_LANDMARKS.items():
            finger_errors = errors_mm[..., indices]
            metrics[f"{finger_name}_mean_mm"] = float(np.mean(finger_errors))

        # Excluding non-moving landmarks (proximal frames attached to wrist)
        moving_indices = [
            i for i, lm in enumerate(LANDMARKS)
            if lm.name not in NO_MOVEMENT_LANDMARKS
        ]
        moving_errors = errors_mm[..., moving_indices]
        metrics["moving_landmarks_mean_mm"] = float(np.mean(moving_errors))

        return metrics

    def calculate_errors(
        self,
        pred_angles: np.ndarray,
        target_angles: np.ndarray,
        emg_window_size: int = 1791,
        emg_chunk_size: int = 80,
        prefill_samples: int = 1711,
    ) -> Dict[str, float]:
        """
        Calculate position errors between predicted and target joint angles.

        DEPRECATED: Use calculate_errors_aligned() with pre-aligned data instead.
        This method assumes target_angles is raw 2000Hz data and does its own alignment,
        which can cause double-alignment bugs if the caller already downsampled the GT.

        Args:
            pred_angles: (N, 20) predicted joint angles in radians (downsampled)
            target_angles: (M, 20) ground truth joint angles in radians (original 2000Hz)
            emg_window_size: EMG window size for feature extraction
            emg_chunk_size: EMG chunk size (stride) for predictions
            prefill_samples: Number of samples used for initial buffer fill

        Returns:
            Dictionary with error metrics in millimeters
        """
        # Align target with prediction time indices
        # Causal model: prediction corresponds to end of window (LEFT_CONTEXT=1791)
        left_context = emg_window_size  # 1791
        downsample_step = emg_chunk_size // 2  # 40 (50Hz output from half-step interpolation)

        pred_indices = np.arange(len(pred_angles)) * downsample_step + left_context

        # Handle edge case where indices exceed target length
        valid_mask = pred_indices < len(target_angles)
        if not valid_mask.all():
            pred_angles = pred_angles[valid_mask]
            pred_indices = pred_indices[valid_mask]

        # Extract aligned target angles
        aligned_target = target_angles[pred_indices]

        return self.calculate_errors_aligned(pred_angles, aligned_target)

    def print_errors(self, metrics: Dict[str, float]):
        """Print error metrics in a formatted table."""
        print("\n" + "=" * 60)
        print(f"HAND POSITION ERRORS (mm) - {self.side.upper()} HAND")
        print("=" * 60)

        print("\nOverall Statistics:")
        print(f"  Mean Error:   {metrics['mean_error_mm']:.2f} mm")
        print(f"  Median Error: {metrics['median_error_mm']:.2f} mm")
        print(f"  Std Error:    {metrics['std_error_mm']:.2f} mm")
        print(f"  Max Error:    {metrics['max_error_mm']:.2f} mm")

        print("\nFingertip Errors:")
        print(f"  All Fingertips: {metrics['fingertip_mean_mm']:.2f} mm (mean)")
        for finger in FINGERS:
            key = f"{finger}_fingertip_mm"
            if key in metrics:
                print(f"    {finger.capitalize()}: {metrics[key]:.2f} mm")

        print("\nPer-Finger Errors (all joints):")
        for finger in FINGERS:
            key = f"{finger}_mean_mm"
            if key in metrics:
                print(f"    {finger.capitalize()}: {metrics[key]:.2f} mm")

        print(f"\nMoving Landmarks: {metrics['moving_landmarks_mean_mm']:.2f} mm")
        print("=" * 60 + "\n")


# Global instances for convenience
_calculator_left: Optional[HandPositionErrorCalculator] = None
_calculator_right: Optional[HandPositionErrorCalculator] = None


def get_calculator(device: str = "cpu", side: str = "left") -> HandPositionErrorCalculator:
    """Get or create the global calculator instance."""
    global _calculator_left, _calculator_right
    if side.lower() == "right":
        if _calculator_right is None:
            _calculator_right = HandPositionErrorCalculator(device=device, side="right")
        return _calculator_right
    else:
        if _calculator_left is None:
            _calculator_left = HandPositionErrorCalculator(device=device, side="left")
        return _calculator_left


def calculate_hand_errors(
    pred_angles: np.ndarray,
    target_angles: np.ndarray,
    device: str = "cpu",
    side: str = "left",
    print_output: bool = True,
) -> Dict[str, float]:
    """
    Calculate position errors from joint angles.

    Args:
        pred_angles: (..., 20) predicted joint angles in radians
        target_angles: (..., 20) ground truth joint angles in radians
        device: torch device
        side: 'left' or 'right' hand (default: 'left')
        print_output: whether to print results

    Returns:
        Dictionary with error metrics in millimeters
    """
    calculator = get_calculator(device=device, side=side)
    metrics = calculator.calculate_errors(pred_angles, target_angles)

    if print_output:
        print(f"\n[{side.upper()} HAND]")
        calculator.print_errors(metrics)

    return metrics


def calculate_hand_errors_from_npz(
    npz_path: str,
    pred_key: str = "pred_joint_angles",
    target_key: str = "joint_angles",
    device: str = "cpu",
    side: str = "left",
    print_output: bool = True,
) -> Dict[str, float]:
    """
    Calculate position errors from a npz file.

    Args:
        npz_path: path to npz file containing joint angles
        pred_key: key for predicted angles (if different from target)
        target_key: key for ground truth angles
        device: torch device
        side: 'left' or 'right' hand (default: 'left')
        print_output: whether to print results

    Returns:
        Dictionary with error metrics in millimeters
    """
    data = np.load(npz_path)

    # Get predicted angles (might be same as target for data analysis)
    if pred_key in data:
        pred_angles = data[pred_key]
    else:
        pred_angles = data[target_key]

    target_angles = data[target_key]

    return calculate_hand_errors(pred_angles, target_angles, device, side, print_output)


def calculate_angle_errors(
    pred_angles: np.ndarray,
    target_angles: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate pure angular errors (in degrees) for comparison.

    Args:
        pred_angles: (..., 20) predicted joint angles in radians
        target_angles: (..., 20) ground truth joint angles in radians

    Returns:
        Dictionary with angular error metrics in degrees
    """
    # Absolute angular error
    angle_errors = np.abs(pred_angles - target_angles)

    # Convert to degrees
    angle_errors_deg = np.rad2deg(angle_errors)

    metrics = {
        "mean_angle_error_deg": float(np.mean(angle_errors_deg)),
        "median_angle_error_deg": float(np.median(angle_errors_deg)),
        "std_angle_error_deg": float(np.std(angle_errors_deg)),
        "max_angle_error_deg": float(np.max(angle_errors_deg)),
    }

    return metrics


def _test_forward_kinematics():
    """Test forward kinematics output format."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "emg2pose-main"))

    # Create test angles (all zeros = neutral hand pose)
    test_angles = np.zeros((10, 20), dtype=np.float32)

    calculator = HandPositionErrorCalculator(device="cpu", side="left")

    print(f"Input shape: {test_angles.shape}")
    positions = calculator.angles_to_positions(test_angles)
    print(f"Output shape: {positions.shape}")
    print(f"Output range: min={positions.min():.4f}, max={positions.max():.4f}")
    print(f"First position sample:\n{positions[0]}")

    # Test with same angles (should give zero error)
    errors = calculator.calculate_errors(test_angles, test_angles)
    print(f"\nSelf-comparison errors (should be ~0):")
    for k, v in errors.items():
        if 'mean' in k:
            print(f"  {k}: {v:.6f}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Calculate hand position errors from joint angles"
    )
    parser.add_argument(
        "input",
        nargs='?',
        default=None,
        help="Path to npz file or 'random' for synthetic test",
    )
    parser.add_argument(
        "--test-fk",
        action="store_true",
        help="Test forward kinematics output format",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="left",
        help="Hand side: 'left' or 'right' (default: left)",
    )
    parser.add_argument(
        "--pred-key",
        default="joint_angles",
        help="Key for predicted angles in npz (default: joint_angles)",
    )
    parser.add_argument(
        "--target-key",
        default="joint_angles",
        help="Key for target angles in npz (default: joint_angles)",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Don't print results",
    )

    args = parser.parse_args()

    if args.test_fk:
        _test_forward_kinematics()
        return

    if args.input is None:
        parser.print_help()
        return

    if args.input == "random":
        # Test with random data
        print(f"Generating random test data for {args.side} hand...")
        pred_angles = np.random.randn(100, 20) * 0.5  # 100 time steps
        target_angles = pred_angles + np.random.randn(100, 20) * 0.1  # Small error

        metrics = calculate_hand_errors(
            pred_angles,
            target_angles,
            device=args.device,
            side=args.side,
            print_output=not args.no_print,
        )

        # Also show angular errors for comparison
        angle_metrics = calculate_angle_errors(pred_angles, target_angles)
        print("\nAngular Errors (for reference):")
        for key, val in angle_metrics.items():
            print(f"  {key}: {val:.2f}")

    else:
        metrics = calculate_hand_errors_from_npz(
            args.input,
            pred_key=args.pred_key,
            target_key=args.target_key,
            device=args.device,
            side=args.side,
            print_output=not args.no_print,
        )


if __name__ == "__main__":
    main()
