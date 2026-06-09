#!/usr/bin/env python3
"""Export Robometer progress predictions as per-frame expected-value JSON.

Example:
  uv run python scripts/export_progress_json.py \
    --model-path ./logs/behavior_local_rbm_lora/final_checkpoint \
    --video /path/to/video.mp4 \
    --task "turn on the radio" \
    --episode-id 3 \
    --stride 10 \
    --out expected_values_series_rbm_task0_interp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decord
import numpy as np
import torch

from scripts.example_inference_local import compute_rewards_per_frame_local


def load_video_stride(video_path: str, stride: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Load video frames every `stride` original frames.

    Returns:
        sampled_frames: uint8 array [T, H, W, C]
        sampled_indices: original frame indices corresponding to sampled_frames
        total_frames: total number of frames in the original video
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    if total_frames <= 0:
        raise RuntimeError(f"No frames found in video: {video_path}")

    sampled_indices = np.arange(0, total_frames, stride, dtype=np.int64)
    if sampled_indices[-1] != total_frames - 1:
        sampled_indices = np.append(sampled_indices, total_frames - 1)

    sampled_frames = vr.get_batch(sampled_indices).asnumpy()
    if sampled_frames.dtype != np.uint8:
        sampled_frames = np.clip(sampled_frames, 0, 255).astype(np.uint8)

    return sampled_frames, sampled_indices, total_frames


def interpolate_to_all_frames(
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    total_frames: int,
) -> np.ndarray:
    """Linearly interpolate sparse progress predictions to every frame."""
    sampled_values = np.asarray(sampled_values, dtype=np.float64).reshape(-1)
    if sampled_values.size == 0:
        raise RuntimeError("Model returned no progress predictions.")

    if sampled_values.size != sampled_indices.size:
        raise RuntimeError(
            "Prediction count does not match sampled frame count: "
            f"{sampled_values.size} predictions for {sampled_indices.size} sampled frames."
        )

    all_indices = np.arange(total_frames, dtype=np.float64)
    dense = np.interp(all_indices, sampled_indices.astype(np.float64), sampled_values)
    return dense


def write_expected_values_json(
    output_path: str,
    episode_id: str,
    dense_values: np.ndarray,
    indent: int | None,
) -> None:
    payload = {
        str(episode_id): {
            str(frame_idx): float(value)
            for frame_idx, value in enumerate(np.asarray(dense_values).reshape(-1))
        }
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Robometer progress inference on a video and export per-frame JSON labels."
    )
    parser.add_argument("--model-path", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--task", required=True, help="Task instruction")
    parser.add_argument("--episode-id", required=True, help="Outer JSON key, e.g. 3")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample one frame every N original frames before interpolation. For 30 fps, 10 means 3 Hz.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent. Use 0 for compact output.")
    args = parser.parse_args()

    frames, sampled_indices, total_frames = load_video_stride(args.video, args.stride)
    device = torch.device(args.device) if args.device else None

    progress_values, success_probs = compute_rewards_per_frame_local(
        model_path=args.model_path,
        video_frames=frames,
        task=args.task,
        device=device,
    )
    dense_values = interpolate_to_all_frames(sampled_indices, progress_values, total_frames)
    write_expected_values_json(
        output_path=args.out,
        episode_id=args.episode_id,
        dense_values=dense_values,
        indent=None if args.indent == 0 else args.indent,
    )

    summary = {
        "video": args.video,
        "task": args.task,
        "episode_id": str(args.episode_id),
        "total_frames": int(total_frames),
        "stride": int(args.stride),
        "sampled_frames": int(len(sampled_indices)),
        "predictions": int(len(progress_values)),
        "success_predictions": int(len(success_probs)),
        "out": args.out,
        "progress_min": float(np.min(dense_values)),
        "progress_max": float(np.max(dense_values)),
        "progress_mean": float(np.mean(dense_values)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
