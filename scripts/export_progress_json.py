#!/usr/bin/env python3
"""Export Robometer progress predictions as per-frame expected-value JSON.

Example:
  uv run python scripts/export_progress_json.py \
    --model-path ./logs/behavior_local_rbm_lora/final_checkpoint \
    --video /path/to/video.mp4 \
    --task "turn on the radio" \
    --episode-id 3 \
    --sampling-method uniform \
    --num-sampled-frames 64 \
    --out expected_values_series_rbm_task0_interp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decord
import numpy as np
import torch
from tqdm import tqdm

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def load_video_frames(
    video_path: str,
    *,
    sampling_method: str,
    stride: int,
    num_sampled_frames: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load sampled video frames.

    Returns:
        sampled_frames: uint8 array [T, H, W, C]
        sampled_indices: original frame indices corresponding to sampled_frames
        total_frames: total number of frames in the original video
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    if total_frames <= 0:
        raise RuntimeError(f"No frames found in video: {video_path}")

    if sampling_method == "stride":
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        sampled_indices = np.arange(0, total_frames, stride, dtype=np.int64)
        if sampled_indices[-1] != total_frames - 1:
            sampled_indices = np.append(sampled_indices, total_frames - 1)
    elif sampling_method == "uniform":
        if num_sampled_frames <= 0:
            raise ValueError(f"num_sampled_frames must be positive, got {num_sampled_frames}")
        count = min(num_sampled_frames, total_frames)
        sampled_indices = np.linspace(0, total_frames - 1, count, dtype=np.int64)
        sampled_indices = np.unique(sampled_indices)
    else:
        raise ValueError(f"Unsupported sampling_method: {sampling_method}")

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


def _load_model_for_progress(
    model_path: str,
    device: torch.device | None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path,
        device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    return device, tokenizer, reward_model, batch_collator, is_discrete, num_bins


def predict_progress_full(
    model_path: str,
    sampled_frames: np.ndarray,
    task: str,
    *,
    device: torch.device | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict progress for all sampled frames in one full-trajectory forward pass."""
    device, tokenizer, reward_model, batch_collator, is_discrete, num_bins = _load_model_for_progress(
        model_path,
        device,
    )

    traj = Trajectory(
        frames=sampled_frames,
        frames_shape=tuple(sampled_frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": int(sampled_frames.shape[0])},
        video_embeddings=None,
    )
    sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    with torch.inference_mode():
        results = compute_batch_outputs(
            reward_model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )

    progress_pred = results.get("progress_pred", [])
    if not progress_pred or len(progress_pred[0]) == 0:
        raise RuntimeError("Model returned no progress predictions.")

    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []

    return (
        np.asarray(progress_pred[0], dtype=np.float32),
        np.asarray(success_probs[0], dtype=np.float32) if success_probs else np.asarray([], dtype=np.float32),
    )


def predict_progress_windowed(
    model_path: str,
    sampled_frames: np.ndarray,
    task: str,
    *,
    window_size: int,
    device: torch.device | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one progress value per sampled frame using bounded-size windows."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    device, tokenizer, reward_model, batch_collator, is_discrete, num_bins = _load_model_for_progress(
        model_path,
        device,
    )

    progress_values: list[float] = []
    success_values: list[float] = []

    with torch.inference_mode():
        for end_idx in tqdm(range(len(sampled_frames)), desc="Predicting sampled progress"):
            start_idx = max(0, end_idx - window_size + 1)
            window = sampled_frames[start_idx : end_idx + 1]

            traj = Trajectory(
                frames=window,
                frames_shape=tuple(window.shape),
                task=task,
                id=str(end_idx),
                metadata={"subsequence_length": int(window.shape[0])},
                video_embeddings=None,
            )
            sample = ProgressSample(trajectory=traj, sample_type="progress")
            batch = batch_collator([sample])

            progress_inputs = batch["progress_inputs"]
            for key, value in progress_inputs.items():
                if hasattr(value, "to"):
                    progress_inputs[key] = value.to(device)

            results = compute_batch_outputs(
                reward_model,
                tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=is_discrete,
                num_bins=num_bins,
            )

            progress_pred = results.get("progress_pred", [])
            if not progress_pred or len(progress_pred[0]) == 0:
                raise RuntimeError(f"Model returned no progress prediction for sampled frame {end_idx}")
            progress_values.append(float(progress_pred[0][-1]))

            outputs_success = results.get("outputs_success", {})
            success_probs = outputs_success.get("success_probs", []) if outputs_success else []
            if success_probs and len(success_probs[0]) > 0:
                success_values.append(float(success_probs[0][-1]))

            del batch, progress_inputs, results

    return np.asarray(progress_values, dtype=np.float32), np.asarray(success_values, dtype=np.float32)


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
        "--sampling-method",
        choices=["uniform", "stride"],
        default="uniform",
        help="How to sample frames before interpolation. uniform covers the whole video; stride samples every N frames.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=["full", "windowed"],
        default="full",
        help="full sends all sampled frames in one pass; windowed uses recent sampled history windows.",
    )
    parser.add_argument(
        "--num-sampled-frames",
        type=int,
        default=64,
        help="Number of frames to uniformly sample across the whole video when --sampling-method=uniform.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample one frame every N original frames when --sampling-method=stride. For 30 fps, 10 means 3 Hz.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Maximum sampled frames sent to the model at once. Keep this near training max_frames to avoid OOM.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent. Use 0 for compact output.")
    args = parser.parse_args()

    frames, sampled_indices, total_frames = load_video_frames(
        args.video,
        sampling_method=args.sampling_method,
        stride=args.stride,
        num_sampled_frames=args.num_sampled_frames,
    )
    device = torch.device(args.device) if args.device else None

    if args.inference_mode == "full":
        progress_values, success_probs = predict_progress_full(
            model_path=args.model_path,
            sampled_frames=frames,
            task=args.task,
            device=device,
        )
    else:
        progress_values, success_probs = predict_progress_windowed(
            model_path=args.model_path,
            sampled_frames=frames,
            task=args.task,
            window_size=args.window_size,
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
        "sampling_method": args.sampling_method,
        "inference_mode": args.inference_mode,
        "stride": int(args.stride),
        "num_sampled_frames": int(args.num_sampled_frames),
        "window_size": int(args.window_size),
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
