#!/usr/bin/env python3
"""
Local BEHAVIOR dataset loader for Robometer conversion.

This loader intentionally uses only trajectory-level q_score labels from
q_scores_with_episode_ids.json. It does not read DTW or stage-progress files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal smoke-test envs
    def tqdm(iterable, **_kwargs):
        return iterable


HEAD_CAMERA = "observation.images.rgb.head"
DEFAULT_RFT_PATH = r"C:\Users\27642\Desktop\behavior\behavior-1k-rl\dataset\rft_dataset_from_outputs4"
DEFAULT_EXPERT_PATH = r"D:\behavior\behavior_224_rgb"


class BehaviorVideoFrameLoader:
    """Pickle-able video frame loader used by dataset_upload conversion workers."""

    def __init__(self, video_path: str):
        self.video_path = video_path

    def __call__(self) -> np.ndarray:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")

        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames decoded from video: {self.video_path}")
        return np.asarray(frames, dtype=np.uint8)


def _task_to_instruction(task_name: str) -> str:
    task_name = (task_name or "unknown task").strip()
    known = {
        "turning_on_radio": "turn on the radio",
        "picking_up_trash": "pick up trash",
    }
    return known.get(task_name, task_name.replace("_", " "))


def _quality_from_q_score(q_score: float) -> str:
    if q_score >= 1.0:
        return "successful"
    if q_score <= 0.0:
        return "failure"
    return "suboptimal"


def _episode_video_path(dataset_root: Path, task_index: int, episode_id: int) -> Path:
    return dataset_root / "videos" / f"task-{task_index:04d}" / HEAD_CAMERA / f"episode_{episode_id:08d}.mp4"


def _normalize_task_ids(task_ids: Any) -> set[int] | None:
    if task_ids is None:
        return None
    if isinstance(task_ids, str):
        task_ids = [part.strip() for part in task_ids.split(",") if part.strip()]
    normalized = {int(task_id) for task_id in task_ids}
    return normalized or None


def _load_rft_trajectories(
    rft_path: Path,
    max_trajectories: int | None = None,
    task_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    q_scores_path = rft_path / "q_scores_with_episode_ids.json"
    if not q_scores_path.exists():
        raise FileNotFoundError(f"Missing q-score file: {q_scores_path}")

    with open(q_scores_path, encoding="utf-8") as f:
        payload = json.load(f)

    episodes = payload.get("episodes", [])
    trajectories: list[dict[str, Any]] = []
    quality_counts: dict[str, int] = {}

    for ep in tqdm(episodes, desc="Loading BEHAVIOR RFT episodes"):
        if max_trajectories is not None and max_trajectories > 0 and len(trajectories) >= max_trajectories:
            break

        q_score = float(ep["q_score"])
        task_name = str(ep["task_name"])
        task_index = int(ep["task_index"])
        if task_ids is not None and task_index not in task_ids:
            continue
        episode_id = int(ep["episode_id"])
        video_path = _episode_video_path(rft_path, task_index, episode_id)
        if not video_path.exists():
            print(f"Warning: missing RFT head-camera video, skipping: {video_path}")
            continue

        quality = _quality_from_q_score(q_score)
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

        trajectories.append({
            "id": f"behavior_rft_{episode_id:08d}",
            "frames": BehaviorVideoFrameLoader(str(video_path)),
            "task": _task_to_instruction(task_name),
            "is_robot": True,
            "quality_label": quality,
            "partial_success": q_score,
            "data_source": "behavior_rft",
            "metadata": {
                "source_video_path": str(video_path),
                "source_rel_run_path": ep.get("source_rel_run_path"),
                "task_name": task_name,
                "task_index": task_index,
                "episode_id": episode_id,
                "q_score": q_score,
            },
        })

    print(f"Loaded {len(trajectories)} BEHAVIOR RFT trajectories")
    for quality in sorted(quality_counts):
        print(f"  RFT {quality}: {quality_counts[quality]}")
    return trajectories


def _discover_expert_videos(expert_path: Path, task_ids: set[int] | None = None) -> list[Path]:
    videos_root = expert_path / "videos"
    if not videos_root.exists():
        raise FileNotFoundError(f"Missing expert videos directory: {videos_root}")
    if task_ids is None:
        return sorted(videos_root.glob(f"task-*/{HEAD_CAMERA}/episode_*.mp4"))

    video_paths: list[Path] = []
    for task_id in sorted(task_ids):
        video_paths.extend(sorted((videos_root / f"task-{task_id:04d}" / HEAD_CAMERA).glob("episode_*.mp4")))
    return video_paths


def _load_expert_trajectories(
    expert_path: Path,
    max_trajectories: int | None = None,
    task_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    video_paths = _discover_expert_videos(expert_path, task_ids=task_ids)
    trajectories: list[dict[str, Any]] = []

    for video_path in tqdm(video_paths, desc="Loading BEHAVIOR expert episodes"):
        if max_trajectories is not None and max_trajectories > 0 and len(trajectories) >= max_trajectories:
            break

        task_dir = video_path.parents[1].name
        task_index = int(task_dir.split("-")[-1])
        episode_stem = video_path.stem.replace("episode_", "")
        episode_id = int(episode_stem)

        # Expert metadata does not include explicit task_name in this loader. Keep the
        # task-index mapping aligned with the known RFT tasks used in this dataset.
        task_name_by_index = {
            0: "turning_on_radio",
            1: "picking_up_trash",
        }
        task_name = task_name_by_index.get(task_index, f"task {task_index}")

        trajectories.append({
            "id": f"behavior_expert_task{task_index:04d}_{episode_id:08d}",
            "frames": BehaviorVideoFrameLoader(str(video_path)),
            "task": _task_to_instruction(task_name),
            "is_robot": False,
            "quality_label": "successful",
            "partial_success": 1.0,
            "data_source": "behavior_expert",
            "metadata": {
                "source_video_path": str(video_path),
                "task_name": task_name,
                "task_index": task_index,
                "episode_id": episode_id,
            },
        })

    print(f"Loaded {len(trajectories)} BEHAVIOR expert trajectories")
    return trajectories


def load_behavior_local_dataset(dataset_path: str, max_trajectories: int | None = None) -> dict[str, list[dict]]:
    """Load local BEHAVIOR RFT rollouts and expert demos.

    dataset_path may be:
      - a JSON object string with rft_path/expert_path keys,
      - a path to a JSON config with those keys,
      - empty, in which case the default local paths are used.
    """
    rft_path = Path(DEFAULT_RFT_PATH)
    expert_path = Path(DEFAULT_EXPERT_PATH)
    task_ids: set[int] | None = {0, 1}

    if dataset_path:
        config: dict[str, Any] = {}
        stripped = dataset_path.strip()
        if stripped.startswith("{"):
            config = json.loads(stripped)
        elif Path(os.path.expanduser(dataset_path)).is_file():
            with open(Path(os.path.expanduser(dataset_path)), encoding="utf-8") as f:
                config = json.load(f)
        elif Path(os.path.expanduser(dataset_path)).exists():
            # Treat a direct path as the RFT root and keep the default expert root.
            config = {"rft_path": dataset_path}
        if config:
            rft_path = Path(os.path.expanduser(config.get("rft_path", str(rft_path))))
            expert_path = Path(os.path.expanduser(config.get("expert_path", str(expert_path))))
            task_ids = _normalize_task_ids(config.get("task_ids", task_ids))

    if not rft_path.exists():
        raise FileNotFoundError(f"BEHAVIOR RFT path not found: {rft_path}")
    if not expert_path.exists():
        raise FileNotFoundError(f"BEHAVIOR expert path not found: {expert_path}")

    rft_limit = None
    expert_limit = None
    if max_trajectories is not None and max_trajectories > 0:
        # Keep smoke tests balanced across both sources.
        rft_limit = max_trajectories
        expert_limit = max_trajectories

    if task_ids is not None:
        print(f"Filtering BEHAVIOR local tasks to task IDs: {sorted(task_ids)}")

    trajectories = _load_rft_trajectories(rft_path, max_trajectories=rft_limit, task_ids=task_ids)
    trajectories.extend(_load_expert_trajectories(expert_path, max_trajectories=expert_limit, task_ids=task_ids))

    task_data: dict[str, list[dict]] = {}
    for traj in trajectories:
        task_data.setdefault(traj["task"], []).append(traj)

    print("BEHAVIOR local task summary:")
    for task, task_trajs in sorted(task_data.items()):
        counts: dict[str, int] = {}
        for traj in task_trajs:
            counts[traj["quality_label"]] = counts.get(traj["quality_label"], 0) + 1
        print(f"  {task}: {len(task_trajs)} trajectories | {counts}")
    return task_data
