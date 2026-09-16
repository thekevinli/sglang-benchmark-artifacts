#!/usr/bin/env python3
"""Reproducible, resumable Cosmos3-Super quality examples on two H100s.

This intentionally uses the checkpoint's full 1280x720x189 example recipe.
The earlier generation-benchmark/ clips are compact plumbing/latency probes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import runpy
import shutil
import subprocess
import threading
import time
from pathlib import Path

from sglang_omni.client import GenerateRequest


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "generation-quality"
HELPERS = runpy.run_path(str(ROOT / "tests/integration/cosmos3/test_super_gpu.py"))
_check_audio = HELPERS["_check_audio"]
_check_media = HELPERS["_check_media"]
_config = HELPERS["_config"]
_generate = HELPERS["_generate"]
_running = HELPERS["_running"]


class MemorySampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak_mib = [0, 0]
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(0.5):
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            values = [int(line.strip()) for line in output.splitlines()]
            self.peak_mib = [
                max(old, new) for old, new in zip(self.peak_mib, values)
            ]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)


def checkpoint_assets(model: Path) -> dict[str, Path]:
    assets = model / "assets"
    required = {
        "i2v_prompt": assets / "example_i2v_prompt.json",
        "t2v_prompt": assets / "example_t2v_prompt.json",
        "t2vs_prompt": assets / "example_t2vs_prompt.json",
        "negative_prompt": assets / "negative_prompt.json",
        "i2v_input": assets / "example_i2v_input.jpg",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint assets: {missing}")
    return required


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def cases(model: Path) -> dict[str, dict]:
    assets = checkpoint_assets(model)
    negative = json.dumps(load_json(assets["negative_prompt"]), separators=(",", ":"))
    common = {
        "negative_prompt": negative,
        "width": 1280,
        "height": 720,
        "num_frames": 189,
        "adjust_frames": False,
        "fps": 24,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "max_sequence_length": 4096,
        "flow_shift": 10.0,
        "seed": 17,
        "use_resolution_template": False,
        "use_duration_template": False,
    }
    i2v_prompt = json.dumps(load_json(assets["i2v_prompt"]), separators=(",", ":"))
    t2v_prompt = json.dumps(load_json(assets["t2v_prompt"]), separators=(",", ":"))
    t2vs_prompt = json.dumps(load_json(assets["t2vs_prompt"]), separators=(",", ":"))
    return {
        "t2v": {**common, "prompt": t2v_prompt},
        "i2v": {
            **common,
            "prompt": i2v_prompt,
            "image_path": str(assets["i2v_input"]),
        },
        "t2vs": {
            **common,
            "prompt": t2vs_prompt,
            "sound_duration": 7.875,
        },
        "i2vs": {
            **common,
            "prompt": i2v_prompt,
            "image_path": str(assets["i2v_input"]),
            "sound_duration": 7.875,
        },
    }


def save_record(record: dict) -> None:
    records_path = OUT / "results.jsonl"
    with records_path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


async def run_case(runner, name: str, inputs: dict) -> dict:
    destination = OUT / f"{name}-output.mp4"
    started = time.perf_counter()
    with MemorySampler() as memory:
        chunk = await _generate(runner, GenerateRequest(prompt=inputs, stream=False))
    latency = time.perf_counter() - started
    item = chunk.media[0]
    _check_media(item, inputs["num_frames"], inputs["width"], inputs["height"])
    if name.endswith("s"):
        _check_audio(item["path"], inputs["sound_duration"])
    shutil.copy2(item["path"], destination)
    record = {
        "task": name,
        "status": "pass",
        "latency_s": round(latency, 3),
        "peak_memory_mib": memory.peak_mib,
        "output": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "request": inputs,
        "checkpoint_revision": os.environ["COSMOS3_SUPER_CHECKPOINT_REVISION"],
        "native_revision": os.environ["COSMOS3_SUPER_NATIVE_REVISION"],
    }
    save_record(record)
    print("RESULT", json.dumps(record, sort_keys=True), flush=True)
    return record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tasks",
        nargs="*",
        choices=("t2v", "i2v", "t2vs", "i2vs"),
        default=["t2v", "i2v", "t2vs", "i2vs"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs that already exist and are nonempty.",
    )
    args = parser.parse_args()
    model = Path(os.environ["COSMOS3_SUPER_MODEL_PATH"])
    devices = [
        int(value)
        for value in os.environ.get("COSMOS3_SUPER_GPU_IDS", "0,1").split(",")
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    assets = checkpoint_assets(model)
    for key, source in assets.items():
        shutil.copy2(source, OUT / f"source-{key}{source.suffix}")

    all_cases = cases(model)
    selected = []
    for name in args.tasks:
        destination = OUT / f"{name}-output.mp4"
        if destination.exists() and destination.stat().st_size and not args.force:
            print(f"SKIP {name}: {destination} already exists (use --force)", flush=True)
        else:
            selected.append(name)
    if not selected:
        return

    config = _config("generation", str(model), devices, OUT / "native-media")
    async with _running(config) as runner:
        for name in selected:
            await run_case(runner, name, all_cases[name])


if __name__ == "__main__":
    asyncio.run(main())
