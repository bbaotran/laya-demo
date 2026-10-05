"""Dự đoán thật từ mô hình Laya, kèm "khiên an toàn" (tùy chọn).

Dựa trên laya_mlx/snake/policy.py (mizorewww/laya-mlx, Apache-2.0).
Mô hình chạy bằng PyTorch (CPU hoặc CUDA) qua `laya.load(...)`. Cách hỏi mô hình (prompt)
là bản "compact" của mizorewww/laya-mlx.
"""

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .game import DIRECTIONS

HF_REPO = "convaiinnovations/laya"
HF_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
DEFAULT_DIR = Path("models") / "laya"
DEFAULT_SUBFOLDER = "multilingual"
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def download_checkpoint(directory=DEFAULT_DIR, subfolder=DEFAULT_SUBFOLDER):
    """Tải mô hình về một thư mục thường (không dùng symlink, chạy được trên Windows)."""
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    return snapshot_download(
        HF_REPO,
        revision=HF_REVISION,
        local_dir=str(directory),
        allow_patterns=[prefix + name for name in CHECKPOINT_FILES],
    )


def local_checkpoint(value=None, subfolder=DEFAULT_SUBFOLDER):
    """Tìm thư mục mô hình trên máy. Khi chơi game không dùng mạng."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    path = Path(value).expanduser() if value else DEFAULT_DIR
    if not (path / (subfolder or "") / "model.safetensors").is_file():
        raise FileNotFoundError(f"Không thấy mô hình trong: {path / (subfolder or '')}")
    return path


@dataclass
class Decision:
    probabilities: dict     # xác suất AI chọn mỗi hướng
    proposed: str           # hướng AI muốn đi
    executed: str           # hướng thực sự sẽ đi (có thể khác nếu khiên can thiệp)
    safe_directions: list
    intervened: bool        # khiên có đổi nước đi không
    dead_end_risk: float
    food_reachable: float
    inference_ms: float

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(self, model=None, *, subfolder=DEFAULT_SUBFOLDER, guarded=True, device=None):
        import laya

        path = local_checkpoint(model, subfolder)
        self.agent = laya.load(str(path), device=device, subfolder=subfolder or None)
        self.guarded = guarded
        dev = self.agent.device
        if dev.type == "cpu":
            self.device_name = "CPU"
        else:
            import torch

            self.device_name = torch.cuda.get_device_name(dev)

    def decide(self, game):
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe:
            # Chỉ xảy ra khi người chơi phá rắn: chọn nước hợp lệ còn nhiều chỗ nhất.
            safe = [m for m in moves if m.legal]
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, _ = game.food_reachability()

        state = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        criteria = {
            m.direction: (
                "Blocked. Collision." if not m.legal
                else "Unsafe. Traps the snake." if not m.safe
                else "Safe. Eat food now. Best." if m.eats
                else "Safe. Best route to food." if m.direction == preferred
                else "Safe. Slower route."
            )
            for m in moves
        }
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Choose the best safe move toward food.",
                "criteria": criteria,
            },
            "risk": {"type": "noul", "instructions": "Is a safe route available?"},
            "food": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
        }

        start = time.perf_counter()
        answers = self.agent.predict(state, questions)["answers"]
        inference_ms = (time.perf_counter() - start) * 1000

        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["food"]["noul"]]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
            raise ValueError("Mô hình trả về xác suất không hợp lệ")

        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and allowed and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"],
            food_reachable=answers["food"]["noul"],
            inference_ms=inference_ms,
        )
