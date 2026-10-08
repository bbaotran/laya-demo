"""Dự đoán thật từ mô hình Laya, kèm "khiên an toàn" (tùy chọn).

Dựa trên laya_mlx/snake/policy.py (mizorewww/laya-mlx, Apache-2.0).
Mô hình chạy bằng PyTorch (CPU hoặc CUDA). Có hai cách nạp weights (tham số `engine`):
  - "laya":   dùng gói `laya` từ pip (`laya.load(...)`).
  - "native": tự nạp weights vào kiến trúc trong common.py bằng loader.py, không cần gói `laya`.
Cách hỏi mô hình (prompt) là bản "compact" của upstream.
"""

import math
import time
from dataclasses import asdict, dataclass

from .game import DIRECTIONS
# Việc tải weights và tìm thư mục mô hình nằm trong loader.py; import lại ở đây để gui.py dùng chung.
from .loader import DEFAULT_DIR, DEFAULT_SUBFOLDER, download_checkpoint, local_checkpoint  # noqa: F401


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
    def __init__(self, model=None, *, subfolder=DEFAULT_SUBFOLDER, guarded=True, device=None, engine="laya"):
        path = local_checkpoint(model, subfolder)
        if engine == "native":
            from .loader import load_agent

            self.agent = load_agent(path, subfolder, device)
        elif engine == "laya":
            import laya

            self.agent = laya.load(str(path), device=device, subfolder=subfolder or None)
        else:
            raise ValueError("engine phải là 'laya' hoặc 'native'")
        self.engine = engine
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