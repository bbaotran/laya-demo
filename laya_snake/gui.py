"""Giao diện đồ họa (tkinter) cho Laya Snake. Mặc định chạy bằng CPU.

  python -m laya_snake --download   # tải mô hình (~0.65 GB), chỉ cần 1 lần
  python -m laya_snake              # mở cửa sổ game

Phím: W A S D / mũi tên = "phá" rắn   Space = tạm dừng   R = ván mới   + / - = tốc độ   Q = thoát
"""

import argparse
import threading
import time
import tkinter as tk

from .game import SnakeGame
from .policy import DEFAULT_DIR, DEFAULT_SUBFOLDER, LayaPolicy, download_checkpoint

CELL = 30           # kích thước 1 ô (pixel)
PANEL_W = 340       # bề rộng bảng thông tin bên phải
PUSH_MOVES = 3      # mỗi lần bấm phím = đẩy rắn đi hướng đó 3 bước

BG, FG, MUTED = "#0a0a0a", "#f5f0e6", "#8f8676"
GREEN, ORANGE, GRID, DIM = "#8ef1a4", "#ff6933", "#1c1c1c", "#2a2a2a"
BODY_DARK, BODY_LIGHT = (0x1D, 0x35, 0x24), (0x8E, 0xF1, 0xA4)
FONT = ("Segoe UI", 11)
FONT_BIG = ("Segoe UI", 14, "bold")

KEYS = {"w": "UP", "up": "UP", "s": "DOWN", "down": "DOWN",
        "a": "LEFT", "left": "LEFT", "d": "RIGHT", "right": "RIGHT"}


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.fps = args.fps
        self.paused = False
        self.stop = False
        self.restart_flag = False
        self.push = None            # [hướng, số bước còn lại]
        self.latest = None          # dữ liệu mới nhất để vẽ
        self.message = "Đang tải mô hình...\n(lần đầu có thể mất 10-60 giây, xin chờ)"
        self.flash_text, self.flash_until = "", 0.0
        self.hardware = "CPU"

        self.board_w = args.width * CELL
        self.board_h = args.height * CELL
        self.height = max(self.board_h, 470)

        root.title("Laya Snake - demo")
        root.configure(bg=BG)
        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x")
        buttons = (
            ("Tạm dừng / Tiếp tục (Space)", self.toggle_pause),
            ("Ván mới (R)", self.restart),
            ("Chậm hơn (-)", lambda: self.change_fps(-1)),
            ("Nhanh hơn (+)", lambda: self.change_fps(+1)),
            ("Thoát (Q)", self.quit),
        )
        for label, command in buttons:
            tk.Button(bar, text=label, command=command, takefocus=0).pack(side="left", padx=4, pady=4)

        self.canvas = tk.Canvas(
            root, width=self.board_w + PANEL_W, height=self.height, bg=BG, highlightthickness=0
        )
        self.canvas.pack()
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.quit)

        threading.Thread(target=self.worker, daemon=True).start()
        self.refresh()

    # ---------- điều khiển ----------
    def toggle_pause(self):
        self.paused = not self.paused

    def restart(self):
        self.restart_flag = True

    def change_fps(self, delta):
        self.fps = max(1, min(60, self.fps + delta))

    def quit(self):
        self.stop = True
        self.root.destroy()

    def on_key(self, event):
        key = event.keysym.lower()
        if key in KEYS:
            self.push = [KEYS[key], PUSH_MOVES]
        elif key == "space":
            self.toggle_pause()
        elif key == "r":
            self.restart()
        elif key in ("plus", "equal"):
            self.change_fps(+1)
        elif key == "minus":
            self.change_fps(-1)
        elif key == "q":
            self.quit()

    def flash(self, message):
        self.flash_text, self.flash_until = message, time.perf_counter() + 1.2

    # ---------- luồng chạy game + AI (chạy nền để cửa sổ không bị đơ) ----------
    def worker(self):
        a = self.args
        try:
            policy = LayaPolicy(
                a.model, subfolder=a.subfolder, guarded=not a.unassisted, device=a.device,
                engine=a.engine,
            )
        except FileNotFoundError as error:
            self.message = (
                f"CHƯA CÓ MÔ HÌNH.\nHãy chạy:  python -m laya_snake --download\n\n{error}"
            )
            return
        except Exception as error:  # noqa: BLE001
            self.message = f"Lỗi khi tải mô hình:\n{error!r}"
            return
        self.hardware = policy.device_name

        round_no, best, interventions = 1, 0, 0
        game = SnakeGame(a.width, a.height, a.seed, a.initial_length)
        try:
            while not self.stop:
                if self.restart_flag:
                    self.restart_flag = False
                    round_no += 1
                    game = SnakeGame(a.width, a.height, a.seed + round_no - 1, a.initial_length)
                    self.push = None
                if self.paused:
                    time.sleep(0.05)
                    continue

                started = time.perf_counter()
                decision = policy.decide(game)          # bước chậm nhất: AI suy nghĩ
                pushed = False
                push = self.push
                if push:
                    direction, left = push
                    if game.legal_reason(direction) == "legal":
                        decision.executed, decision.intervened = direction, False
                        pushed = True
                        self.flash(f"BẠN ĐẨY {direction}")
                    else:
                        self.flash("NÉ ĐƯỢC! (đẩy sẽ chết)")
                    if self.push is push:
                        self.push = [direction, left - 1] if left > 1 else None

                interventions += decision.intervened
                best = max(best, game.score)
                self.latest = {
                    "board": game.snapshot(),
                    "decision": decision.to_dict(),
                    "pushed": pushed,
                    "round": round_no,
                    "best": best,
                    "interventions": interventions,
                }
                remaining = 1 / self.fps - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

                game.step(decision.executed)
                best = max(best, game.score)
                if not game.alive or game.won:
                    self.latest = {**self.latest, "board": game.snapshot(), "decision": {}, "best": best}
                    time.sleep(2)
                    round_no += 1
                    self.push = None
                    game = SnakeGame(a.width, a.height, a.seed + round_no - 1, a.initial_length)
        except Exception as error:  # noqa: BLE001
            self.latest = None
            self.message = f"Lỗi trong lúc chạy:\n{error!r}"

    # ---------- vẽ ----------
    def refresh(self):
        if self.stop:
            return
        c = self.canvas
        c.delete("all")
        data = self.latest
        if data is None:
            c.create_text(
                (self.board_w + PANEL_W) // 2, self.height // 2, text=self.message,
                fill=FG, font=FONT_BIG, justify="center",
            )
        else:
            self.draw_board(data)
            self.draw_panel(data)
        self.root.after(80, self.refresh)

    def draw_board(self, data):
        c, g = self.canvas, data["board"]
        w, h = self.board_w, self.board_h
        c.create_rectangle(0, 0, w, h, fill="#101010", outline=DIM)
        for x in range(1, g["width"]):
            c.create_line(x * CELL, 0, x * CELL, h, fill=GRID)
        for y in range(1, g["height"]):
            c.create_line(0, y * CELL, w, y * CELL, fill=GRID)
        body = g["body"]
        for index, (x, y) in reversed(list(enumerate(body))):
            fraction = 1 - index / max(1, len(body))
            color = FG if index == 0 else "#" + "".join(
                f"{int(d + (l - d) * fraction):02x}" for d, l in zip(BODY_DARK, BODY_LIGHT)
            )
            c.create_rectangle(
                x * CELL + 2, y * CELL + 2, (x + 1) * CELL - 2, (y + 1) * CELL - 2,
                fill=color, outline="",
            )
        if g["food"]:
            x, y = g["food"]
            c.create_oval(
                x * CELL + 6, y * CELL + 6, (x + 1) * CELL - 6, (y + 1) * CELL - 6,
                fill=ORANGE, outline="",
            )
        overlay = None
        if self.paused:
            overlay = "TẠM DỪNG"
        elif g["won"]:
            overlay = "THẮNG! (kín bàn)"
        elif not g["alive"]:
            overlay = f"THUA ({g['death_reason']}) - ván mới sắp bắt đầu"
        if overlay:
            c.create_rectangle(0, h // 2 - 26, w, h // 2 + 26, fill=BG, outline=DIM)
            c.create_text(w // 2, h // 2, text=overlay, fill=ORANGE, font=FONT_BIG)

    def draw_panel(self, data):
        c = self.canvas
        g, dec = data["board"], data["decision"]
        x0 = self.board_w + 24
        y = [12]

        def line(text, color=FG, font=FONT, gap=24):
            c.create_text(x0, y[0], text=text, anchor="nw", fill=color, font=font)
            y[0] += gap

        def bar(label, value, color):
            value = max(0.0, min(1.0, value))
            c.create_text(x0, y[0], text=label, anchor="nw", fill=color, font=FONT)
            c.create_rectangle(x0 + 100, y[0] + 6, x0 + 220, y[0] + 16, fill=DIM, outline="")
            c.create_rectangle(x0 + 100, y[0] + 6, x0 + 100 + int(120 * value), y[0] + 16, fill=color, outline="")
            c.create_text(x0 + 232, y[0], text=f"{value:.2f}", anchor="nw", fill=color, font=FONT)
            y[0] += 22

        line("LAYA đang chơi rắn", GREEN, FONT_BIG, 30)
        line(f"Điểm: {g['score']}    Độ dài: {g['length']}", FG)
        line(f"Cao nhất: {data['best']}    Ván: {data['round']}", MUTED, gap=30)

        line("Xác suất AI chọn hướng", MUTED)
        probabilities = dec.get("probabilities", {})
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            chosen = direction == dec.get("proposed")
            bar(f"{'›' if chosen else ' '} {direction}", probabilities.get(direction, 0), GREEN if chosen else MUTED)

        y[0] += 6
        tag = "  [BẠN ĐẨY]" if data["pushed"] else "  [KHIÊN AN TOÀN]" if dec.get("intervened") else ""
        line(f"Thực hiện: {dec.get('executed', '-')}{tag}", ORANGE if tag else GREEN, gap=28)

        bar("Nguy cơ ngõ cụt", dec.get("dead_end_risk", 0), ORANGE)
        bar("Tới được mồi", dec.get("food_reachable", 0), "#c4b8a0")
        y[0] += 6
        line(f"Thời gian AI nghĩ: {dec.get('inference_ms', 0):.0f} ms", FG)
        line(f"Chạy bằng: {self.hardware}", MUTED)
        line(f"Tốc độ tối đa: {self.fps}/giây   Khiên can thiệp: {data['interventions']}", MUTED)
        mode = g.get("mode")
        if mode:
            line("Tuyến: " + ("CỐ ĐỊNH" if mode == "cycle" else "TỰ DO (đang tự sửa)"),
                 GREEN if mode == "cycle" else ORANGE)
        if time.perf_counter() < self.flash_until:
            line(self.flash_text, ORANGE, FONT_BIG)

        c.create_text(
            x0, self.height - 10, anchor="sw", fill=MUTED, font=("Segoe UI", 10),
            text="WASD/mũi tên: phá rắn   Space: dừng\nR: ván mới   +/-: tốc độ   Q: thoát",
        )


def main():
    parser = argparse.ArgumentParser(prog="python -m laya_snake", description="Laya Snake (CPU)")
    parser.add_argument("--download", action="store_true", help="Tải mô hình về máy rồi thoát")
    parser.add_argument("--model", help=f"Thư mục chứa mô hình (mặc định: {DEFAULT_DIR})")
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER)
    parser.add_argument("--device", default="cpu", help="cpu (mặc định) hoặc cuda")
    parser.add_argument("--engine", default="laya", choices=("laya", "native"),
                        help="laya: dùng gói laya (mặc định); native: tự nạp weights bằng loader.py")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--fps", type=int, default=4, help="Số quyết định tối đa mỗi giây")
    parser.add_argument("--unassisted", action="store_true", help="Tắt khiên an toàn (AI dễ chết hơn)")
    args = parser.parse_args()

    if args.download:
        folder = args.model or DEFAULT_DIR
        print(f"Đang tải mô hình về {folder} (~0.65 GB), xin chờ...")
        download_checkpoint(folder, args.subfolder)
        print("Xong! Giờ chạy:  python -m laya_snake")
        return
    try:
        SnakeGame(args.width, args.height, args.seed, args.initial_length)
    except ValueError as error:
        parser.error(str(error))

    root = tk.Tk()
    App(root, args)
    root.mainloop()