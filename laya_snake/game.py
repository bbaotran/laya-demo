"""Luật chơi Snake (xác định) và bộ lập kế hoạch an toàn.

Dựa trên laya_mlx/snake/game.py (mizorewww/laya-mlx, Apache-2.0).
- Khi thân rắn còn theo đúng thứ tự "chu trình Hamilton" -> chế độ "cycle" (tuyến cố định).
- Khi người chơi làm rắn lệch khỏi chu trình -> chế độ "free": dùng BFS để tìm nước đi an toàn.
"""

import random
from collections import deque
from dataclasses import dataclass

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def hamiltonian_cycle(width, height):
    """Đường đi qua mọi ô đúng 1 lần rồi quay lại điểm đầu."""
    if min(width, height) < 4 or (width % 2 and height % 2):
        raise ValueError("Bàn phải có cạnh >= 4 và ít nhất một cạnh chẵn")
    if height % 2:
        return [(y, x) for x, y in hamiltonian_cycle(height, width)]
    path = [(0, 0)]
    for y in range(height):
        xs = range(1, width) if y % 2 == 0 else range(width - 1, 0, -1)
        path.extend((x, y) for x in xs)
    path.extend((0, y) for y in range(height - 1, 0, -1))
    return path


@dataclass(frozen=True)
class MoveInfo:
    direction: str
    legal: bool     # đi được (không đâm tường / thân / quay đầu)
    safe: bool      # đi được và không dẫn tới ngõ cụt
    advance: int    # điểm xếp hạng nước đi (cao hơn = tốt hơn)
    reason: str
    eats: bool      # nước đi này ăn mồi


class SnakeGame:
    def __init__(self, width=20, height=14, seed=7, initial_length=6):
        self.width, self.height = width, height
        self.cycle = hamiltonian_cycle(width, height)
        self.indices = {cell: i for i, cell in enumerate(self.cycle)}
        self.capacity = width * height
        if not 2 <= initial_length < self.capacity:
            raise ValueError("Độ dài ban đầu phải >= 2 và nhỏ hơn kích thước bàn")
        self.rng = random.Random(seed)
        start = self.indices[(width // 2, height // 2)]
        self.body = deque(self.cycle[(start - i) % self.capacity] for i in range(initial_length))
        self.score = 0
        self.alive, self.won = True, False
        self.death_reason = None
        self.food = self._spawn_food()

    @property
    def head(self):
        return self.body[0]

    def _inside(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _spawn_food(self):
        occupied = set(self.body)
        empty = [cell for cell in self.cycle if cell not in occupied]
        return self.rng.choice(empty) if empty else None

    def target(self, direction):
        dx, dy = VECTORS[direction]
        return self.head[0] + dx, self.head[1] + dy

    def legal_reason(self, direction):
        cell = self.target(direction)
        if not self._inside(cell):
            return "wall"
        if cell == self.body[1]:
            return "reverse"
        occupied = set(self.body)
        if cell != self.food:
            occupied.remove(self.body[-1])  # đuôi di chuyển nếu không ăn mồi
        return "body" if cell in occupied else "legal"

    def cycle_order_valid(self):
        indices = [self.indices[cell] for cell in reversed(self.body)]
        distances = [(b - a) % self.capacity for a, b in zip(indices, indices[1:])]
        return all(d > 0 for d in distances) and sum(distances) < self.capacity

    @property
    def mode(self):
        return "cycle" if self.cycle_order_valid() else "free"

    def moves(self):
        if not self.alive or self.won:
            return []
        if self.mode == "free":
            return self._free_moves()
        head_index = self.indices[self.head]
        tail_distance = (self.indices[self.body[-1]] - head_index) % self.capacity
        food_distance = (self.indices[self.food] - head_index) % self.capacity
        moves = []
        for direction in DIRECTIONS:
            reason = self.legal_reason(direction)
            legal = reason == "legal"
            target = self.target(direction)
            advance = (self.indices.get(target, head_index) - head_index) % self.capacity
            eats = target == self.food
            safe = legal
            if safe and (advance > tail_distance or (advance == tail_distance and eats)):
                safe, reason = False, "would cross the tail"
            if safe and (advance == 0 or advance > food_distance):
                safe, reason = False, "would skip the food on the safe route"
            moves.append(MoveInfo(direction, legal, safe, advance, reason, eats))
        return moves

    def _flood(self, start, blocked, goal=None):
        """BFS trên các ô trống. Trả về (số ô tới được, khoảng cách tới goal hoặc None)."""
        seen = {start: 0}
        queue = deque([start])
        found = 0 if start == goal else None
        while queue:
            cell = queue.popleft()
            for dx, dy in VECTORS.values():
                nxt = cell[0] + dx, cell[1] + dy
                if nxt in seen or not self._inside(nxt) or nxt in blocked:
                    continue
                seen[nxt] = seen[cell] + 1
                if nxt == goal and found is None:
                    found = seen[nxt]
                queue.append(nxt)
        return len(seen), found

    def _free_moves(self):
        """Chế độ tự do: nước đi an toàn nếu đầu rắn vẫn còn đường tới đuôi."""
        moves = []
        body = list(self.body)
        for direction in DIRECTIONS:
            reason = self.legal_reason(direction)
            target = self.target(direction)
            eats = target == self.food
            if reason != "legal":
                moves.append(MoveInfo(direction, False, False, -1, reason, eats))
                continue
            new_body = [target] + (body if eats else body[:-1])
            area, to_tail = self._flood(target, set(new_body[1:-1]), goal=new_body[-1])
            safe = to_tail is not None or len(new_body) <= 2
            to_food = 0 if eats else self._flood(target, set(new_body[1:]), goal=self.food)[1]
            advance = 100000 - to_food * 1000 + min(area, 999) if to_food is not None else area
            moves.append(
                MoveInfo(direction, True, safe, advance, reason if safe else "would trap the snake", eats)
            )
        return moves

    def food_reachability(self):
        """Mồi có tới được qua các ô trống không, và có bao nhiêu ô trống tới được."""
        blocked = set(self.body) - {self.head}
        visited = {self.head}
        queue = deque([self.head])
        while queue:
            x, y = queue.popleft()
            for dx, dy in VECTORS.values():
                cell = x + dx, y + dy
                if self._inside(cell) and cell not in blocked and cell not in visited:
                    visited.add(cell)
                    queue.append(cell)
        return self.food in visited, len(visited)

    def step(self, direction):
        """Đi 1 bước. Trả về True nếu ăn được mồi."""
        if not self.alive or self.won:
            raise RuntimeError("Ván chơi đã kết thúc")
        if direction not in DIRECTIONS:
            raise ValueError(f"Hướng không hợp lệ: {direction}")
        reason = self.legal_reason(direction)
        if reason != "legal":
            self.alive, self.death_reason = False, reason
            return False
        target = self.target(direction)
        self.body.appendleft(target)
        if target == self.food:
            self.score += 1
            if len(self.body) == self.capacity:
                self.won, self.food = True, None
            else:
                self.food = self._spawn_food()
            return True
        self.body.pop()
        return False

    def snapshot(self):
        """Ảnh chụp trạng thái hiện tại, để giao diện vẽ."""
        return {
            "width": self.width,
            "height": self.height,
            "body": [list(cell) for cell in self.body],
            "food": list(self.food) if self.food else None,
            "mode": self.mode if self.alive and not self.won else None,
            "score": self.score,
            "length": len(self.body),
            "alive": self.alive,
            "won": self.won,
            "death_reason": self.death_reason,
        }
