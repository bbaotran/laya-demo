"""Thêm các "bộ quyết định" khác để chọn hướng đi cho con rắn.

  jev        Jev của TypeSafe, gọi qua OpenRouter           (cần OPENROUTER_API_KEY trong .env)
  decision1  Microsoft-Decision-1, gọi qua Azure Foundry    (cần AZURE_ENDPOINT, DEPLOYMENT_NAME, AZURE_API_KEY)
  gemma      EmbeddingGemma 2 270M, chạy trên máy bằng MediaPipe Decision Maker

Cả ba trả về cùng dạng với Laya:
  {"answers": {"move": {"probabilities": {...}}, "risk": {"noul": x}, "food": {"noul": y}}}.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # thư mục chứa laya_snake, models, .env

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
AZURE_PATH = "/providers/microsoft/v1/systemone"

GEMMA_REPO = "litert-community/embeddinggemma-2-text-270m-litert-lm"
GEMMA_URL = "https://storage.googleapis.com/mediapipe-models/decision_maker/embeddinggemma-2-270m-it.litertlm"
GEMMA_DIR = ROOT / "models" / "embeddinggemma2"

# Mỗi biến có thể có vài tên; lấy tên đầu tiên có giá trị trong .env hoặc biến môi trường.
ENV_NAMES = {
    "openrouter_key": ("OPENROUTER_API_KEY", "OPENROUTER_KEY"),
    "jev_model": ("JEV_MODEL",),
    "jev_url": ("JEV_API_URL",),  # chỉ dùng khi muốn đổi địa chỉ API (ví dụ để thử nghiệm)
    "azure_key": ("AZURE_API_KEY", "AZURE_DECISION_API_KEY", "AZURE_FOUNDRY_API_KEY", "AZURE_OPENAI_API_KEY"),
    "azure_endpoint": ("AZURE_ENDPOINT", "AZURE_DECISION_ENDPOINT", "AZURE_FOUNDRY_ENDPOINT", "AZURE_OPENAI_ENDPOINT"),
    "azure_deployment": ("DEPLOYMENT_NAME", "AZURE_DEPLOYMENT_NAME", "AZURE_DECISION_DEPLOYMENT"),
    "hf_token": ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
}


class BackendError(RuntimeError):
    """Lỗi cấu hình hoặc lỗi gọi API, có thông điệp tiếng Việt để hiện thẳng cho người dùng."""


# ---------------------------------------------------------------- đọc .env
def load_env(path=None):
    """Đọc file .env vào os.environ. Biến đã có sẵn trong hệ thống được ưu tiên. Trả về các file đã đọc."""
    files = [Path(path)] if path else [ROOT / ".env", Path.cwd() / ".env"]
    loaded = []
    for file in files:
        if not file.is_file() or file in loaded:
            continue
        for raw in file.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: bỏ BOM của Notepad
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, value = (part.strip() for part in line.split("=", 1))
            if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if key:
                os.environ.setdefault(key, value)
        loaded.append(file)
    return loaded


def get_env(kind):
    """Giá trị đầu tiên có trong các tên của `kind` (xem ENV_NAMES), hoặc None."""
    for name in ENV_NAMES[kind]:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def mask(secret):
    """Che key khi in ra màn hình: chỉ lộ 4 ký tự đầu và cuối."""
    if not secret:
        return "(chưa có)"
    return f"{secret[:4]}…{secret[-4:]}" if len(secret) > 12 else "***"


# ---------------------------------------------------------------- Jev và Decision-1: API "System One"
class RemoteDecisionAgent:
    """Gửi {model, state, questions} tới một API quyết định rồi đổi kết quả về dạng của Laya."""

    def __init__(self, url, api_key, model, *, auth, label, device_name, noul_criteria, timeout=30):
        self.url, self.model, self.label = url, model, label
        self.device_name, self.noul_criteria, self.timeout = device_name, noul_criteria, timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "laya-snake-demo",
            ("Authorization" if auth == "bearer" else "api-key"): f"Bearer {api_key}" if auth == "bearer" else api_key,
        }

    def predict(self, state, questions):
        reply = self._post({"model": self.model, "state": state, "questions": self._prepare(questions)})
        answers = reply.get("answers") if isinstance(reply, dict) else None
        if not isinstance(answers, dict):
            raise BackendError(f"{self.label}: phản hồi không có mục 'answers': {str(reply)[:300]}")
        return {"model": reply.get("model", self.model), "answers": self._clean(answers, questions),
                "usage": reply.get("usage", {})}

    def _prepare(self, questions):
        prepared = {}
        for qid, q in questions.items():
            item = {"type": q["type"], "instructions": q["instructions"]}
            if q.get("criteria"):
                item["criteria"] = q["criteria"]
            elif q["type"] == "noul" and self.noul_criteria:
                item["criteria"] = {"true": "Yes", "false": "No"}  # OpenRouter: ví dụ chính thức có kèm mục này
            prepared[qid] = item
        return prepared

    def _clean(self, answers, questions):
        clean = {}
        for qid, q in questions.items():
            answer = answers.get(qid)
            if not isinstance(answer, dict):
                raise BackendError(f"{self.label}: thiếu câu trả lời cho '{qid}': {str(answers)[:300]}")
            try:
                if q["type"] == "choice":
                    probs = {str(k).lower(): float(v) for k, v in answer["probabilities"].items()}
                    clean[qid] = {"type": "choice", "choice": answer.get("choice"),
                                  "probabilities": {k: probs[k.lower()] for k in q["criteria"]}}
                else:
                    clean[qid] = {"type": "noul", "noul": float(answer["noul"])}
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                raise BackendError(f"{self.label}: câu trả lời '{qid}' sai định dạng ({error!r}): {str(answer)[:300]}")
        return clean

    def _post(self, body):
        data = json.dumps(body).encode("utf-8")
        problem = ""
        for attempt in range(3):  # thử lại khi quá tải, mạng chập chờn
            request = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = _error_detail(error)
                if error.code in (429, 500, 502, 503, 504) and attempt < 2:
                    problem = f"HTTP {error.code}"
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise BackendError(self._explain(error.code, detail)) from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                problem = str(getattr(error, "reason", error))
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise BackendError(f"{self.label}: phản hồi không phải JSON hợp lệ ({error})") from error
        raise BackendError(f"{self.label}: không kết nối được sau 3 lần thử ({problem}). Kiểm tra mạng và địa chỉ {self.url}")

    def _explain(self, code, detail):
        hints = {
            401: "API key sai hoặc hết hạn",
            402: "tài khoản hết credit",
            403: "không có quyền dùng model này",
            404: "sai địa chỉ (endpoint) hoặc tên deployment",
            400: "yêu cầu bị từ chối",
            422: "yêu cầu bị từ chối",
            429: "gọi quá nhanh, hãy giảm --fps",
        }
        return f"{self.label}: HTTP {code}, {hints.get(code, 'lỗi từ máy chủ')}. {detail}".strip()


def _error_detail(error):
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    try:
        obj = json.loads(body)
        err = obj.get("error", obj) if isinstance(obj, dict) else obj
        message = err.get("message") if isinstance(err, dict) else err
        return str(message)[:300]
    except ValueError:
        return body[:300]


def make_jev():
    key = get_env("openrouter_key")
    if not key:
        raise BackendError("Chưa có key OpenRouter. Thêm dòng OPENROUTER_API_KEY=... vào file .env "
                           "(kiểm tra bằng: python -m laya_snake.decision_backends --env)")
    return RemoteDecisionAgent(get_env("jev_url") or JEV_URL, key, get_env("jev_model") or JEV_MODEL,
                               auth="bearer", label="OpenRouter/Jev", device_name="OpenRouter · Jev",
                               noul_criteria=True)


def make_decision1():
    endpoint, key, deployment = get_env("azure_endpoint"), get_env("azure_key"), get_env("azure_deployment")
    missing = [name for name, value in (("AZURE_ENDPOINT", endpoint), ("AZURE_API_KEY", key),
                                        ("DEPLOYMENT_NAME", deployment)) if not value]
    if missing:
        raise BackendError("Thiếu trong file .env: " + ", ".join(missing) +
                           " (kiểm tra bằng: python -m laya_snake.decision_backends --env)")
    url = endpoint.rstrip("/")
    if "systemone" not in url:  # cho phép dán nguyên địa chỉ đầy đủ hoặc chỉ địa chỉ gốc của resource
        url += AZURE_PATH
    return RemoteDecisionAgent(url, key, deployment, auth="api-key", label="Azure/Decision-1",
                               device_name="Azure · Decision-1", noul_criteria=False)


# ---------------------------------------------------------------- EmbeddingGemma 2: chạy trên máy
# Mô hình embedding so sánh "câu hỏi + trạng thái" với mô tả từng lựa chọn, nên mô tả cần là câu đầy đủ
# (xem mục Best Practices của MediaPipe Decision Maker). Các nhãn ngắn của policy.py được viết rõ hơn ở đây.
OPTION_TEXT = {
    "Blocked. Collision.": "A blocked move that hits a wall or the snake's own body. Unsafe, must be avoided.",
    "Unsafe. Traps the snake.": "An unsafe move that leads into a dead end and traps the snake.",
    "Safe. Eat food now. Best.": "A safe move that eats the food right now. This is the best move.",
    "Safe. Best route to food.": "A safe move on the best route toward the food.",
    "Safe. Slower route.": "A safe move, but a slower route toward the food.",
}
CONDITION_TEXT = {
    "Is a safe route available?": "The snake has a safe route forward.",
    "Is food reachable through empty cells?": "The food is reachable through the empty cells.",
}


def find_gemma_model(directory=GEMMA_DIR):
    files = sorted(Path(directory).rglob("*.litertlm")) if Path(directory).is_dir() else []
    return files[0] if files else None


def download_gemma(directory=GEMMA_DIR):
    """Tải EmbeddingGemma 2 270M: thử Hugging Face trước (như notebook), rồi tới link công khai của Google."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(GEMMA_REPO, local_dir=str(directory), allow_patterns=["*.litertlm"],
                          token=get_env("hf_token"))
        found = find_gemma_model(directory)
        if found:
            return found
        print("Repo Hugging Face không có file .litertlm, thử link của Google...")
    except Exception as error:  # noqa: BLE001
        print(f"Không tải được từ Hugging Face ({type(error).__name__}).")
        print(f"Nếu repo yêu cầu đồng ý giấy phép Gemma: mở https://huggingface.co/{GEMMA_REPO}, đăng nhập và đồng ý,")
        print("rồi thêm HF_TOKEN=... vào file .env. Đang thử link công khai của Google...")
    target = directory / "embeddinggemma-2-270m-it.litertlm"
    part = target.with_suffix(".part")
    try:
        urllib.request.urlretrieve(GEMMA_URL, part)
    except Exception as error:  # noqa: BLE001
        raise BackendError(f"Không tải được EmbeddingGemma 2 từ cả hai nguồn ({error}). "
                           f"Hãy tải tay file .litertlm từ trang Hugging Face rồi đặt vào {directory}") from error
    part.replace(target)
    return target


class GemmaAgent:
    """EmbeddingGemma 2 qua MediaPipe Decision Maker: ChoiceQuestion cho hướng đi, BooleanQuestion cho risk/food."""

    def __init__(self, model_path):
        try:
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.decision import decision_maker
        except ImportError as error:
            raise BackendError("Chưa dùng được MediaPipe Decision Maker: " + str(error) + ". Thử: "
                               "python -m pip install -U mediapipe (xem README mục EmbeddingGemma)") from error
        self._dm = decision_maker
        options = decision_maker.DecisionMakerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(model_path)), max_num_tokens=512)
        self._maker = decision_maker.DecisionMaker.create_from_options(options)
        self._choice, self._boolean = {}, {}  # dùng lại câu hỏi giống nhau để prewarm một lần
        self.device_name = "EmbeddingGemma 2 · MediaPipe"

    def predict(self, state, questions):
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                criteria = {key: OPTION_TEXT.get(text, text) for key, text in q["criteria"].items()}
                result = self._maker.evaluate_choice(f"{q['instructions']} {state}",
                                                     self._choice_question(q["instructions"], criteria))
                answers[qid] = {"type": "choice", "choice": result.selected_key,
                                "probabilities": self._probabilities(result, list(criteria))}
            elif q["type"] == "noul":
                condition = CONDITION_TEXT.get(q["instructions"], q["instructions"])
                result = self._maker.evaluate_boolean(state, self._boolean_question(condition))
                answers[qid] = {"type": "noul", "noul": float(result.probability_true)}
            else:
                raise BackendError("EmbeddingGemma chỉ hỗ trợ câu hỏi choice và noul")
        return {"model": "embeddinggemma-2-270m-it", "answers": answers, "usage": {}}

    def _choice_question(self, instructions, criteria):
        key = (instructions, tuple(criteria.items()))
        if key not in self._choice:
            question = self._dm.ChoiceQuestion(instructions=instructions, criteria=criteria, normalize_prior=True)
            try:
                self._maker.prewarm_choice(question)  # tính trước embedding của các lựa chọn
            except Exception:  # noqa: BLE001  (chỉ là tối ưu tốc độ)
                pass
            self._choice[key] = question
        return self._choice[key]

    def _boolean_question(self, condition):
        if condition not in self._boolean:
            self._boolean[condition] = self._dm.BooleanQuestion(condition=condition, threshold=0.5,
                                                                normalize_prior=True)
        return self._boolean[condition]

    @staticmethod
    def _probabilities(result, keys):
        probs = result.probabilities
        if isinstance(probs, dict):
            lowered = {str(k).lower(): float(v) for k, v in probs.items()}
            try:
                return {k: lowered[k.lower()] for k in keys}
            except KeyError as error:
                raise BackendError(f"EmbeddingGemma trả về nhãn lạ: {list(probs)}") from error
        values = [float(v) for v in probs]  # dạng danh sách theo đúng thứ tự lựa chọn
        if len(values) != len(keys):
            raise BackendError("EmbeddingGemma trả về số xác suất không khớp số lựa chọn")
        return dict(zip(keys, values))

    def close(self):
        try:
            self._maker.close()
        except Exception:  # noqa: BLE001
            pass


def make_gemma():
    path = find_gemma_model()
    if path is None:
        raise BackendError(f"Chưa có file EmbeddingGemma 2 (.litertlm) trong {GEMMA_DIR}. "
                           "Chạy: python -m laya_snake.decision_backends --download-gemma")
    return GemmaAgent(path)


# ---------------------------------------------------------------- điểm vào cho policy.py
BACKENDS = ("jev", "decision1", "gemma")


def load_backend(engine):
    load_env()
    factory = {"jev": make_jev, "decision1": make_decision1, "gemma": make_gemma}.get(engine)
    if factory is None:
        raise ValueError(f"engine phải là một trong {BACKENDS}")
    return factory()


# ---------------------------------------------------------------- dòng lệnh: kiểm tra .env, tải model, thử 1 nước đi
def _show_env():
    files = load_env()
    print("File .env đã đọc:", ", ".join(str(f) for f in files) if files else
          f"KHÔNG THẤY .env (đặt ở {ROOT / '.env'})")
    secret = {"openrouter_key", "azure_key", "hf_token"}
    optional = {"jev_model", "jev_url", "hf_token"}
    for kind, names in ENV_NAMES.items():
        value = get_env(kind)
        if value:
            print(f"  {names[0]:<20} {mask(value) if kind in secret else value}")
        elif kind in optional:
            print(f"  {names[0]:<20} (chưa có, không bắt buộc)")
        else:
            print(f"  {names[0]:<20} (CHƯA CÓ)   [chấp nhận các tên: {', '.join(names)}]")


def _check(engine):
    from .game import SnakeGame
    from .policy import LayaPolicy

    policy = LayaPolicy(engine=engine)
    decision = policy.decide(SnakeGame())
    print(f"Chạy bằng: {policy.device_name}")
    print("Xác suất 4 hướng:", {k: round(v, 3) for k, v in decision.probabilities.items()})
    print(f"Chọn {decision.proposed}; nguy cơ ngõ cụt {decision.dead_end_risk:.2f}; "
          f"tới được mồi {decision.food_reachable:.2f}; mất {decision.inference_ms:.0f} ms")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m laya_snake.decision_backends", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", action="store_true", help="Kiểm tra file .env (key được che bớt)")
    parser.add_argument("--download-gemma", action="store_true", help="Tải EmbeddingGemma 2 270M")
    parser.add_argument("--check", choices=BACKENDS, help="Thử 1 nước đi bằng model đã chọn")
    args = parser.parse_args(argv)
    try:
        if args.env:
            _show_env()
        elif args.download_gemma:
            load_env()
            print("Đã tải xong:", download_gemma())
        elif args.check:
            _check(args.check)
        else:
            parser.print_help()
    except BackendError as error:
        print("LỖI:", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())