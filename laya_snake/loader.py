"""Tải weights Laya từ Hugging Face rồi nạp vào kiến trúc trong common.py.

File này thay cho `laya.load(...)` của gói `laya`: nó tự làm đủ 4 việc
  1. tải checkpoint từ Hugging Face về một thư mục thường        -> download_checkpoint()
  2. đọc cấu hình, tokenizer và dựng DecisionModel (common.py)    -> load_agent()
  3. nạp weights từ model.safetensors vào model (strict=True)     -> load_agent()
  4. trả về một LayaAgent có hàm predict(state, questions)        -> LayaAgent

Cách làm bám theo `Agent.__init__` và `Agent.predict_batch` trong agent.py gốc, chỉ giữ phần
cần cho game Snake (câu hỏi `choice` / `noul` / `score`, chạy CPU hoặc CUDA).

Chạy thử từ dòng lệnh (đứng ở thư mục chứa `laya_snake`):
  python -m laya_snake.loader --download     # tải weights (~0.65 GB)
  python -m laya_snake.loader --check        # nạp weights, chạy thử 1 nước đi
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

HF_REPO = "convaiinnovations/laya"
HF_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"  # revision đã dùng khi thử
# Thư mục mô hình nằm cạnh thư mục laya_snake, tính từ vị trí file này.
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models" / "laya"
DEFAULT_SUBFOLDER = "multilingual"
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
QUESTION_TYPES = ("choice", "score", "noul")


# ---------------------------------------------------------------- 1. tải weights
def download_checkpoint(directory=DEFAULT_DIR, subfolder=DEFAULT_SUBFOLDER, revision=HF_REVISION):
    """Tải một checkpoint Laya từ Hugging Face về thư mục thường (không dùng symlink)."""
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    return snapshot_download(
        HF_REPO,
        revision=revision,
        local_dir=str(directory),
        allow_patterns=[prefix + name for name in CHECKPOINT_FILES],
        token=os.environ.get("HF_TOKEN") or None,
    )


def local_checkpoint(value=None, subfolder=DEFAULT_SUBFOLDER):
    """Tìm thư mục mô hình trên máy. Khi chơi game thì không dùng mạng."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    path = Path(value).expanduser() if value else DEFAULT_DIR
    if not (path / (subfolder or "") / "model.safetensors").is_file():
        raise FileNotFoundError(f"Không thấy mô hình trong: {path / (subfolder or '')}")
    return path


# ---------------------------------------------------------------- 2. đọc cấu hình, tokenizer
def fix_tokenizer_config(model_dir):
    """Sửa tokenizer_config.json để đọc được trên mọi bản `transformers` (như agent.py gốc)."""
    cfg_file = Path(model_dir) / "tokenizer" / "tokenizer_config.json"
    if not cfg_file.is_file():
        return
    try:
        tcfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):  # transformers cần dạng mapping, không phải list
            tcfg["extra_special_tokens"] = {f"extra_{i}": t for i, t in enumerate(extra)}
            changed = True
        if changed:
            cfg_file.write_text(json.dumps(tcfg, indent=2), encoding="utf-8")
    except Exception as error:  # noqa: BLE001
        warnings.warn(f"Không sửa được {cfg_file} ({error}); tokenizer có thể không nạp được.", RuntimeWarning)


def _check_weights(model, cfg, weights):
    """Kiểm tra weights khớp kiến trúc trước khi nạp (như _verify_compatibility trong agent.py)."""
    for key in ("encoder", "head_layers"):
        if key not in cfg:
            raise ValueError(f"rl_agent_config.json thiếu khóa {key!r}: không phải checkpoint Laya hợp lệ")
    for prefix in ("encoder.", "type_emb.", "scorer.", "act_head."):
        if not any(name.startswith(prefix) for name in weights):
            raise ValueError(f"model.safetensors thiếu các tham số '{prefix}...'")
    missing, wrong_shape = [], []
    for name, param in model.named_parameters():
        if name not in weights:
            missing.append(name)
        elif tuple(weights[name].shape) != tuple(param.shape):
            wrong_shape.append(f"{name}: cần {tuple(param.shape)}, file có {tuple(weights[name].shape)}")
    if wrong_shape:
        raise ValueError("Weights không khớp kiến trúc:\n  " + "\n  ".join(wrong_shape[:5]))
    if missing:
        raise ValueError(f"Weights thiếu {len(missing)} tham số, ví dụ: {missing[:3]}")


def _pick_device(device):
    import torch

    if device is None or str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("Máy không có CUDA, chuyển sang CPU.", RuntimeWarning)
        return torch.device("cpu")
    return target


# ---------------------------------------------------------------- 3. nạp weights vào model
def load_agent(path=None, subfolder=DEFAULT_SUBFOLDER, device="cpu"):
    """Dựng DecisionModel (common.py), nạp weights từ model.safetensors, trả về LayaAgent."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from . import common

    model_dir = Path(local_checkpoint(path, subfolder)) / (subfolder or "")
    fix_tokenizer_config(model_dir)

    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    if common.uses_parallel_layout(cfg):
        raise NotImplementedError("Checkpoint dùng option_layout='parallel', bản gọn này chưa hỗ trợ.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))

    # pretrained=False: chỉ dựng khung từ encoder/config.json, KHÔNG khởi tạo tham số
    # (weights thật sắp được nạp ngay bên dưới).
    encoder_dir = model_dir / "encoder"
    model = common.build_model(cfg, encoder_dir=str(encoder_dir) if encoder_dir.exists() else None,
                               pretrained=False)

    weights = load_file(str(model_dir / "model.safetensors"))
    _check_weights(model, cfg, weights)
    model.load_state_dict(weights, strict=True)
    del weights

    try:  # ModernBERT mặc định tự torch.compile; tắt đi cho chắc (như agent.py)
        model.encoder.config.reference_compile = False
    except Exception:  # noqa: BLE001
        pass

    target = _pick_device(device)
    dtype, amp = torch.float32, False
    if target.type == "cuda":
        amp = True
        major = torch.cuda.get_device_capability(target)[0]
        # GPU đời cũ (compute capability < 8, ví dụ MX330) không có bf16 -> dùng fp16
        dtype = torch.float16 if major < 8 else common.amp_dtype(cfg.get("amp_dtype", "fp16"))
    try:
        model.to(target).eval()
    except RuntimeError as error:  # hết VRAM -> quay về CPU
        if target.type == "cpu":
            raise
        warnings.warn(f"Không đặt được model lên {target} ({error}); chạy bằng CPU.", RuntimeWarning)
        target, dtype, amp = torch.device("cpu"), torch.float32, False
        model.to(target).eval()

    return LayaAgent(model, tokenizer, cfg, target, dtype, amp)


# ---------------------------------------------------------------- 4. hỏi mô hình
def _to_internal(qid, qdef):
    """Đổi định nghĩa câu hỏi của người dùng sang dạng nội bộ (như Agent._to_internal)."""
    if not isinstance(qdef, dict):
        raise ValueError(f"câu hỏi {qid!r}: định nghĩa phải là dict")
    qtype = qdef.get("type")
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"câu hỏi {qid!r}: type {qtype!r} không hợp lệ, dùng một trong {QUESTION_TYPES}")
    ins = qdef.get("instructions")
    if ins is None or (isinstance(ins, str) and not ins.strip()):
        raise ValueError(f"câu hỏi {qid!r}: thiếu 'instructions'")
    crit = qdef.get("criteria")
    if qtype == "choice":
        if isinstance(crit, list):
            crit = {c: None for c in crit}
        if not isinstance(crit, dict) or not crit:
            raise ValueError(f"câu hỏi {qid!r}: choice cần 'criteria' là dict nhãn -> mô tả")
    elif qtype == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    question = {"t": qtype, "ins": ins, "crit": crit}
    if "labels" in qdef:
        question["labels"] = qdef["labels"]
    if "option_order" in qdef:
        question["option_order"] = [int(i) for i in qdef["option_order"]]
    return question


class LayaAgent:
    """Bản gọn của laya.Agent: chỉ có predict() cho 1 trạng thái và nhiều câu hỏi."""

    def __init__(self, model, tokenizer, cfg, device, dtype, amp):
        from . import common

        self.model, self.tok, self.cfg = model, tokenizer, cfg
        self.device, self.dtype, self.amp = device, dtype, amp
        raw = cfg.get("temperature", [1.0, 1.0, 1.0])
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError(f"'temperature' trong config phải là list 3 số, nhận được {raw!r}")
        # Chỉ dùng giá trị đã kẹp vào [0.5, 5.0], như agent.py gốc.
        self.temperature = [common.clamp_temperature(t) for t in raw]
        self.temperature_by_options = {
            k: common.clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()
        }

    def predict(self, state, questions):
        """Trả về {"model", "answers": {qid: ...}, "usage": {...}} giống laya.Agent.predict."""
        import numpy as np
        import torch

        from . import common

        qids = list(questions)
        usage = {"input_tokens": 0, "output_tokens": 0}
        if not qids:
            return {"model": "laya-rl-agent", "answers": {}, "usage": usage}
        internal = {qid: _to_internal(qid, questions[qid]) for qid in qids}

        # --- (a) chữ -> token: [CLS] câu hỏi [SEP] [MASK] lựa chọn ... [SEP] trạng thái [SEP]
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        state_ids = common.encode_text(
            self.tok, common.serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        items = []
        for qid in qids:
            q = internal[qid]
            ids, markers = common.build_sequence(
                self.tok, state, q, max_len, head_max_len,
                option_order=q.get("option_order"), state_ids=state_ids,
            )
            if len(markers) != len(common.render_options(q)):
                raise ValueError(f"câu hỏi {qid!r}: không đủ chỗ cho các lựa chọn trong max_len={max_len}")
            items.append({"ids": ids, "markers": markers, "qtype": common.QTYPES[q["t"]]})
        batch = common.collate_items([items], self.tok.pad_token_id)
        usage["input_tokens"] = int(batch["attention_mask"].sum())

        # --- (b) chạy mô hình: mỗi câu hỏi là 1 hàng trong batch
        use = self.device
        with torch.no_grad():
            if self.amp:
                scope = torch.autocast(device_type=use.type, dtype=self.dtype)
            else:
                from contextlib import nullcontext
                scope = nullcontext()
            with scope:
                logits, act = self.model(
                    batch["input_ids"].to(use), batch["attention_mask"].to(use),
                    batch["marker_pos"].to(use), batch["marker_mask"].to(use),
                    batch["qtype"].to(use),
                )
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()

        # --- (c) logits -> xác suất (chia nhiệt độ rồi softmax) -> câu trả lời
        answers = {}
        for row, qid in enumerate(qids):
            q = internal[qid]
            k = len(items[row]["markers"])
            qtype = common.QTYPES[q["t"]]
            scale = self.temperature_by_options.get(common.temp_bucket(qtype, k), self.temperature[qtype])
            z = logits[row, :k] / scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            p = common.unpermute_probs(p, q.get("option_order"))
            answer_conf = round(common.answer_confidence(p, k), 4)
            extra = {"action": {"act_probability": round(float(act[row, 0]), 4)}}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {key: round(float(v), 4) for key, v in zip(keys, p)},
                    "confidence": round(common.confidence_from_probs(p, k), 4),
                    "answer_confidence": answer_conf, **extra,
                }
            elif q["t"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": round(float((np.arange(k) * p).sum()), 4),
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(common.confidence_from_probs(p, k), 4),
                    "answer_confidence": answer_conf, **extra,
                }
            else:  # noul: p[1] là xác suất "true"
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "answer_confidence": answer_conf, **extra,
                }
        return {"model": "laya-rl-agent", "answers": answers, "usage": usage}


# ---------------------------------------------------------------- dòng lệnh: tải và tự kiểm tra
def _check(args):
    """Nạp weights bằng loader này, chạy thử 1 nước đi; so với gói `laya` nếu máy có cài."""
    from .game import SnakeGame
    from .policy import LayaPolicy

    game = SnakeGame()
    started = time.perf_counter()
    native = LayaPolicy(args.model, subfolder=args.subfolder, device=args.device, engine="native")
    print(f"Nạp weights xong sau {time.perf_counter() - started:.1f} giây, chạy bằng {native.device_name}")
    mine = native.decide(game)
    print("Xác suất 4 hướng:", {k: round(v, 3) for k, v in mine.probabilities.items()})
    print(f"AI chọn {mine.proposed}, suy nghĩ {mine.inference_ms:.0f} ms")
    try:
        import laya  # noqa: F401
    except ImportError:
        print("Máy chưa cài gói `laya` nên bỏ qua bước so sánh.")
        return 0
    reference = LayaPolicy(args.model, subfolder=args.subfolder, device=args.device, engine="laya")
    theirs = reference.decide(game)
    diff = max(abs(mine.probabilities[d] - theirs.probabilities[d]) for d in mine.probabilities)
    verdict = "KHỚP" if diff < 1e-3 else "LỆCH, hãy báo lại"
    print(f"Chênh lệch lớn nhất so với gói laya: {diff:.5f} -> {verdict}")
    return 0 if diff < 1e-3 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m laya_snake.loader", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="Tải weights từ Hugging Face")
    parser.add_argument("--check", action="store_true", help="Nạp weights và chạy thử 1 nước đi")
    parser.add_argument("--model", help=f"Thư mục chứa mô hình (mặc định: {DEFAULT_DIR})")
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.download:
        folder = args.model or DEFAULT_DIR
        print(f"Đang tải weights về {folder} (~0.65 GB), xin chờ...")
        download_checkpoint(folder, args.subfolder)
        print("Xong!")
        return 0
    if args.check:
        return _check(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())