# Laya Snake Demo

Demo mô hình **Laya** chơi game Snake, có giao diện đồ họa (`tkinter`), chạy được bằng **CPU**.

## Cài đặt (Windows)

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy

### Laya

```
python -m laya_snake --download   # tải mô hình (~0.65 GB), chỉ cần 1 lần
python -m laya_snake              # mở cửa sổ game
```

#### Hai cách nạp mô hình
 
| Cách | Lệnh | Ghi chú |
|---|---|---|
| `laya` (mặc định) | `python -m laya_snake` | Dùng gói `laya` từ pip |
| `native` | `python -m laya_snake --engine native` | Tự tải weights từ Hugging Face rồi nạp vào kiến trúc trong `common.py` (file `loader.py`), không cần gói `laya` |

### EmbeddingGemma 2
```
python -m pip install -U mediapipe
python -m laya_snake.decision_backends --download-gemma
python -m laya_snake --engine gemma
```

### Jev
```
python -m laya_snake --engine jev
```

### Microsoft-Decision-1
```
python -m laya_snake --engine decision1
```

## Cấu trúc
 
```
laya_snake/
├── __init__.py
├── __main__.py   # điểm vào: python -m laya_snake
├── game.py       # luật chơi + bộ lập kế hoạch an toàn
├── policy.py     # gọi mô hình Laya + khiên an toàn
├── loader.py     # tải weights từ Hugging Face và nạp vào mô hình
├── common.py     # kiến trúc mô hình Laya (DecisionModel, token hóa câu hỏi)
├── decision_backends.py  # Jev (OpenRouter), Decision-1 (Azure), EmbeddingGemma 2
└── gui.py        # giao diện tkinter
```