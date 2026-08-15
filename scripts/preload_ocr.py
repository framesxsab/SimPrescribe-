from paddleocr import PaddleOCR

for kwargs in (
    {"lang": "en", "device": "cpu", "use_textline_orientation": True, "show_log": False},
    {"lang": "en", "device": "cpu", "use_textline_orientation": True},
    {"lang": "en", "use_angle_cls": True, "use_gpu": False, "show_log": False},
    {"lang": "en", "use_angle_cls": True, "use_gpu": False},
):
    try:
        PaddleOCR(**kwargs)
        break
    except (TypeError, ValueError):
        pass

print("PaddleOCR models ready")
