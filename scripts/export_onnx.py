from gliner import GLiNER

from app.recognizers import GLINER_MODEL_NAME, GLINER_ONNX_DIR, GLINER_ONNX_FILE


def main() -> None:
    model = GLiNER.from_pretrained(GLINER_MODEL_NAME, map_location="cpu")
    paths = model.export_to_onnx(GLINER_ONNX_DIR, onnx_filename=GLINER_ONNX_FILE)
    print(f"ONNX salvo em {paths['onnx_path']}")


if __name__ == "__main__":
    main()
