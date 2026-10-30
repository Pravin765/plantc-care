import json, os, threading
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

interp = Interpreter(model_path=os.path.join(MODEL_DIR, "plantulus_doctor_fp16.tflite"))
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
H, W = int(inp["shape"][1]), int(inp["shape"][2])  # read from the model
lock = threading.Lock()  # the interpreter is not thread-safe


def load_labels():
    classes = json.load(open(os.path.join(MODEL_DIR, "classes.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(MODEL_DIR, "class_info.json"), encoding="utf-8"))
    if isinstance(classes, dict):
        classes = [classes[k] for k in sorted(classes, key=lambda k: int(k))]
    result = []
    for c in classes:
        c = str(c)
        d = {}
        if isinstance(info, dict):
            d = info.get(c, {}) or {}
        elif isinstance(info, list):
            d = next((x for x in info if isinstance(x, dict) and c in map(str, x.values())), {})
        parts = c.split("___") if "___" in c else [c, c]
        plant = d.get("plant") or parts[0]
        disease = d.get("disease") or parts[-1]
        healthy = d.get("is_healthy", d.get("healthy", "healthy" in c.lower()))
        result.append({
            "label": c,
            "plant": str(plant).replace("_", " "),
            "disease": str(disease).replace("_", " "),
            "healthy": bool(healthy),
        })
    return result


LABELS = load_labels()
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "input": [H, W], "classes": len(LABELS)}


@app.get("/labels")
def labels():
    return LABELS


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB").resize((W, H))
    except Exception:
        raise HTTPException(status_code=400, detail="Bad image")
    x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)  # 0-255, no /255
    x = x.astype(inp["dtype"])
    with lock:
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
    if abs(float(y.sum()) - 1.0) > 0.05 or float(y.min()) < 0:  # logits -> softmax
        e = np.exp(y - y.max())
        y = e / e.sum()
    return {"scores": [round(float(v), 4) for v in y]}


app.mount("/", StaticFiles(directory=os.path.join(BASE, "static"), html=True), name="static")
