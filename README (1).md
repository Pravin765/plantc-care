# Leaf Disease Scanner (FastAPI + Render)

## 1. Add the model files
Download from https://huggingface.co/KumarjitC/plantulus-doctor and put in `model/`:
- plantulus_doctor_fp16.tflite
- classes.json
- class_info.json

## 2. Run locally
```
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://localhost:8000 (camera works on localhost). Check http://localhost:8000/health
shows the input size and 54 classes.

If `ai-edge-litert` fails to install, try `pip install tflite-runtime` or `pip install tensorflow`.

## 3. Deploy on Render
1. Push this folder (including `model/`) to a GitHub repo.
2. Render > New > Web Service > connect the repo.
3. Runtime: Python. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Environment variable: `PYTHON_VERSION` = `3.11.9`
6. Deploy, then open the https URL (camera needs HTTPS, which Render provides).

Free instances may sleep when idle. Open the site a few minutes before a demo.

## Troubleshooting
- Open `/health` first. If it fails, read the Render logs for the exact error.
- Predictions look wrong: check `classes.json` order matches the model output.
- Camera does not open: allow camera permission, and use https or localhost.
