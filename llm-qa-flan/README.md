# FLAN-T5 QA: Fine-tuned SQuAD → FastAPI + Streamlit

This mini-project fine-tunes **FLAN-T5** on **SQuAD v1.1** to answer questions from a context passage. 
It then exposes a **FastAPI** endpoint and a simple **Streamlit** UI. Ready for **Google Cloud Run**.

## 💡 Why this helps (Generative AI Specialist)
- Demonstrates **LLM fine-tuning** (FLAN-T5-small/base) on a standard QA dataset.
- Shows **serving** with FastAPI + **lightweight UI** with Streamlit.
- Includes a path to **containerize & deploy on Cloud Run**.
- Easy to extend into **RAG** (use `sentence-transformers` + FAISS).

---

## 📦 Project Structure
```
llm-qa-flan/
├── training/
│   └── fine_tune_flan_t5_squad.py
├── app/
│   ├── inference.py
│   └── main.py
├── ui/
│   └── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quickstart (Local)

### 0) Create and activate venv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
```

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Fine-tune (default: flan-t5-small)
This will download SQuAD & the model (needs internet). Adjust args as needed.
```bash
python training/fine_tune_flan_t5_squad.py   --model_name google/flan-t5-small   --output_dir outputs/flan-t5-small-squad   --num_train_epochs 1   --per_device_train_batch_size 8   --per_device_eval_batch_size 8   --lr 3e-4   --seed 42
```

> Tip: Start with 1 epoch to finish quickly; then bump to 2–3 for better scores.

### 3) Serve the model (FastAPI)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   --env-file .env
```
Create a `.env` (optional) for `MODEL_DIR`:
```
MODEL_DIR=outputs/flan-t5-small-squad
```

### 4) Test API
```bash
curl -X POST http://localhost:8000/qa   -H "Content-Type: application/json"   -d '{"context":"Alan Turing was a pioneering computer scientist.","question":"Who was Alan Turing?"}'
```

### 5) Launch the UI (Streamlit)
```bash
streamlit run ui/streamlit_app.py
```
Enter a context paragraph + question; get an answer.

---

## 🐳 Docker (Cloud Run Ready)

### Build
```bash
docker build -t flan-qa:latest .
```

### Run
```bash
docker run -p 8080:8080 -e MODEL_DIR=/app/outputs/flan-t5-small-squad flan-qa:latest
```
Container listens on **8080** (Cloud Run default).

### Deploy to Google Cloud Run (high level)
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud run deploy flan-qa   --source .   --region asia-south1   --allow-unauthenticated   --port 8080
```
> First deploy can be done without a model; then **mount a Cloud Storage** or **bake the model** by copying `outputs/...` into the image (see tips in `Dockerfile`).

---

## 🧩 RAG Upgrade (Optional, 1–2 hours)
- Build embeddings with `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
- Index documents in FAISS; retrieve top-k chunks per query.
- Construct prompt: `context_chunks + user_question` → feed to FLAN-T5 for grounded answers.

---

## 🧪 Evaluation (Optional)
Use `squad_v1` validation for EM/F1 metrics. Script prints eval loss and can be extended.

---

## 📚 Notes
- Start with `flan-t5-small` for speed. Upgrade to `flan-t5-base` once everything works.
- If RAM is low, set smaller batch sizes & enable gradient accumulation.
- For reproducibility, pin versions in `requirements.txt`.
