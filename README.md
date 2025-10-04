# FLAN-T5 QA — Fine-tuned on SQuAD • FastAPI • Streamlit • Docker

Production-style QA system:
- Fine-tunes `google/flan-t5-small` on **SQuAD** (subset/full).
- **API:** FastAPI `/qa` with Swagger.
- **UI:** Streamlit front-end.
- **Deploy:** Docker (Cloud Run ready).
- Switch between **base HF model** and **fine-tuned checkpoint** via `MODEL_DIR`.

![Swagger](docs/swagger.png)
![Streamlit](docs/streamlit.png)

## Quickstart

```bash
# 1) deps
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) train (quick subset)
python training/fine_tune_flan_t5_squad.py \
  --model_name google/flan-t5-small \
  --output_dir outputs/flan-t5-small-squad \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --lr 3e-4 --seed 42 \
  --max_train_samples 200 --max_eval_samples 200

# 3) serve (base or fine-tuned)
MODEL_DIR=google/flan-t5-small uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# or
MODEL_DIR=outputs/flan-t5-small-squad uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4) UI
streamlit run ui/streamlit_app.py  # set API URL to http://localhost:8000/qa
