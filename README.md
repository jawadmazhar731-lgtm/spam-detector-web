# SMS Spam Detector — Web Version

ML Mids Project (Zeeshan Ali, BSCS, Iqra University). A web version of the
SMS spam detector: type a message, get **SPAM** or **HAM (not spam)**.

## What the backend is

The backend is **Jawad's original code, unchanged** — `spam_detector.py` is
a copy of `spam_detector_kaggle_0_4kvq.py` with exactly ONE change: the
interactive `while True: input()` loop at the end is wrapped in
`if __name__ == "__main__":` so the module can be imported by a web app
without blocking on terminal input. The training pipeline, `clean_text()`
(leet-speak fix, e.g. FR33 → FREE), the extra local spam/ham examples
(parcel scam, SIM-block scam, bank scam, Roman Urdu), the TF-IDF bigrams +
ComplementNB model, and `predict_message()` are exactly as written.

The app trains on the real SMS Spam Collection dataset (`spam.csv`,
5,574 messages, from UCI) at startup, then serves predictions from the
trained model.

## Run locally

```bash
cd spam-detector-web
pip install -r requirements.txt
python3 app.py
```

Open http://localhost:5000 — type a message, press **Check Message**.

API: `POST /api/predict` with JSON `{"message": "..."}` returns
`{"result": "SPAM" | "HAM", "message": "..."}`.

## Deploy

No deployment has been done from here — push this folder to GitHub first,
then use any option below (all free tiers).

### Option 1 — Render (Flask app, recommended)

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → New → Web Service → pick the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Deploy. Render gives a public `https://<name>.onrender.com` link.
   (Free tier sleeps after inactivity; first load takes ~30 seconds.)

### Option 2 — Hugging Face Spaces (Streamlit)

1. Push this folder to a GitHub repo.
2. On [huggingface.co/spaces](https://huggingface.co/spaces) → Create Space →
   SDK: **Streamlit**, then upload these files (or connect the GitHub repo).
3. The Space runs `streamlit_app.py`, which imports the same backend
   (`spam_detector.py`) and the same `spam.csv`.
4. The Space URL is public: `https://huggingface.co/spaces/<you>/<space>`.

### Option 3 — Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io) → New app → pick the
   repo, branch, and main file `streamlit_app.py` → Deploy.
3. Streamlit gives a public `https://<app>.streamlit.app` link.
   (Requires a GitHub account; free.)

## Files

- `spam_detector.py` — the original detector code (one `__main__` guard added)
- `app.py` — Flask web app (imports `spam_detector`, serves page + API)
- `templates/index.html` — the web page
- `streamlit_app.py` — Streamlit frontend for Streamlit Cloud / HF Spaces
- `spam.csv` — SMS Spam Collection dataset, 5,574 messages (UCI)
- `spam_model.pkl`, `tfidf_vectorizer.pkl` — generated on first run
- `requirements.txt` — flask, pandas, scikit-learn, joblib, gunicorn
