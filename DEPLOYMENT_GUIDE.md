# 🤖 ML Case Study — 23CSE301
## Deployment Guide for Hugging Face Spaces

### Step 1 — Train & Export Models (do this locally)

```bash
cd ML-CASE-STUDY-main
python train_models.py
# This creates: models/reg_pipeline.pkl, models/reg_meta.pkl,
#               models/clf_pipeline.pkl, models/clf_meta.pkl
```

---

### Step 2 — Create a Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **New Space**
2. Fill in:
   - **Owner**: your HF username
   - **Space name**: `ml-case-study-23cse301`
   - **License**: MIT
   - **SDK**: **Streamlit**
   - **Visibility**: Public (required for bonus marks)
3. Click **Create Space**

---

### Step 3 — Upload files to the Space

Upload these files/folders to your Space repository:

```
app.py
requirements.txt
models/
  reg_pipeline.pkl
  reg_meta.pkl
  clf_pipeline.pkl
  clf_meta.pkl
.streamlit/
  config.toml
```

> ⚠️ The `models/` folder with `.pkl` files must be included.
> HF Spaces supports files up to 10 GB (LFS for large files).

You can use the HF web UI → "Files" tab → "Add file" → "Upload files".

**Or via Git (recommended):**

```bash
# Install git-lfs (for large pkl files)
git lfs install

# Clone your HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/ml-case-study-23cse301
cd ml-case-study-23cse301

# Copy files
copy ..\app.py .
copy ..\requirements.txt .
xcopy ..\models models\ /E
xcopy ..\.streamlit .streamlit\ /E

# Track pkl files with LFS
git lfs track "*.pkl"
git add .gitattributes
git add .
git commit -m "Initial deployment: ML Case Study 23CSE301"
git push
```

---

### Step 4 — Verify deployment

After pushing, your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/ml-case-study-23cse301
```

The Space will auto-build (1-3 minutes). Check the "App" tab.

---

### Alternative: Streamlit Community Cloud

1. Push your code to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> ⚠️ Streamlit Cloud has a 1 GB file size limit. If pkl files are large,
> use Git LFS or consider re-training with fewer trees.

---

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| `FileNotFoundError: models/...pkl` | Upload the `models/` folder |
| App crashes on large inputs | Inputs are clipped to training range — this is expected |
| Slow startup | First load caches the models; subsequent predictions are fast |
