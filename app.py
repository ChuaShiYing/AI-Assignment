# streamlit_app.py — Streamlit UI for CN→EN Translator (Marian)
# Run locally:
#   streamlit run streamlit_app.py
#
# On Streamlit Cloud:
#   - Push this file + main.py + requirements.txt to GitHub
#   - App file: streamlit_app.py
#   - Optional env:
#       RUN_DIR  : path with best_model/ (if you saved fine-tuned weights)
#       HF_MODEL : HF repo id (e.g., "yourname/marian-zh-en-ft") if no best_model/
#       HF_TOKEN : token for private HF repos (add in Streamlit "Secrets")

from __future__ import annotations
import os
import time
import pandas as pd
import streamlit as st

# Optional login for private HF repos
try:
    from huggingface_hub import login
    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
except Exception:
    pass

from main import Translator

st.set_page_config(page_title="CN→EN Translator", page_icon="🌐", layout="wide")
st.title("🇨🇳 ➜ 🇬🇧 CN → EN Translator (Marian)")
st.caption(
    "Loads **RUN_DIR/best_model** if available; otherwise uses `HF_MODEL` env or base Marian. "
    "GPU is used automatically if available."
)

# ---------- Sidebar ----------
st.sidebar.header("Settings")
default_run_dir = os.environ.get("RUN_DIR", "marian_zh_en_scratch_run")
run_dir = st.sidebar.text_input("Run directory (contains best_model/)", value=default_run_dir)
force_cpu = st.sidebar.checkbox("Force CPU", value=False)

st.sidebar.subheader("Decoding")
beams = st.sidebar.slider("Beam size", 1, 8, 4, 1)
max_new = st.sidebar.slider("Max new tokens", 16, 256, 128, 8)
no_rep = st.sidebar.slider("No-repeat n-gram size", 0, 5, 3, 1)
length_pen = st.sidebar.slider("Length penalty", -1.0, 2.0, 1.0, 0.1)

st.sidebar.subheader("Batching")
batch_size = st.sidebar.slider("Batch size (batch tab)", 1, 64, 16, 1)

# ---------- Cached translator ----------
@st.cache_resource(show_spinner=True)
def get_translator_cached(run_dir: str, force_cpu: bool):
    dev = "cpu" if force_cpu else None
    return Translator(run_dir=run_dir, device=dev)

translator = get_translator_cached(run_dir, force_cpu)

# ---------- Tabs ----------
tab_single, tab_batch = st.tabs(["🔹 Single", "📄 Batch"])

with tab_single:
    st.subheader("Translate a single sentence")
    cn_text = st.text_area("Chinese input", height=160, placeholder="输入中文句子…")

    if st.button("Translate 🚀", type="primary"):
        t0 = time.time()
        out, info = translator.translate(
            cn_text,
            num_beams=beams,
            max_new_tokens=max_new,
            no_repeat_ngram_size=no_rep,
            length_penalty=length_pen,
            return_info=True,
        )
        st.markdown("**English translation**")
        st.text_area("Output", out, height=160)
        st.caption(f"{info} • elapsed={(time.time()-t0)*1000:.0f} ms")

with tab_batch:
    st.subheader("Translate multiple lines")
    st.write("Paste text (one per line) or upload a TXT/CSV file.")

    mode = st.radio("Input mode", ["Paste text", "Upload file"], horizontal=True)
    lines = []

    if mode == "Paste text":
        bulk = st.text_area("One sentence per line", height=200)
        if bulk.strip():
            lines = [ln.strip() for ln in bulk.splitlines() if ln.strip()]
    else:
        up = st.file_uploader("Upload TXT/CSV", type=["txt", "csv"])
        if up:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
                # try common column names; else first column
                cand = [c for c in df.columns if c.lower() in ("cn", "zh", "chinese", "text", "input")]
                col = cand[0] if cand else df.columns[0]
                lines = df[col].astype(str).tolist()
            else:
                txt = up.read().decode("utf-8", errors="ignore")
                lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    if lines:
        if st.button(f"Translate {len(lines)} lines 🚀", type="primary"):
            t0 = time.time()
            outs = translator.translate_batch(
                lines,
                batch_size=batch_size,
                num_beams=beams,
                max_new_tokens=max_new,
                no_repeat_ngram_size=no_rep,
                length_penalty=length_pen,
            )
            dt = time.time() - t0
            df_out = pd.DataFrame({"cn": lines, "pred_en": outs})
            st.dataframe(df_out, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="translations.csv",
                mime="text/csv",
            )
            st.caption(f"Done in {dt:.2f} s • batch_size={batch_size} • beams={beams}")
