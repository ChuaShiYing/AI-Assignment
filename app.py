# streamlit_app.py — Single-sentence CN→EN UI (best_model only)

from __future__ import annotations
import os, time
import streamlit as st

# Friendly message if torch / main.py isn't importable
try:
    from main import Translator, DEFAULT_RUN_DIR
except Exception as e:
    st.error(
        "Failed to import `Translator` from `main.py`.\n\n"
        "Make sure your repository has:\n"
        "• main.py (the best_model-only version)\n"
        "• requirements.txt (includes torch CPU wheel and streamlit)\n\n"
        f"Import error:\n{e}"
    )
    st.stop()

st.set_page_config(page_title="CN→EN Translator (best_model only)", page_icon="🌐", layout="centered")

st.title("🇨🇳 ➜ 🇬🇧 CN → EN Translator")
st.caption("Loads only from **RUN_DIR/best_model/**. No fallback. Single sentence only.")

# ---- Sidebar settings ----
st.sidebar.header("Settings")
run_dir = st.sidebar.text_input("Run directory (must contain best_model/)",
                                value=os.environ.get("RUN_DIR", DEFAULT_RUN_DIR))
force_cpu = st.sidebar.checkbox("Force CPU", value=False)

st.sidebar.subheader("Decoding")
beams      = st.sidebar.slider("Beam size", 1, 8, 4, 1)
max_new    = st.sidebar.slider("Max new tokens", 16, 256, 128, 8)
no_repeat  = st.sidebar.slider("No-repeat n-gram size", 0, 5, 3, 1)
len_pen    = st.sidebar.slider("Length penalty", -1.0, 2.0, 1.0, 0.1)

# ---- Cache the translator (reloads if run_dir/force_cpu change) ----
@st.cache_resource(show_spinner=True)
def get_translator(run_dir: str, cpu: bool):
    device = "cpu" if cpu else None
    # Translator will raise FileNotFoundError if best_model/ is missing
    return Translator(run_dir=run_dir, device=device)

# Build/verify the translator once
try:
    translator = get_translator(run_dir, force_cpu)
except FileNotFoundError as e:
    st.error(
        f"best_model not found or malformed.\n\n"
        f"Expected folder: **{run_dir}/best_model/** with Hugging Face files (config.json, weights, tokenizer, ...).\n\n"
        f"Details:\n{e}"
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize Translator:\n{e}")
    st.stop()

# ---- Single sentence box ----
st.subheader("Translate a single Chinese sentence")
cn_text = st.text_area("Chinese input", height=160, placeholder="输入中文句子…")

if st.button("Translate 🚀", type="primary"):
    if not cn_text.strip():
        st.warning("Please enter a sentence.")
    else:
        t0 = time.time()
        out, info = translator.translate(
            cn_text,
            num_beams=beams,
            max_new_tokens=max_new,
            no_repeat_ngram_size=no_repeat,
            length_penalty=len_pen,
            return_info=True,
        )
        st.markdown("**English translation**")
        st.text_area("Output", out, height=160)
        st.caption(f"{info} • elapsed={(time.time()-t0)*1000:.0f} ms")

st.divider()
