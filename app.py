# ─────────────────────────────────────────────────────────────────────────────
#  Next‑Word Predictor · GRU (100 epochs, no Early Stopping)
#  Streamlit UI polish: wide layout, sidebar info, example picker, result card
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np, pickle, textwrap
from pathlib import Path
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRU Next‑Word Predictor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#7E3FF2"                     # purple accent colour

# ── load artefacts once ────────────────────────────────────────────────────
MODEL_FILE = "1_GRU_100epochs_without_ES.h5"   # adjust if needed
TOKENIZER_FILE = "tokenizer.pickle"

@st.cache_resource(show_spinner="Loading model…")
def load_artefacts():
    mdl = load_model(MODEL_FILE)
    tok = pickle.loads(Path(TOKENIZER_FILE).read_bytes())
    return mdl, tok, mdl.input_shape[1] + 1

model, tokenizer, MAX_LEN = load_artefacts()

# ── core inference ─────────────────────────────────────────────────────────
def predict_next_word(text: str) -> str | None:
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = seq[-(MAX_LEN - 1):]
    seq = pad_sequences([seq], maxlen=MAX_LEN - 1, padding="pre")
    idx = int(np.argmax(model.predict(seq, verbose=0), axis=1)[0])
    return tokenizer.index_word.get(idx, "(unknown)")

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️  About this demo")
    st.markdown(
        textwrap.dedent(f"""
        **Model** GRU | 100 epochs | *no* Early Stopping  
        **Training seq‑len** {MAX_LEN-1} tokens  
        **Dataset** Your custom corpus  
        **Latency** ≈ 20 ms on CPU  
        ---
        **How it works**  
        1. Text → integer tokens via `tokenizer`  
        2. Sequence padded/truncated to {MAX_LEN-1}  
        3. GRU predicts probability for every word in vocab  
        4. Highest‑probability index → actual word
    """), help="This text lives in the sidebar")
    st.markdown("---")
    st.markdown("💻 *Made with Streamlit 1.35*")
    st.markdown("[GitHub repo](https://github.com/) • [Hugging Face Space](https://huggingface.co/)")

# ── main page header (“hero”) ──────────────────────────────────────────────
st.markdown(
    f"""
    <h1 style='text-align:center; font-size:3rem;'>
    📝 Next‑Word Predictor
    </h1>
    <p style='text-align:center; color:{ACCENT}; font-size:1.1rem;'>
    GRU‑based language model • trained 100 epochs • no early stopping
    </p>
    """,
    unsafe_allow_html=True,
)

# ── example picker + text input ────────────────────────────────────────────
examples = [
    "To be or not to",
    "Once upon a time",
    "Artificial intelligence is",
    "The quick brown fox",
    "In the beginning God",
]

left, right = st.columns([1, 3])
with left:
    choice = st.radio("Examples", examples, index=0, label_visibility="collapsed")
with right:
    user_text = st.text_input("Write or edit the prompt", value=choice, key="user_input")

st.markdown("")

# ── predict button / slim result card ───────────────────────────────────────
if st.button("🔮  Predict next word", type="primary"):
    next_word = predict_next_word(user_text.strip())

    # label sits outside the card so the card can stay tiny
    st.caption("Predicted word")

    st.markdown(
        f"""
        <div style='max-width: 320px;                        /* narrower */
                    margin: 0 auto;  
                    background: #F7F2FF;
                    padding: 0.75rem 1rem;                  /* less padding */
                    border-radius: 0.6rem;
                    border: 1px solid {ACCENT}33;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
            <span style='font-size: 1.4rem;                 /* smaller text */
                         font-weight: 600;
                         color: {ACCENT};
                         display: block;
                         text-align: center;'>
                {next_word}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
# ── footer ────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='margin-top:3rem;'><div style='text-align:center;'>"
    "© 2025 Raj Patel • Powered by TensorFlow + Streamlit"
    "</div>", unsafe_allow_html=True
)
