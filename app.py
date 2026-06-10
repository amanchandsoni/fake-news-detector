import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

@st.cache_resource
def load_artifacts():
    model = Sequential([
        Embedding(input_dim=20000, output_dim=128, input_length=300),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.build(input_shape=(None, 300))
    model.load_weights('fake_news_weights.weights.h5')
    with open('tokenizer (1).pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('max_len.pkl', 'rb') as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

def predict_news(text, model, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)[0][0]
    label = "Real News" if pred >= 0.5 else "Fake News"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, float(confidence), float(pred)

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

*, html, body { margin:0; padding:0; box-sizing:border-box; }
.stApp { background:#07070f; font-family:'DM Sans',sans-serif; }

/* ORBS */
.orb { position:fixed; border-radius:50%; filter:blur(90px); pointer-events:none; z-index:0; }
.orb1 { width:600px; height:600px; background:radial-gradient(circle,#7c3aed,transparent); top:-200px; left:-200px; opacity:0.15; animation:o1 10s ease-in-out infinite alternate; }
.orb2 { width:500px; height:500px; background:radial-gradient(circle,#1d4ed8,transparent); bottom:-150px; right:-150px; opacity:0.12; animation:o2 12s ease-in-out infinite alternate; }
.orb3 { width:400px; height:400px; background:radial-gradient(circle,#be185d,transparent); top:40%; left:40%; opacity:0.08; animation:o3 8s ease-in-out infinite alternate; }
@keyframes o1 { from{transform:translate(0,0) scale(1)} to{transform:translate(40px,30px) scale(1.1)} }
@keyframes o2 { from{transform:translate(0,0) scale(1)} to{transform:translate(-30px,-40px) scale(1.08)} }
@keyframes o3 { from{transform:translate(0,0)} to{transform:translate(20px,-20px)} }

/* HERO */
.hero { text-align:center; padding:3rem 1rem 2rem; position:relative; z-index:1; }
.badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(29,78,216,0.2));
    border:1px solid rgba(124,58,237,0.4);
    border-radius:100px; padding:0.5rem 1.5rem;
    font-size:0.7rem; letter-spacing:4px; text-transform:uppercase; color:#a78bfa;
    margin-bottom:1.5rem;
}
.hero-title {
    font-family:'Playfair Display',serif; font-size:5.5rem; font-weight:900;
    line-height:1; color:#fff; margin-bottom:1rem;
    text-shadow:0 0 120px rgba(124,58,237,0.5);
}
.gradient-text {
    background:linear-gradient(135deg,#a78bfa 0%,#60a5fa 40%,#f472b6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { color:#374151; font-size:1rem; font-weight:300; letter-spacing:0.5px; }

/* STATS */
.stats { display:flex; gap:1rem; justify-content:center; margin:2rem auto; max-width:900px; position:relative; z-index:1; }
.sc {
    flex:1; background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:20px; padding:1.5rem; text-align:center;
    transition:all 0.3s; position:relative; overflow:hidden;
}
.sc::after { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,#7c3aed,transparent); opacity:0; transition:0.3s; }
.sc:hover::after { opacity:1; }
.sc:hover { border-color:rgba(124,58,237,0.3); transform:translateY(-4px); box-shadow:0 20px 40px rgba(0,0,0,0.3); }
.sv { font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:#a78bfa; }
.sk { font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; color:#1f2937; margin-top:0.3rem; }

/* BUTTONS — override streamlit */
.stButton > button {
    font-family:'DM Sans',sans-serif !important;
    font-weight:600 !important; font-size:0.95rem !important;
    border-radius:14px !important; padding:0.8rem 1.5rem !important;
    width:100% !important; border:none !important; cursor:pointer !important;
    transition:all 0.3s ease !important; letter-spacing:0.3px !important;
}

/* Primary analyze button */
div[data-testid="column"]:first-child .stButton > button {
    background:linear-gradient(135deg,#7c3aed,#4f46e5,#2563eb) !important;
    color:white !important;
    box-shadow:0 4px 20px rgba(124,58,237,0.5), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    transform:translateY(-3px) !important;
    box-shadow:0 8px 30px rgba(124,58,237,0.7) !important;
}

/* Clear button */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background:rgba(255,255,255,0.05) !important;
    color:#9ca3af !important;
    border:1px solid rgba(255,255,255,0.1) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background:rgba(239,68,68,0.15) !important;
    color:#ef4444 !important;
    border-color:rgba(239,68,68,0.3) !important;
}

/* Sample buttons */
.stButton > button[kind="secondary"] {
    background:rgba(255,255,255,0.03) !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    color:#9ca3af !important;
}

/* Textarea */
.stTextArea textarea {
    background:rgba(255,255,255,0.03) !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-radius:18px !important; color:#e2e8f0 !important;
    font-family:'DM Sans',sans-serif !important; font-size:0.95rem !important;
    line-height:1.7 !important; padding:1.2rem !important;
    transition:border-color 0.3s !important;
}
.stTextArea textarea:focus {
    border-color:#7c3aed !important;
    box-shadow:0 0 0 4px rgba(124,58,237,0.12) !important;
}

/* PANEL */
.panel-label {
    font-size:0.65rem; letter-spacing:4px; text-transform:uppercase; color:#7c3aed;
    margin-bottom:1rem; display:flex; align-items:center; gap:0.5rem;
}
.panel-label::before { content:'▸'; }

/* RESULTS */
.res-real {
    background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(5,150,105,0.05));
    border:2px solid #10b981; border-radius:28px; padding:2.5rem 2rem;
    text-align:center; animation:pop 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
    box-shadow:0 0 80px rgba(16,185,129,0.12),inset 0 0 80px rgba(16,185,129,0.02);
    position:relative; overflow:hidden;
}
.res-real::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,#10b981,transparent);
}
.res-fake {
    background:linear-gradient(135deg,rgba(239,68,68,0.1),rgba(185,28,28,0.05));
    border:2px solid #ef4444; border-radius:28px; padding:2.5rem 2rem;
    text-align:center; animation:pop 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
    box-shadow:0 0 80px rgba(239,68,68,0.12),inset 0 0 80px rgba(239,68,68,0.02);
    position:relative; overflow:hidden;
}
.res-fake::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,#ef4444,transparent);
}
@keyframes pop { from{transform:scale(0.75);opacity:0} to{transform:scale(1);opacity:1} }

.res-icon { font-size:4.5rem; animation:iconpop 0.5s 0.2s cubic-bezier(0.34,1.56,0.64,1) both; }
@keyframes iconpop { from{transform:scale(0) rotate(-20deg);opacity:0} to{transform:scale(1) rotate(0);opacity:1} }
.res-label-real { font-family:'Playfair Display',serif; font-size:2.5rem; font-weight:900; color:#10b981; letter-spacing:4px; margin:0.3rem 0; }
.res-label-fake { font-family:'Playfair Display',serif; font-size:2.5rem; font-weight:900; color:#ef4444; letter-spacing:4px; margin:0.3rem 0; }
.res-desc { color:#4b5563; font-size:0.88rem; font-weight:300; }

/* CONFIDENCE BAR */
.conf-box { margin-top:1.8rem; }
.conf-row { display:flex; justify-content:space-between; margin-bottom:0.5rem; }
.conf-t { font-size:0.68rem; letter-spacing:2px; text-transform:uppercase; color:#374151; }
.conf-v-real { font-size:1rem; font-weight:700; color:#10b981; }
.conf-v-fake { font-size:1rem; font-weight:700; color:#ef4444; }
.bar-bg { background:rgba(255,255,255,0.04); border-radius:100px; height:8px; overflow:hidden; }
.bar-real { height:100%; border-radius:100px; background:linear-gradient(90deg,#059669,#10b981,#34d399); animation:grow 1s ease both; }
.bar-fake { height:100%; border-radius:100px; background:linear-gradient(90deg,#b91c1c,#ef4444,#f87171); animation:grow 1s ease both; }
@keyframes grow { from{width:0!important} }

/* MODEL SPEC CARD */
.spec-card {
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07);
    border-radius:20px; padding:1.5rem; margin-top:1rem;
}
.spec-row { display:flex; justify-content:space-between; align-items:center; padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.spec-row:last-child { border-bottom:none; }
.spec-key { font-size:0.78rem; color:#374151; }
.spec-val { font-size:0.82rem; color:#a78bfa; font-weight:600; }

/* EMPTY STATE */
.empty { text-align:center; padding:5rem 2rem; }
.empty-icon { font-size:3.5rem; opacity:0.08; margin-bottom:1rem; }
.empty-text { color:#1f2937; font-size:0.85rem; letter-spacing:1px; }

/* SIDEBAR */
section[data-testid="stSidebar"] { background:#060610 !important; border-right:1px solid rgba(255,255,255,0.04) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background:linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color:white !important; box-shadow:0 4px 16px rgba(124,58,237,0.4) !important;
}

#MainMenu, footer, header { visibility:hidden; }
div[data-testid="stMarkdownContainer"] p { color:#6b7280; }
</style>

<div class='orb orb1'></div>
<div class='orb orb2'></div>
<div class='orb orb3'></div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0;'>
    <div style='font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;color:#f0eeff;margin-bottom:0.3rem'>🧠 Model Specs</div>
    <div style='color:#374151;font-size:0.78rem;margin-bottom:1.5rem'>Bidirectional LSTM Architecture</div>
    </div>
    """, unsafe_allow_html=True)

    specs = {
        "🏗️ Architecture": "Bidirectional LSTM",
        "📚 Vocab Size": "20,000 words",
        "🔢 Embedding Dim": "128 dimensions",
        "🧬 LSTM Units": "64 × 2 (BiLSTM)",
        "🎯 Dropout": "0.5 (50%)",
        "📏 Max Sequence": "300 tokens",
        "⚡ Optimizer": "Adam (lr=0.001)",
        "📉 Loss Fn": "Binary Crossentropy",
        "✅ Test Accuracy": "100%",
        "📦 Training Data": "8,980 samples",
        "🟢 Real News": "4,729 samples",
        "🔴 Fake News": "4,251 samples",
    }

    for k, v in specs.items():
        color = "#10b981" if "100%" in v else "#a78bfa"
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
            padding:0.65rem 0.9rem;margin-bottom:0.4rem;
            background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
            border-radius:12px;'>
            <span style='font-size:0.78rem;color:#4b5563'>{k}</span>
            <span style='font-size:0.78rem;font-weight:600;color:{color}'>{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.15);border-radius:14px;padding:1rem;'>
    <div style='font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#7c3aed;margin-bottom:0.7rem'>💡 Tips</div>
    <div style='color:#374151;font-size:0.8rem;line-height:1.9'>
    • Full articles work best<br>
    • English news only<br>
    • Min 5 words needed<br>
    • More text = better accuracy
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='badge'>◆ &nbsp; Deep Learning &nbsp;·&nbsp; BiLSTM &nbsp;·&nbsp; NLP &nbsp;·&nbsp; RRN Part 3 &nbsp; ◆</div>
    <div class='hero-title'>Fake News<br><span class='gradient-text'>Detector</span></div>
    <div class='hero-sub'>AI-powered misinformation detection using Bidirectional LSTM neural network</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD ───────────────────────────────────────────────────────────────────────
try:
    model, tokenizer, max_len = load_artifacts()
    model_ok = True
except Exception as e:
    st.error(f"❌ Error: {e}")
    model_ok = False

if model_ok:
    # STATS
    st.markdown(f"""
    <div class='stats'>
        <div class='sc'><div class='sv'>100%</div><div class='sk'>Test Accuracy</div></div>
        <div class='sc'><div class='sv'>8,980</div><div class='sk'>Training Samples</div></div>
        <div class='sc'><div class='sv'>{max_len}</div><div class='sk'>Max Sequence Length</div></div>
        <div class='sc'><div class='sv'>BiLSTM</div><div class='sk'>Model Architecture</div></div>
        <div class='sc'><div class='sv'>20K</div><div class='sk'>Vocabulary Size</div></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("<div class='panel-label'>Input Article</div>", unsafe_allow_html=True)

        if 'nt' not in st.session_state:
            st.session_state['nt'] = ''

        news = st.text_area("", value=st.session_state['nt'],
            placeholder="Paste your news article here...\n\nTip: Longer articles give more accurate predictions.",
            height=260, key="ta")

        wc = len(news.strip().split()) if news.strip() else 0
        cc = len(news.strip())

        col_info, col_a, col_c = st.columns([1.2, 2, 0.8])
        with col_info:
            st.caption(f"📝 {wc} words")
        with col_a:
            analyze = st.button("🔍  Analyze Article", use_container_width=True)
        with col_c:
            clear = st.button("🗑️ Clear", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='panel-label'>Quick Samples</div>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            if st.button("✅  Real News Sample", use_container_width=True):
                st.session_state['nt'] = "WASHINGTON (Reuters) - The United States Federal Reserve raised interest rates by a quarter of a percentage point on Wednesday and projected two more hikes by the end of 2023, reflecting its optimism about economic growth and a tightening labor market despite recent banking sector turmoil."
                st.rerun()
        with s2:
            if st.button("⛔  Fake News Sample", use_container_width=True):
                st.session_state['nt'] = "SHOCKING: NASA scientists have confirmed that a massive alien spacecraft has been orbiting Earth for the past 6 months. The government is hiding this truth. Multiple whistleblowers have come forward with evidence. Share this before it gets deleted!"
                st.rerun()

    with right:
        st.markdown("<div class='panel-label'>Analysis Result</div>", unsafe_allow_html=True)

        if analyze:
            if not news.strip():
                st.warning("⚠️ Please paste some news text first!")
            elif wc < 5:
                st.warning("⚠️ Please enter at least 5 words!")
            else:
                with st.spinner("🧠 Analyzing with BiLSTM model..."):
                    time.sleep(0.5)
                    label, confidence, raw = predict_news(news, model, tokenizer, max_len)

                bw = int(confidence * 100)

                if label == "Real News":
                    st.markdown(f"""
                    <div class='res-real'>
                        <div class='res-icon'>✅</div>
                        <div class='res-label-real'>REAL NEWS</div>
                        <div class='res-desc'>This article appears credible and authentic</div>
                        <div class='conf-box'>
                            <div class='conf-row'>
                                <span class='conf-t'>Model Confidence</span>
                                <span class='conf-v-real'>{confidence*100:.1f}%</span>
                            </div>
                            <div class='bar-bg'><div class='bar-real' style='width:{bw}%'></div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='res-fake'>
                        <div class='res-icon'>⛔</div>
                        <div class='res-label-fake'>FAKE NEWS</div>
                        <div class='res-desc'>This article appears misleading or fabricated</div>
                        <div class='conf-box'>
                            <div class='conf-row'>
                                <span class='conf-t'>Model Confidence</span>
                                <span class='conf-v-fake'>{confidence*100:.1f}%</span>
                            </div>
                            <div class='bar-bg'><div class='bar-fake' style='width:{bw}%'></div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediction", "✅ Real" if label == "Real News" else "⛔ Fake")
                m2.metric("Confidence", f"{confidence*100:.1f}%")
                m3.metric("Raw Score", f"{raw:.4f}")

                # Score bar
                st.markdown(f"""
                <div style='margin-top:1rem;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:1.2rem;'>
                    <div style='font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#374151;margin-bottom:0.8rem'>Score Breakdown</div>
                    <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;'>
                        <span style='font-size:0.78rem;color:#10b981;width:60px'>Real</span>
                        <div style='flex:1;background:rgba(16,185,129,0.1);border-radius:100px;height:6px;'>
                            <div style='width:{int(raw*100)}%;height:100%;background:#10b981;border-radius:100px'></div>
                        </div>
                        <span style='font-size:0.78rem;color:#10b981;width:40px;text-align:right'>{raw*100:.1f}%</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:0.5rem;'>
                        <span style='font-size:0.78rem;color:#ef4444;width:60px'>Fake</span>
                        <div style='flex:1;background:rgba(239,68,68,0.1);border-radius:100px;height:6px;'>
                            <div style='width:{int((1-raw)*100)}%;height:100%;background:#ef4444;border-radius:100px'></div>
                        </div>
                        <span style='font-size:0.78rem;color:#ef4444;width:40px;text-align:right'>{(1-raw)*100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class='empty'>
                <div class='empty-icon'>🔍</div>
                <div class='empty-text'>Paste an article and<br>click Analyze to begin</div>
            </div>
            """, unsafe_allow_html=True)

        if clear:
            st.session_state['nt'] = ''
            st.rerun()

st.markdown("""
<div style='text-align:center;padding:3rem 1rem 1.5rem;color:#111827;font-size:0.78rem;position:relative;z-index:1;letter-spacing:1px'>
    BUILT WITH ❤️ &nbsp;·&nbsp; FAKE NEWS DETECTOR &nbsp;·&nbsp; BIDIRECTIONAL LSTM &nbsp;·&nbsp; RRN PART 3 OF DL
</div>
""", unsafe_allow_html=True)