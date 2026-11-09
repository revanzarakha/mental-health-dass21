# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
import warnings
import json
from pathlib import Path
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# =========================================================
# Helper functions
# =========================================================
def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[load_model_safe] failed to load {path}: {e}")
        return None

def recommendation_text(domain_name, level_key):
    # level_key: normal/mild/moderate/severe/extreme
    # Dibenerin: NORMAL jangan disebut "ringan"
    if level_key == "normal":
        return (
            "Tidak terindikasi gejala yang signifikan.\n"
            "Pertahankan pola tidur, makan, aktivitas fisik, dan manajemen stres."
        )
    if level_key == "mild":
        return (
            "Gejala terindikasi ringan.\n"
            "Pertahankan pola tidur, makan, dan aktivitas fisik; kurangi pemicu stres."
        )
    if level_key == "moderate":
        return (
            "Gejala berada di tingkat sedang.\n"
            "Coba teknik relaksasi, journaling, perbaiki pola tidur, dan pertimbangkan screening lanjutan."
        )
    return (
        "Gejala terindikasi berat.\n"
        "Disarankan screening lebih mendalam atau konsultasi dengan profesional kesehatan mental."
    )

def top_contributors_text(domain_questions, answers, top_k=3):
    arr = np.array(answers, dtype=float)
    idx = np.argsort(-arr)[:top_k]
    selected = [domain_questions[i] for i in idx if arr[i] > 0]
    return selected if selected else []

def severity_bar_color(level_key: str):
    # Hijau: Normal/Ringan | Kuning: Sedang | Merah: Berat/Sangat Berat
    if level_key in ("normal", "mild"):
        return "rgba(162, 245, 205, 0.95)"  # hijau lembut
    if level_key == "moderate":
        return "rgba(255, 236, 156, 0.95)"  # kuning lembut
    return "rgba(255, 166, 166, 0.95)"      # merah lembut


# =========================================================
# ML helpers (pipeline-first, legacy fallback)
# =========================================================
def load_pipeline_or_legacy(pipeline_path, legacy_path):
    """
    Prioritas:
    1) pipeline_*.pkl (scaler+model) -> paling aman
    2) model_*.pkl    (legacy)       -> fallback
    Return: (model, is_pipeline: bool)
    """
    m = load_model_safe(pipeline_path)
    if m is not None:
        print(f"[INFO] Loaded pipeline: {pipeline_path}")
        return m, True
    m = load_model_safe(legacy_path)
    if m is not None:
        print(f"[WARN] Loaded legacy: {legacy_path}")
        return m, False
    print(f"[ERROR] Failed to load: {pipeline_path} | {legacy_path}")
    return None, False

@st.cache_resource
def load_artifacts():
    feature_cols = None
    feature_path = MODEL_DIR / "feature_columns.json"
    if feature_path.exists():
        try:
            with open(feature_path, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
            print(f"[INFO] Loaded feature_columns.json with {len(feature_cols)} features")
        except Exception as e:
            print(f"[WARN] Failed to load feature_columns.json: {e}")
            feature_cols = None

    lr_dep, lr_dep_is_pipe = load_pipeline_or_legacy(str(MODEL_DIR / "pipeline_logreg_depresi.pkl"), str(MODEL_DIR / "model_logreg_depresi.pkl"))
    rf_dep, rf_dep_is_pipe = load_pipeline_or_legacy(str(MODEL_DIR / "pipeline_rf_depresi.pkl"),     str(MODEL_DIR / "model_rf_depresi.pkl"))

    lr_anx, lr_anx_is_pipe = load_pipeline_or_legacy(str(MODEL_DIR / "pipeline_logreg_anxiety.pkl"), str(MODEL_DIR / "model_logreg_anxiety.pkl"))
    rf_anx, rf_anx_is_pipe = load_pipeline_or_legacy(str(MODEL_DIR / "pipeline_rf_anxiety.pkl"),     str(MODEL_DIR / "model_rf_anxiety.pkl"))

    lr_str, lr_str_is_pipe = load_pipeline_or_legacy(str(MODEL_DIR / "pipeline_logreg_stress.pkl"), str(MODEL_DIR / "model_logreg_stress.pkl"))
    rf_str, rf_str_is_pipe = load_pipeline_or_legacy(str(MODEL_DIR / "pipeline_rf_stress.pkl"),     str(MODEL_DIR / "model_rf_stress.pkl"))

    scaler = load_model_safe(str(MODEL_DIR / "scaler.pkl"))  # legacy only

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "models": {
            "dep": {"lr": lr_dep, "rf": rf_dep, "lr_pipe": lr_dep_is_pipe, "rf_pipe": rf_dep_is_pipe},
            "anx": {"lr": lr_anx, "rf": rf_anx, "lr_pipe": lr_anx_is_pipe, "rf_pipe": rf_anx_is_pipe},
            "str": {"lr": lr_str, "rf": rf_str, "lr_pipe": lr_str_is_pipe, "rf_pipe": rf_str_is_pipe},
        }
    }

def align_and_scale_input(X_df, scaler):
    """
    Untuk legacy model: align feature & scale
    """
    X_vals = X_df.values.astype(float)

    if scaler is None:
        print("[align_and_scale_input] scaler not found -> fallback normalize (/3.0)")
        return np.clip(X_vals / 3.0, 0, 1)

    if hasattr(scaler, "feature_names_in_"):
        expected_names = list(scaler.feature_names_in_)
        aligned = []
        for name in expected_names:
            if name in X_df.columns:
                aligned.append(float(X_df.at[0, name]))
            else:
                aligned.append(0.0)
                print(f"[align_and_scale_input] missing feature '{name}' -> padded 0.0")
        X_aligned = np.array([aligned], dtype=float)
    else:
        expected_n = int(getattr(scaler, "n_features_in_", getattr(scaler, "mean_", np.array([])).shape[0]))
        actual_n = X_df.shape[1]
        arr = X_df.values[0].astype(float)

        if actual_n < expected_n:
            pad = np.zeros(expected_n - actual_n, dtype=float)
            X_aligned = np.array([np.concatenate([arr, pad])], dtype=float)
        elif actual_n > expected_n:
            X_aligned = np.array([arr[:expected_n]], dtype=float)
        else:
            X_aligned = np.array([arr], dtype=float)

    try:
        X_scaled = scaler.transform(X_aligned)
    except Exception as e:
        print(f"[align_and_scale_input] scaler.transform failed: {e} -> falling back to /3.0 normalization")
        X_scaled = np.clip(X_aligned / 3.0, 0, 1)

    return X_scaled

def guard_overconfidence(conf, answers_subset, pred_class=None):
    """
    Guard tampilan agar tidak misleading jika model terlalu yakin (overconfident).
    Return: (conf_calibrated, adjusted_flag)
    """
    if conf is None or np.isnan(conf):
        return conf, False

    conf = float(conf)
    original = conf

    mean_ans = float(np.mean(answers_subset))
    std_ans = float(np.std(answers_subset))

    if conf >= 0.97 and mean_ans < 2.2:
        shrink = 0.06 + (2.2 - mean_ans) * 0.04
        conf -= shrink

    if conf >= 0.95 and std_ans < 0.6:
        conf -= 0.04

    conf = float(np.clip(conf, 0.05, 0.99))
    adjusted = (abs(conf - original) >= 1e-9)
    return conf, adjusted

def model_pred_and_conf(model, X):
    """
    Multiclass safe:
    - pakai classes_ kalau ada (biar mapping kelas gak ngaco)
    - confidence = proba kelas terpilih
    """
    if model is None:
        return None, None

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            classes = getattr(model, "classes_", None)
            pred = int(classes[idx]) if classes is not None else idx
            conf = float(proba[idx])
            return pred, conf

        pred = model.predict(X)
        return int(pred[0]), 0.50
    except Exception as e:
        print(f"[model_pred_and_conf] failed: {e}")
        return None, None


# =========================================================
# UI Setup
# =========================================================
st.set_page_config(page_title="Mental Health Analyzer (DASS-21)", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"], .stApp { font-size: 16px !important; }

/* ===== Animasi ===== */
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
h1, h2, h3, h4 { animation: fadeIn 0.8s ease forwards; }

section.main > div.block-container{
  max-width: 980px;
  padding-top: 1.2rem;
}

/* ===== Form ===== */
div[data-testid="stForm"] {
  animation: fadeIn 0.6s ease forwards;
  border: 1px solid rgba(255,255,255,0.10);
  padding: 20px;
  border-radius: 14px;
  background: rgba(30, 30, 30, 0.55);
  box-shadow: 0 10px 28px rgba(0,0,0,0.18);
}

/* ===== Button ===== */
button[kind="primary"] {
  background: linear-gradient(90deg, #00BFFF, #1E90FF);
  color: white !important;
  border: none;
  transition: 0.3s;
  border-radius: 12px !important;
  font-weight: 700;
}
button[kind="primary"]:hover {
  transform: scale(1.02);
  background: linear-gradient(90deg, #1E90FF, #00BFFF);
  box-shadow: 0 0 12px rgba(0,191,255,0.35);
}

table, tr, td { border: none !important; background: transparent !important; }

/* ===== Likert ===== */
.likert-question { margin: 18px 0 6px 0; font-size: 15px; font-weight: 650; color: #EAEAEA; }
.likert-ends { font-size: 13px; opacity: .85; margin-top: 8px; white-space: nowrap; }
.likert-ends.left { text-align: left; color: #BFEED7; }
.likert-ends.right { text-align: right; color: #E0D2FF; }

/* ===== Card ===== */
.wf-card{
  background: rgba(30,30,30,0.55);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 38px 40px;
  box-shadow: 0 14px 32px rgba(0,0,0,0.25);
}

.wf-title{
  text-align:center;
  font-size: 18px;
  font-weight: 780;
  color: rgba(255,255,255,0.92);
  margin-bottom: 18px;
}

/* ===== Layout ===== */
.wf-main{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 18px;
  margin-bottom: 18px;
}

.wf-left{
  display:flex;
  flex-direction:column;
  align-items:flex-start;
  min-width: 170px;
}

.wf-right{
  display:flex;
  flex-direction:column;
  align-items:flex-end;
  text-align:right;
  gap: 6px;
}

/* ===== Text ===== */
.wf-pct{ font-size: 40px; font-weight: 900; color:#fff; }

.wf-pill{
  display:inline-block;
  padding: 10px 18px;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.90);
  font-size: 14.5px;
  line-height: 1.35;
  text-align:center;
  max-width: fit-content;
}

.wf-sub-left{
  margin-top: 6px;
  font-size: 12.5px;
  color: rgba(255,255,255,0.70);
  text-align: left;
}

.wf-sub-right{
  margin-top: 6px;
  font-size: 12.5px;
  color: rgba(255,255,255,0.70);
  text-align: right;
}

/* ===== Bar ===== */
.wf-bar-wrap{
  margin-top: 10px;
  margin-bottom: 22px;
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.12);
  overflow: hidden;
}
.wf-bar{ height:100%; border-radius:999px; }

/* ===== Content ===== */
.wf-text{
  margin-top: 16px;
  font-size: 16px;
  line-height: 1.65;
  color: rgba(255,255,255,0.80);
}
.wf-text b{ color: rgba(255,255,255,0.95); }

.wf-text ul{ margin: 10px 0 0 20px; }
.wf-text li{ margin: 7px 0; }

.wf-rekom{
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid rgba(255,255,255,0.10);
}

.note-box{
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  border-radius: 14px;
  padding: 12px 14px;
  color: rgba(255,255,255,0.78);
  font-size: 13.5px;
}

.card-spacer{ height: 20px; }

/* ===== Likert Radio (punya kamu, gak diubah) ===== */
div[data-testid="stRadio"]{
  width: 100% !important;
}
div[data-testid="stRadio"] div[role="radiogroup"]{
  display: flex !important;
  flex-direction: row !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 18px !important;

  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}
div[data-testid="stRadio"] > div{
  width: 100% !important;
}
div[data-testid="stRadio"] label > div:last-child{
  display: none !important;
}
div[data-testid="stRadio"] label{
  padding: 0 !important;
  margin: 0 !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child{
  width: 40px !important;
  height: 30px !important;
  border-radius: 999px !important;
  background: transparent !important;
  border: 2.2px solid rgba(220, 220, 220, 0.35) !important;
  position: relative !important;
  box-shadow: none !important;
  transition: transform .12s ease, background .12s ease, border-color .12s ease !important;
}
div[data-testid="stRadio"] label:hover [data-baseweb="radio"] > div:first-child{
  transform: scale(1.05);
}
div[data-testid="stRadio"] input:checked + div{
  border-color: rgba(200,200,200,0.85) !important;
  background: rgba(200,200,200,0.85) !important;
}
div[data-testid="stRadio"] input:checked + div::after{
  content: "✓";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -54%);
  font-weight: 900;
  font-size: 16px;
  color: #0B0B0B;
}
div[data-testid="stRadio"] label:nth-of-type(1) [data-baseweb="radio"] > div:first-child{
  border-color: rgba(162, 245, 205, 0.85) !important;
}
div[data-testid="stRadio"] label:nth-of-type(1) input:checked + div{
  background: rgba(162, 245, 205, 0.92) !important;
  border-color: rgba(162, 245, 205, 0.92) !important;
}
div[data-testid="stRadio"] label:nth-of-type(2) [data-baseweb="radio"] > div:first-child{
  border-color: rgba(160, 210, 255, 0.85) !important;
}
div[data-testid="stRadio"] label:nth-of-type(2) input:checked + div{
  background: rgba(160, 210, 255, 0.92) !important;
  border-color: rgba(160, 210, 255, 0.92) !important;
}
div[data-testid="stRadio"] label:nth-of-type(3) [data-baseweb="radio"] > div:first-child{
  border-color: rgba(255, 236, 156, 0.85) !important;
}
div[data-testid="stRadio"] label:nth-of-type(3) input:checked + div{
  background: rgba(255, 236, 156, 0.92) !important;
  border-color: rgba(255, 236, 156, 0.92) !important;
}
div[data-testid="stRadio"] label:nth-of-type(4) [data-baseweb="radio"] > div:first-child{
  border-color: rgba(190, 150, 255, 0.80) !important;
}
div[data-testid="stRadio"] label:nth-of-type(4) input:checked + div{
  background: rgba(190, 150, 255, 0.90) !important;
  border-color: rgba(190, 150, 255, 0.90) !important;
}
.likert-center{
  width: 100%;
  display: flex;
  justify-content: center;
}
div[data-testid="stRadio"]{
  width: 100% !important;
}
div[data-testid="stRadio"] div[role="radiogroup"]{
  width: 220px !important;
  justify-content: space-between !important;
  gap: 0 !important;
  margin: 0 auto !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Header
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top:-10px; animation:fadeIn 0.6s ease;">
  <h1 style="color:#00BFFF; font-weight:900; font-size:36px;"> Mental Health Analyzer</h1>
  <p style="color:#B0B0B0; font-size:15px; margin-top:-8px;">
    Berdasarkan Kuesioner <b>DASS-21</b> — deteksi awal <b>Depresi, Kecemasan, dan Stres</b>
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Silahkan mengisi kuesioner singkat (skala **0 – 3**) untuk melihat estimasi probabilitas **Depression**, "
    "**Anxiety**, dan **Stress**."
)
st.markdown("---")

# =========================================================
# Questions (pastikan teks ini match dengan feature_columns.json training)
# =========================================================
depression_qs = [
    "Saya merasa tidak memiliki semangat untuk melakukan hal-hal yang biasa saya nikmati.",
    "Saya sulit untuk menikmati aktivitas sehari-hari.",
    "Saya merasa kehilangan energi atau cepat lelah.",
    "Saya merasa kurang berguna dalam hidup.",
    "Saya merasa hidup tidak berharga.",
    "Saya merasa sedih atau murung tanpa alasan yang jelas.",
    "Saya merasa belum merasakan arti hidup."
]
anxiety_qs = [
    "Saya merasa khawatir akan hal-hal yang belum tentu terjadi.",
    "Saya merasa tegang atau gugup tanpa alasan jelas.",
    "Saya mengalami kesulitan untuk rileks.",
    "Saya mudah panik dalam situasi tertentu.",
    "Saya merasakan jantung berdebar atau napas terasa cepat saat cemas.",
    "Saya sering merasa ketakutan tanpa alasan yang jelas.",
    "Saya sulit mengendalikan kekhawatiran saya."
]
stress_qs = [
    "Saya merasa sulit untuk tenang setelah menghadapi masalah kecil.",
    "Saya merasa mudah tersinggung atau marah.",
    "Saya merasa tekanan hidup terlalu berat.",
    "Saya merasa sulit untuk istirahat dengan tenang.",
    "Saya merasa sulit memusatkan perhatian karena banyak pikiran.",
    "Saya merasa waktu selalu tidak cukup untuk menyelesaikan tugas.",
    "Saya merasa kehilangan kontrol terhadap hal-hal di sekitar saya."
]

st.markdown(
    """
    <div style="background-color:#1e1e1e; padding:15px; border-radius:12px;
                border-left:5px solid #00BFFF; margin-bottom:20px;">
        <p style="font-size:15px; margin-bottom:10px;">
            <b>Jawablah setiap pertanyaan sesuai kondisi dalam minggu terakhir.</b>
        </p>
        <p style="font-size:14px; margin-bottom:5px;"><b>Skala Jawaban:</b></p>
        <table style="font-size:14px; margin-left:10px; border-collapse:collapse; width:auto;">
            <tr><td style="width:25px;">0.</td><td>Tidak Pernah</td></tr>
            <tr><td>1.</td><td>Kadang-kadang</td></tr>
            <tr><td>2.</td><td>Cukup Sering</td></tr>
            <tr><td>3.</td><td>Sering Sekali</td></tr>
        </table>
    </div>
    """,
    unsafe_allow_html=True
)

def likert_dass(question: str, key: str, no: int = None):
    nomor = f"{no}. " if no is not None else ""
    st.markdown(f"<div class='likert-question'>{nomor}{question}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.6, 2.8, 1.6])

    c1.markdown("<div class='likert-ends left'>Tidak Pernah</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='likert-center'>", unsafe_allow_html=True)
        val = st.radio(
            label="",
            options=[0, 1, 2, 3],
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c3.markdown("<div class='likert-ends right'>Sering Sekali</div>", unsafe_allow_html=True)

    st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
    return val

# =========================================================
# FORM INPUT
# =========================================================
with st.form("dass_form_input"):
    st.markdown("<h2 style='text-align:center; margin-top:6px;'>Bagian Depresi</h2>", unsafe_allow_html=True)
    dep_answers = [likert_dass(q, key=f"dep_{i}_input") for i, q in enumerate(depression_qs)]

    st.markdown("<h2 style='text-align:center; margin-top:6px;'>Bagian Kecemasan</h2>", unsafe_allow_html=True)
    anx_answers = [likert_dass(q, key=f"anx_{i}_input") for i, q in enumerate(anxiety_qs)]

    st.markdown("<h2 style='text-align:center; margin-top:6px;'>Bagian Stress</h2>", unsafe_allow_html=True)
    str_answers = [likert_dass(q, key=f"str_{i}_input") for i, q in enumerate(stress_qs)]

    submitted = st.form_submit_button("Deteksi Sekarang", use_container_width=True)

if not submitted:
    st.info("Isi semua bagian lalu tekan **Deteksi Sekarang** untuk melihat hasil.")
    st.stop()

# =========================================================
# Loading/progress
# =========================================================
placeholder = st.empty()
with placeholder.container():
    st.markdown("<div style='text-align:center;'><h3>Sedang menganalisis jawaban kamu...</h3></div>", unsafe_allow_html=True)
    progress = st.progress(0)
    status_text = st.empty()
    for i in range(101):
        progress.progress(i)
        if i < 30:
            status_text.markdown("<p style='text-align:center;'> Menghitung probabilitas model...</p>", unsafe_allow_html=True)
        elif i < 60:
            status_text.markdown("<p style='text-align:center;'> Menggabungkan hasil analisis DASS-21...</p>", unsafe_allow_html=True)
        elif i < 90:
            status_text.markdown("<p style='text-align:center;'> Memvalidasi skor & stabilitas hasil...</p>", unsafe_allow_html=True)
        else:
            status_text.markdown("<p style='text-align:center;'> Menyelesaikan tampilan hasil...</p>", unsafe_allow_html=True)
        time.sleep(0.02)
time.sleep(0.25)
placeholder.empty()

# =========================================================
# Build feature DataFrame 
# =========================================================
answers_all = dep_answers + anx_answers + str_answers
expected_len = len(answers_all)

art = load_artifacts()
feature_cols = art["feature_cols"]
models = art["models"]
scaler = art["scaler"]

def _norm(s: str) -> str:
    return " ".join(str(s).split()).strip()

if not feature_cols or not isinstance(feature_cols, list) or len(feature_cols) != expected_len:
    st.error(f"feature_columns.json tidak valid / jumlah fitur tidak sesuai (harus {expected_len}).")
    st.stop()

ui_questions = depression_qs + anxiety_qs + stress_qs
answer_map = {_norm(q): float(v) for q, v in zip(ui_questions, answers_all)}

values = []
missing = []
for col in feature_cols:
    k = _norm(col)
    if k in answer_map:
        values.append(answer_map[k])
    else:
        values.append(0.0)
        missing.append(col)

if missing:
    st.warning(
        "Ada fitur training yang tidak match dengan pertanyaan UI. Nilai fitur tsb dipaksa 0.\n"
        + "\n".join([f"- {m}" for m in missing[:10]])
        + ("\n- ..." if len(missing) > 10 else "")
    )

X_df = pd.DataFrame([values], columns=feature_cols)

# =========================================================
# Label & DASS rule-based
# =========================================================
LEVELS_ID = ["Normal", "Ringan", "Sedang", "Berat", "Sangat Berat"]
LEVEL_KEY = ["normal", "mild", "moderate", "severe", "extreme"]

# Mapping key -> label (buat kategori DASS rule-based)
RULE_LABEL = dict(zip(LEVEL_KEY, LEVELS_ID))

def dass21_score(domain_answers):
    return int(sum(domain_answers) * 2)

def dass_level_from_score(domain, score):
    domain = domain.lower()
    if domain == "dep":
        if score <= 9: return 0
        elif score <= 13: return 1
        elif score <= 20: return 2
        elif score <= 27: return 3
        else: return 4
    if domain == "anx":
        if score <= 7: return 0
        elif score <= 9: return 1
        elif score <= 14: return 2
        elif score <= 19: return 3
        else: return 4
    if domain == "str":
        if score <= 14: return 0
        elif score <= 18: return 1
        elif score <= 25: return 2
        elif score <= 33: return 3
        else: return 4
    return 0

def predict_domain(domain_key, X_df, answers_subset):
    # 1) DASS rule-based (utama)
    score = dass21_score(answers_subset)
    pred_rule = int(np.clip(dass_level_from_score(domain_key, score), 0, 4))
    rule_key = LEVEL_KEY[pred_rule]

    # 2) ML (sekunder): PRIORITAS LR (RF cuma fallback)
    dom = models.get(domain_key, {})
    model = dom.get("lr")
    is_pipe = dom.get("lr_pipe", False)

    if model is None:
        # fallback RF kalau LR tidak ada
        model = dom.get("rf")
        is_pipe = dom.get("rf_pipe", False)
        if model is not None:
            st.warning(f"[INFO] LR untuk '{domain_key}' tidak ditemukan, memakai RF sebagai fallback.")
        else:
            # no model at all
            return "—", "normal", None, False, score, rule_key

    X_in = X_df if is_pipe else align_and_scale_input(X_df, scaler)
    pred_ml, conf = model_pred_and_conf(model, X_in)

    # fallback kalau model gagal
    if pred_ml is None:
        pred_ml = pred_rule
        conf = 0.20

    # guard confidence (tampilan)
    try:
        conf, adjusted = guard_overconfidence(conf, answers_subset, pred_class=pred_ml)
    except TypeError:
        conf, adjusted = guard_overconfidence(conf, answers_subset), False

    conf = float(np.clip(conf, 0.20, 0.99))
    pct = int(round(conf * 100))

    pred_ml = int(np.clip(pred_ml, 0, 4))
    level_id = LEVELS_ID[pred_ml]
    level_key = LEVEL_KEY[pred_ml]

    return level_id, level_key, pct, adjusted, score, rule_key

# =========================================================
# Prediksi (sekali saja)
# =========================================================
sev_dep_id, sev_dep_key, pct_dep, adj_dep, score_dep, rule_dep_key = predict_domain("dep", X_df, dep_answers)
sev_anx_id, sev_anx_key, pct_anx, adj_anx, score_anx, rule_anx_key = predict_domain("anx", X_df, anx_answers)
sev_str_id, sev_str_key, pct_str, adj_str, score_str, rule_str_key = predict_domain("str", X_df, str_answers)

top_dep = top_contributors_text(depression_qs, dep_answers)
top_anx = top_contributors_text(anxiety_qs, anx_answers)
top_str = top_contributors_text(stress_qs, str_answers)


# =========================================================
# Results
# =================================================
st.markdown("<h2 style='text-align:center; margin-top:6px;'>Ringkasan Hasil Prediksi</h2>", unsafe_allow_html=True)

def confidence_label(pct):
    if pct is None:
        return "—"
    pct = int(pct)
    if pct < 40:
        return "Rendah"
    if pct < 60:
        return "Sedang"
    if pct < 70:
        return "Tinggi"
    return "Sangat Tinggi"

def render_result_card(title, pct, score, rule_key, top_items, rekom):
    pct_int = 0 if pct is None else int(pct)
    score_int = 0 if score is None else int(score)

    dass_label = RULE_LABEL.get(rule_key, "—")
    conf_lbl = confidence_label(pct_int)
    bar_color = severity_bar_color(rule_key)
    w = max(5, min(95, pct_int))

    if top_items:
        bullets = "".join([f"<li>{x}</li>" for x in top_items[:3]])
    else:
        bullets = "<li>(Tidak ada faktor dominan)</li>"

    html = f"""
<div class="wf-card">
  <div class="wf-title">{title}</div>

  <div class="wf-main">
    <div class="wf-left">
      <div class="wf-pct">{pct_int}%</div>
      <div class="wf-sub-left">Probabilitas  model ( confidence )</div>
    </div>

    <div class="wf-right">
      <div class="wf-pill">Kategori DASS-21 : <b>{dass_label}</b></div>
      <div class="wf-sub-right">Tingkat Keyakinan Model : <b>{conf_lbl}</b></div>
      <div class="wf-sub-right">Skor DASS-21 : <b>{score_int}</b></div>
    </div>
  </div>

  <div class="wf-bar-wrap">
    <div class="wf-bar" style="width:{w}%; background:{bar_color};"></div>
  </div>

  <div class="wf-text">
    <b>Faktor utama ( berdasarkan jawaban dengan skor tertinggi ):</b>
    <ul>{bullets}</ul>
  </div>

  <div class="wf-text wf-rekom">
    <b>Rekomendasi singkat:</b><br>
    {rekom.replace(chr(10), "<br>")}
  </div>
</div>
""".strip()

    # Ini kuncinya: hilangkan indent di AWAL tiap baris -> nggak kebaca code block lagi
    html = "\n".join(line.lstrip() for line in html.splitlines())

    st.markdown(html, unsafe_allow_html=True)
# =========================================================
# PANGGIL  CARD =========================================================

render_result_card(
    "Depression",
    pct_dep, score_dep, rule_dep_key,
    top_dep, recommendation_text("Depression", rule_dep_key)
)
st.markdown("<div class='card-spacer'></div>", unsafe_allow_html=True)

render_result_card(
    "Anxiety",
    pct_anx, score_anx, rule_anx_key,
    top_anx, recommendation_text("Anxiety", rule_anx_key)
)
st.markdown("<div class='card-spacer'></div>", unsafe_allow_html=True)

render_result_card(
    "Stress",
    pct_str, score_str, rule_str_key,
    top_str, recommendation_text("Stress", rule_str_key)
)

st.markdown("---")

# =========================================================
# Pie chart
# =========================================================
st.markdown("<h2 style='text-align:center; margin-top:6px;'>Visualisasi : Perbandingan Keyakinan Model per Domain</h2>", unsafe_allow_html=True)

labels = ["Depression", "Anxiety", "Stress"]
values = [
    0 if pct_dep is None else int(pct_dep),
    0 if pct_anx is None else int(pct_anx),
    0 if pct_str is None else int(pct_str),
]
colors = ['#FFA4A4', '#BADFDB', '#D9E9CF']

if sum(values) == 0:
    st.info("Visualisasi tidak ditampilkan karena semua nilai keyakinan = 0%.")
else:
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(color='rgba(255,255,255,0.35)', width=1)
            ),
            hole=0.55,
            textinfo='label+value',
            insidetextorientation='radial'
        )]
    )

    fig.update_layout(
        showlegend=False,
        height=360,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    colL, colM, colR = st.columns([1, 2, 1])
    with colM:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")



# NOTE BOX (ISI DIBIARKAN PERSIS seperti punya kamu)
st.markdown("""
<div class="note-box">
  <b>Catatan :</b><br>
  <b>Kategori DASS-21 (skor)</b> = Kategori berdasarkan <i>cut-off</i> dari skor DASS-21 (Normal – Sangat Berat).<br><br>
  <b>Prediksi Model (ML)</b> = Kategori yang diprediksi oleh model machine learning dari jawaban DASS-21.<br><br>
  <b>Keyakinan model (confidence)</b> = Tingkat keyakinan model terhadap prediksinya. Ini <b>bukan</b> tingkat keparahan klinis dan <b>bukan</b> probabilitas risiko klinis (kecuali dilakukan kalibrasi khusus).<br><br>
  <b>Penting:</b> Hasil ini <b>bukan diagnosis klinis</b>. Jika gejala mengganggu aktivitas harian, pertimbangkan screening lanjutan atau konsultasi dengan profesional kesehatan mental.
</div>
""", unsafe_allow_html=True)
