import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────
BASE_DIR  = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization"
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load thresholds from config ────────────────────────────
_cfg_path = os.path.join(DATA_DIR, "dashboard_config.csv")
if os.path.exists(_cfg_path):
    _cfg = pd.read_csv(_cfg_path)
    P50  = float(_cfg["P50"].iloc[0])
    P85  = float(_cfg["P85"].iloc[0])
else:
    P50 = 0.2183
    P85 = 0.3027

# ── Load feature names ─────────────────────────────────────
_feat_path = os.path.join(MODEL_DIR, "feature_names.pkl")
if os.path.exists(_feat_path):
    with open(_feat_path, "rb") as _f:
        FEATURE_NAMES = pickle.load(_f)
else:
    FEATURE_NAMES = [
        "admission_type_id","discharge_disposition_id","admission_source_id",
        "time_in_hospital","num_lab_procedures","num_procedures","num_medications",
        "number_outpatient","number_emergency","number_inpatient","number_diagnoses",
        "age_num","total_prior_visits","flag_circulatory","flag_respiratory",
        "flag_diabetes","flag_chronic","med_changed","num_active_meds",
        "diag_1_cat","diag_2_cat","diag_3_cat","gender_enc","diabetesMed_enc",
        "race_Asian","race_Caucasian","race_Hispanic","race_Other",
    ]

# ── Lookup maps ────────────────────────────────────────────
AGE_MAP = {
    "[0-10)":1,"[10-20)":2,"[20-30)":3,"[30-40)":4,"[40-50)":5,
    "[50-60)":6,"[60-70)":7,"[70-80)":8,"[80-90)":9,"[90-100)":10
}
DIAG_MAP = {
    "Circulatory":1,"Respiratory":2,"Digestive":3,"Diabetes":4,
    "Injury":5,"Musculoskeletal":6,"Genitourinary":7,"Neoplasms":8,
    "External/Other":9,"Other":0
}

TEAL  = "#00d4aa"
CORAL = "#e84040"
AMBER = "#f0a500"
BLUE  = "#007acc"

# PAGE CONFIG

st.set_page_config(
    page_title = "MediRisk — Readmission Predictor",
    page_icon  = "⚕️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp            { background: #080f1a; color: #c8d6e5; }
.block-container  { padding: 1.6rem 2.2rem 2rem; max-width: 1320px; }

section[data-testid="stSidebar"] {
    background: #0b1628;
    border-right: 1px solid #1a2d4a;
}

h1,h2,h3 { color: #e8f4ff; font-weight: 600; letter-spacing: -0.02em; }
p, li    { color: #a8bed2; line-height: 1.7; }

[data-testid="metric-container"] {
    background: #0e1e35;
    border: 1px solid #1a3050;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #5a8aaa !important;
    font-size: 12px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00d4aa !important;
    font-family: 'DM Mono', monospace;
    font-size: 26px !important;
}

.stSelectbox > div > div {
    background: #0e1e35 !important;
    border: 1px solid #1a3050 !important;
    border-radius: 8px !important;
    color: #c8d6e5 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #007acc 100%) !important;
    color: #080f1a !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    letter-spacing: 0.02em;
}

[data-testid="stExpander"] {
    background: #0b1628 !important;
    border: 1px solid #1a3050 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #7fa8c8 !important; }

hr { border-color: #1a3050 !important; margin: 1.4rem 0 !important; }

.stInfo    { background:#071d30!important; border-left:4px solid #007acc!important; color:#a8bed2!important; }
.stSuccess { background:#061a14!important; border-left:4px solid #00d4aa!important; color:#a8bed2!important; }
.stWarning { background:#1a1400!important; border-left:4px solid #f0a500!important; color:#a8bed2!important; }
.stError   { background:#1a0808!important; border-left:4px solid #e84040!important; color:#a8bed2!important; }

.risk-card { border-radius:14px; padding:20px 24px; margin:12px 0; border-left:6px solid; }
.risk-high   { background:#1a0e0e; border-color:#e84040; }
.risk-medium { background:#1a1400; border-color:#f0a500; }
.risk-low    { background:#071610; border-color:#00d4aa; }
.risk-title  { font-size:20px; font-weight:700; margin:0 0 4px; }
.risk-high   .risk-title { color:#e84040; }
.risk-medium .risk-title { color:#f0a500; }
.risk-low    .risk-title { color:#00d4aa; }
.risk-body   { color:#8aa8c0; font-size:14px; margin:0; }

.action-pill {
    display:inline-block; background:#0e1e35; border:1px solid #1a3050;
    border-radius:20px; padding:7px 14px; font-size:13px;
    color:#a8d4f0; margin:4px 4px 4px 0;
}
.action-pill-urgent { border-color:#e84040; color:#f08080; background:#1a0e0e; }

.section-badge {
    display:inline-block; background:#00d4aa18; border:1px solid #00d4aa44;
    color:#00d4aa; font-size:11px; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; padding:3px 10px; border-radius:20px; margin-bottom:10px;
}

.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:16px 0; }
.kpi-box  { background:#0e1e35; border:1px solid #1a3050; border-radius:12px; padding:16px 18px; text-align:center; }
.kpi-label { font-size:11px; color:#4a7a9a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
.kpi-value { font-family:'DM Mono',monospace; font-size:22px; font-weight:500; color:#00d4aa; }
.kpi-sub   { font-size:11px; color:#3a6a8a; margin-top:3px; }

.header-brand { font-size:13px; color:#00d4aa; font-family:'DM Mono',monospace; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:4px; }
.header-title { font-size:30px; font-weight:700; color:#e8f4ff; margin:0 0 6px; line-height:1.2; }
.header-sub   { font-size:14px; color:#5a8aaa; margin:0; }

.input-group-label {
    font-size:11px; font-weight:600; letter-spacing:0.14em;
    text-transform:uppercase; color:#00d4aa;
    margin:18px 0 10px; padding-bottom:6px;
    border-bottom:1px solid #1a3050;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme
plt.rcParams.update({
    "figure.facecolor":   "#0e1e35",
    "axes.facecolor":     "#0b1628",
    "axes.edgecolor":     "#1a3050",
    "axes.labelcolor":    "#5a8aaa",
    "axes.titlecolor":    "#a8bed2",
    "text.color":         "#a8bed2",
    "xtick.color":        "#4a7a9a",
    "ytick.color":        "#4a7a9a",
    "grid.color":         "#1a3050",
    "grid.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "font.family":        "sans-serif",
})

def dark_fig(w=6, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0e1e35")
    ax.set_facecolor("#0b1628")
    return fig, ax


# ASSET LOADERS  (cached)

@st.cache_resource(show_spinner=False)
def load_assets():
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    sc_path = os.path.join(MODEL_DIR, "scaler.pkl")
    missing = [p for p in [rf_path, sc_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Model files not found:\n" + "\n".join(missing) +
            f"\n\nExpected folder: {MODEL_DIR}"
        )
    with open(rf_path, "rb") as f:
        mdl = pickle.load(f)
    with open(sc_path, "rb") as f:
        sc  = pickle.load(f)
    return mdl, sc


@st.cache_data(show_spinner=False)
def load_population():
    path = os.path.join(DATA_DIR, "diabetic_dashboard.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dashboard CSV not found: {path}\n"
            "Run generate_dashboard.py first."
        )
    return pd.read_csv(path, low_memory=False)


# ── Attempt asset loading 
LOADED   = False
LOAD_ERR = ""
model = scaler = df_pop = None

try:
    model, scaler = load_assets()
    LOADED = True
except Exception as e:
    LOAD_ERR = str(e)

try:
    df_pop = load_population()
except Exception as e:
    if not LOAD_ERR:
        LOAD_ERR = str(e)


# PREDICTION HELPER

def build_and_predict(inp):
    row = {f: 0.0 for f in FEATURE_NAMES}
    dc  = DIAG_MAP.get(inp["diag"], 0)

    candidates = {
        "admission_type_id"       : float(inp["adm_type"]),
        "discharge_disposition_id": 1.0,
        "admission_source_id"     : 7.0,
        "time_in_hospital"        : float(inp["los"]),
        "num_lab_procedures"      : float(inp["lab"]),
        "num_procedures"          : float(inp["proc"]),
        "num_medications"         : float(inp["meds"]),
        "number_outpatient"       : float(inp["n_out"]),
        "number_emergency"        : float(inp["n_em"]),
        "number_inpatient"        : float(inp["n_in"]),
        "number_diagnoses"        : float(inp["n_diag"]),
        "age_num"                 : float(AGE_MAP.get(inp["age"], 7)),
        "total_prior_visits"      : float(inp["n_out"] + inp["n_em"] + inp["n_in"]),
        "flag_circulatory"        : 1.0 if inp["diag"] == "Circulatory"  else 0.0,
        "flag_respiratory"        : 1.0 if inp["diag"] == "Respiratory"  else 0.0,
        "flag_diabetes"           : 1.0 if inp["diag"] == "Diabetes"     else 0.0,
        "flag_chronic"            : 1.0 if inp["diag"] in ["Circulatory","Respiratory","Diabetes"] else 0.0,
        "med_changed"             : 0.0,
        "num_active_meds"         : 1.0 if inp["insulin"] in ["Steady","Up","Down"] else 0.0,
        "diag_1_cat"              : float(dc),
        "diag_2_cat"              : float(dc),
        "diag_3_cat"              : 0.0,
        "gender_enc"              : 1.0 if inp["gender"]  == "Male" else 0.0,
        "diabetesMed_enc"         : 1.0 if inp["diab_med"]== "Yes"  else 0.0,
        "diabetesmed"             : 1.0 if inp["diab_med"]== "Yes"  else 0.0,
        "race_Asian"              : 1.0 if inp["race"] == "Asian"      else 0.0,
        "race_Caucasian"          : 1.0 if inp["race"] == "Caucasian"  else 0.0,
        "race_Hispanic"           : 1.0 if inp["race"] == "Hispanic"   else 0.0,
        "race_Other"              : 1.0 if inp["race"] == "Other"      else 0.0,
        "race_Unknown"            : 1.0 if inp["race"] == "Unknown"    else 0.0,
        "race_AfricanAmerican"    : 1.0 if inp["race"] == "AfricanAmerican" else 0.0,
    }
    for k, v in candidates.items():
        if k in row:
            row[k] = v

    X    = pd.DataFrame([[row[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    Xv   = X.values.astype(float)

    # Detect if scaler is needed
    try:
        val_std = float(Xv.std())
        X_sc = scaler.transform(Xv) if val_std > 0.1 else Xv
    except Exception:
        X_sc = Xv

    prob = float(model.predict_proba(X_sc)[0][1])

    if prob >= P85:
        tier, css, emoji = "HIGH RISK",   "risk-high",   "🔴"
    elif prob >= P50:
        tier, css, emoji = "MEDIUM RISK", "risk-medium", "🟡"
    else:
        tier, css, emoji = "LOW RISK",    "risk-low",    "🟢"

    return prob, tier, css, emoji

# SIDEBAR

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:18px 0 10px'>
      <div style='font-size:36px'>⚕️</div>
      <div style='font-family:DM Mono,monospace;font-size:11px;
                  color:#00d4aa;letter-spacing:.2em;margin-top:6px'>
        MEDIRISK
      </div>
      <div style='font-size:12px;color:#3a6a8a;margin-top:4px'>
        Readmission Predictor v1.0
      </div>
    </div>
    <hr style='border-color:#1a3050;margin:10px 0 16px'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["⚕️  Risk Predictor", "📊  Population Analytics", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1a3050;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:12px;color:#3a6a8a;line-height:2.0'>
      <b style='color:#5a8aaa'>Model</b><br>
      Random Forest · 200 trees<br><br>
      <b style='color:#5a8aaa'>Dataset</b><br>
      101,766 encounters · 130 US hospitals<br><br>
      <b style='color:#5a8aaa'>Features</b><br>
      {len(FEATURE_NAMES)} engineered variables<br><br>
      <b style='color:#5a8aaa'>Risk Thresholds</b><br>
      🔴 High &nbsp;&nbsp; ≥ {P85:.3f}<br>
      🟡 Medium ≥ {P50:.3f}<br>
      🟢 Low &nbsp;&nbsp;&lt; {P50:.3f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if LOADED:
        st.success("✅  Model loaded")
    else:
        st.error("❌  Model not found")


# PAGE 1 — RISK PREDICTOR

if page == "⚕️  Risk Predictor":

    st.markdown("""
    <div style='margin-bottom:22px'>
      <div class='header-brand'>⚕️ MEDIRISK CLINICAL TOOL</div>
      <h1 class='header-title'>30-Day Readmission Risk Predictor</h1>
      <p class='header-sub'>
        Enter patient details to compute readmission probability
        and generate a personalised care action plan.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not LOADED:
        st.error(
            f"**Model files not found.**\n\n"
            f"Expected directory: `{MODEL_DIR}`\n\n"
            f"Files needed:\n"
            f"- `random_forest.pkl`\n"
            f"- `scaler.pkl`\n\n"
            f"Error detail: {LOAD_ERR}"
        )
        st.stop()

    # ── INPUT FORM 
    with st.form("predict_form"):
        col1, col2, col3 = st.columns([1, 1, 1], gap="large")

        with col1:
            st.markdown('<div class="input-group-label">👤 Demographics</div>',
                        unsafe_allow_html=True)
            age      = st.selectbox("Age group", list(AGE_MAP.keys()), index=6)
            gender   = st.selectbox("Gender", ["Female", "Male"])
            race     = st.selectbox("Race / ethnicity",
                                    ["Caucasian","AfricanAmerican","Hispanic",
                                     "Asian","Other","Unknown"])
            diab_med = st.selectbox("On diabetes medication?", ["Yes","No"])

            st.markdown('<div class="input-group-label">🔬 Lab & Diagnosis</div>',
                        unsafe_allow_html=True)
            diag    = st.selectbox("Primary diagnosis", list(DIAG_MAP.keys()))
            insulin = st.selectbox("Insulin usage", ["No","Steady","Up","Down"])

        with col2:
            st.markdown('<div class="input-group-label">🏥 Current Admission</div>',
                        unsafe_allow_html=True)
            los     = st.slider("Length of stay (days)",  1,  14,  5)
            n_diag  = st.slider("Number of diagnoses",    1,  16,  7)
            lab     = st.slider("Lab procedures",         1, 132, 44)
            proc    = st.slider("Surgical procedures",    0,   6,  1)
            meds    = st.slider("Medications prescribed", 1,  81, 16)
            adm_lbl = st.selectbox("Admission type",
                                   ["Emergency","Urgent","Elective","Other"])
            adm_map = {"Emergency":1,"Urgent":2,"Elective":3,"Other":5}

        with col3:
            st.markdown('<div class="input-group-label">📋 Prior Visit History</div>',
                        unsafe_allow_html=True)
            n_in  = st.number_input("Prior inpatient visits  (last year)",  0, 21, 0)
            n_em  = st.number_input("Prior emergency visits  (last year)",  0, 76, 0)
            n_out = st.number_input("Prior outpatient visits (last year)",  0, 42, 0)

            total_prior = int(n_in) + int(n_em) + int(n_out)
            st.markdown(f"""
            <div style='background:#071d30;border:1px solid #1a3050;
                        border-radius:10px;padding:14px 16px;margin-top:14px'>
              <div style='font-size:11px;color:#4a7a9a;
                          text-transform:uppercase;letter-spacing:.1em'>
                Total prior visits
              </div>
              <div style='font-family:DM Mono,monospace;font-size:28px;
                          color:#00d4aa;margin-top:4px'>
                {total_prior}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "⚡ Generate Risk Assessment",
                type="primary",
                use_container_width=True,
            )

    # ── RESULTS 
    if submitted:
        inp = dict(
            age=age, gender=gender, race=race, diab_med=diab_med,
            diag=diag, insulin=insulin,
            los=los, n_diag=n_diag, lab=lab, proc=proc, meds=meds,
            adm_type=adm_map[adm_lbl],
            n_in=int(n_in), n_em=int(n_em), n_out=int(n_out),
        )

        try:
            prob, tier, css, emoji = build_and_predict(inp)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        cost_inr = int(prob * 1_250_000)
        st.markdown("---")

        # KPI row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Risk Probability",   f"{prob:.1%}")
        m2.metric("Risk Tier",          tier.split()[0])
        m3.metric("Est. Readmit Cost",  f"Rs.{cost_inr:,}")
        m4.metric("Prior Inpatient",    int(n_in))
        m5.metric("Total Prior Visits", total_prior)

        # Risk banner
        advice_map = {
            "risk-high"  : "Intensive post-discharge intervention required.",
            "risk-medium": "Enhanced follow-up protocol recommended.",
            "risk-low"   : "Standard discharge procedure — continue monitoring.",
        }
        st.markdown(f"""
        <div class='risk-card {css}'>
          <div class='risk-title'>{emoji} {tier}</div>
          <p class='risk-body'>
            30-day readmission probability:
            <strong style='color:#e8f4ff'>{prob:.1%}</strong>
            &nbsp;·&nbsp; {advice_map[css]}
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        fig, ax = dark_fig(w=8, h=1.2)
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")
        ax.barh([0], [1.0],                color="#131f33",   height=0.5, zorder=1)
        ax.barh([0], [P50],                color="#00d4aa22", height=0.5, zorder=2)
        ax.barh([0], [P85 - P50], left=P50, color="#f0a50022", height=0.5, zorder=2)
        ax.barh([0], [1.0 - P85], left=P85, color="#e8404022", height=0.5, zorder=2)
        fill = CORAL if prob >= P85 else AMBER if prob >= P50 else TEAL
        ax.barh([0], [min(prob, 1.0)], color=fill, height=0.5, zorder=3, alpha=0.9)
        for tv, tl in [(P50, f"Medium {P50:.2f}"), (P85, f"High {P85:.2f}")]:
            ax.axvline(tv, color="#2a4060", linewidth=1.5, zorder=4)
            ax.text(tv, 0.44, tl, ha="center", va="bottom", fontsize=8, color="#4a7a9a")
        px = min(prob, 0.99)
        ax.scatter([px], [0], s=180, color=fill, zorder=5,
                   linewidths=2, edgecolors="#0b1628")
        ax.text(px, -0.44, f"{prob:.1%}", ha="center", va="top",
                fontsize=11, fontweight="bold", color=fill)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, 0.75)
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Action plan + Feature importance
        left, right = st.columns([1.1, 0.9], gap="large")

        with left:
            st.markdown("#### 🏥 Clinical Action Plan")
            if prob >= P85:
                urgency = "⚠️ Intensive intervention required"
                actions = [
                    ("urgent", "📞 24-hour post-discharge nurse phone call"),
                    ("urgent", "👩‍⚕️ Diabetes nurse educator consult before discharge"),
                    ("urgent", "💊 Pharmacist medication reconciliation"),
                    ("urgent", "🏠 Screen: food security, transport, housing"),
                    ("urgent", "📅 7-day follow-up with primary care"),
                    ("",       "📡 Enrol in remote glucose monitoring"),
                    ("",       "📄 Written emergency action plan provided"),
                ]
            elif prob >= P50:
                urgency = "ℹ️ Enhanced follow-up recommended"
                actions = [
                    ("urgent", "📞 72-hour follow-up phone call"),
                    ("",       "📝 Written diabetes self-management education"),
                    ("",       "💊 Confirm 30-day medication supply"),
                    ("",       "📅 14-day outpatient appointment"),
                    ("",       "📋 Share care summary with caregiver"),
                ]
            else:
                urgency = "✅ Standard discharge protocol"
                actions = [
                    ("", "📄 Standard discharge instructions"),
                    ("", "📞 Provide emergency contact number"),
                    ("", "📅 Routine 30-day outpatient follow-up"),
                    ("", "💊 Confirm medication schedule understood"),
                ]

            st.markdown(
                f"<p style='color:#7fa8c8;font-size:13px;margin-bottom:10px'>"
                f"{urgency}</p>",
                unsafe_allow_html=True,
            )
            pills = "".join(
                f'<span class="action-pill '
                f'{"action-pill-urgent" if k == "urgent" else ""}">'
                f'{t}</span>'
                for k, t in actions
            )
            st.markdown(f"<div style='line-height:2.4'>{pills}</div>",
                        unsafe_allow_html=True)

        with right:
            st.markdown("#### 📊 Top Risk Drivers")
            if hasattr(model, "feature_importances_"):
                fi   = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
                top8 = fi.sort_values(ascending=False).head(8).sort_values()
                lbls = [f.replace("_"," ").replace("num ","# ").title()
                        for f in top8.index]
                fig2, ax2 = dark_fig(5, 3.2)
                n     = len(top8)
                clrs  = [CORAL if i >= n-2 else BLUE if i >= n-5 else TEAL
                         for i in range(n)]
                bars  = ax2.barh(lbls, top8.values, color=clrs, height=0.62)
                for bar, v in zip(bars, top8.values):
                    ax2.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                             f"{v:.3f}", va="center", fontsize=8, color="#5a8aaa")
                ax2.set_xlim(0, top8.values.max() * 1.4)
                ax2.tick_params(labelsize=8)
                ax2.set_xlabel("Importance score", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

        with st.expander("📋 Full Patient Input Summary"):
            sa, sb, sc_ = st.columns(3)
            with sa:
                st.markdown(f"**Age group:** {age}")
                st.markdown(f"**Gender:** {gender}")
                st.markdown(f"**Race:** {race}")
                st.markdown(f"**Diagnosis:** {diag}")
                st.markdown(f"**Insulin:** {insulin}")
            with sb:
                st.markdown(f"**Length of stay:** {los} days")
                st.markdown(f"**Lab procedures:** {lab}")
                st.markdown(f"**Procedures:** {proc}")
                st.markdown(f"**Medications:** {meds}")
                st.markdown(f"**Admission type:** {adm_lbl}")
            with sc_:
                st.markdown(f"**Prior inpatient:** {int(n_in)}")
                st.markdown(f"**Prior emergency:** {int(n_em)}")
                st.markdown(f"**Prior outpatient:** {int(n_out)}")
                st.markdown(f"**Total prior visits:** {total_prior}")
                st.markdown(f"**Diabetes medication:** {diab_med}")


# PAGE 2 — POPULATION ANALYTICS

elif page == "📊  Population Analytics":

    st.markdown("""
    <div style='margin-bottom:22px'>
      <div class='header-brand'>📊 POPULATION ANALYTICS</div>
      <h1 class='header-title'>Readmission Insights Dashboard</h1>
      <p class='header-sub'>
        Explore readmission patterns across 101,766 diabetic patient encounters.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if df_pop is None:
        st.error(
            f"**Dashboard data not found.**\n\n"
            f"Expected: `{os.path.join(DATA_DIR, 'diabetic_dashboard.csv')}`\n\n"
            "Run `python generate_dashboard.py` first, then restart the app."
        )
        st.stop()

    df = df_pop.copy()

    # Validate columns
    needed = ["readmitted_30d","risk_score","risk_tier","age_group",
              "gender","diag_category","time_in_hospital",
              "number_diagnoses","readmission_cost_inr"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        st.error(f"Missing columns in dashboard CSV: {miss}\nRe-run generate_dashboard.py")
        st.stop()

    # Filters
    with st.expander("🔧 Filter Population", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        age_order = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                     "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]

        all_genders = sorted(df["gender"].dropna().unique().tolist())
        all_tiers   = [t for t in ["High","Medium","Low"]
                       if t in df["risk_tier"].unique()]
        all_diags   = sorted(df["diag_category"].dropna().unique().tolist())
        all_ages    = [a for a in age_order if a in df["age_group"].unique()]

        gsel = f1.multiselect("Gender",    all_genders, default=all_genders)
        tsel = f2.multiselect("Risk tier", all_tiers,   default=all_tiers)
        dsel = f3.multiselect("Diagnosis", all_diags,   default=all_diags)
        asel = f4.multiselect("Age group", all_ages,    default=all_ages)

    df = df[
        df["gender"].isin(gsel) &
        df["risk_tier"].isin(tsel) &
        df["diag_category"].isin(dsel) &
        df["age_group"].isin(asel)
    ]

    if df.empty:
        st.warning("No data matches your filters — adjust the filter panel.")
        st.stop()

    total  = len(df)
    r_cnt  = int(df["readmitted_30d"].sum())
    r_rate = df["readmitted_30d"].mean() * 100
    h_risk = int((df["risk_tier"] == "High").sum())
    t_cost = df["readmission_cost_inr"].sum()
    avg_rs = df["risk_score"].mean()

    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi-box'>
        <div class='kpi-label'>Total Patients</div>
        <div class='kpi-value'>{total:,}</div>
        <div class='kpi-sub'>filtered cohort</div>
      </div>
      <div class='kpi-box'>
        <div class='kpi-label'>30-Day Readmissions</div>
        <div class='kpi-value'>{r_cnt:,}</div>
        <div class='kpi-sub'>{r_rate:.1f}% rate</div>
      </div>
      <div class='kpi-box'>
        <div class='kpi-label'>High-Risk Patients</div>
        <div class='kpi-value'>{h_risk:,}</div>
        <div class='kpi-sub'>{h_risk/total*100:.1f}% of cohort</div>
      </div>
      <div class='kpi-box'>
        <div class='kpi-label'>Total Cost (INR)</div>
        <div class='kpi-value'>Rs.{t_cost/1e6:.1f}M</div>
        <div class='kpi-sub'>readmission cost</div>
      </div>
      <div class='kpi-box'>
        <div class='kpi-label'>Avg Risk Score</div>
        <div class='kpi-value'>{avg_rs:.3f}</div>
        <div class='kpi-sub'>model probability</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Row 1: Age + Diagnosis
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="section-badge">AGE ANALYSIS</div>',
                    unsafe_allow_html=True)
        st.markdown("**Readmission Rate by Age Group**")
        ao    = [a for a in all_ages if a in df["age_group"].unique()]
        age_r = df.groupby("age_group")["readmitted_30d"].mean().reindex(ao) * 100
        age_n = df.groupby("age_group").size().reindex(ao)
        fig, ax = dark_fig(6, 3.4)
        mx   = age_r.max() if not age_r.empty else 1
        bc   = [CORAL if v == mx else BLUE for v in age_r.values]
        bars = ax.bar(range(len(age_r)), age_r.values, color=bc, width=0.65, zorder=2)
        ax.grid(axis="y", zorder=1)
        ax2b = ax.twinx()
        ax2b.plot(range(len(age_n)), age_n.values, color="#4a7a9a",
                  marker="o", markersize=4, linewidth=1.4, linestyle="--")
        ax2b.set_ylabel("Patient count", fontsize=9, color="#4a7a9a")
        ax2b.tick_params(axis="y", labelcolor="#4a7a9a", labelsize=8)
        for bar, v in zip(bars, age_r.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.15,
                    f"{v:.1f}%", ha="center", fontsize=7.5, color="#7fa8c8")
        ax.set_xticks(range(len(age_r)))
        ax.set_xticklabels(age_r.index, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Readmission rate (%)", fontsize=9)
        ax.set_ylim(0, max(age_r.max() * 1.35, 1))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-badge">DIAGNOSIS ANALYSIS</div>',
                    unsafe_allow_html=True)
        st.markdown("**Readmission Rate by Diagnosis**")
        d_rate = df.groupby("diag_category")["readmitted_30d"].mean().sort_values() * 100
        d_cnt  = df.groupby("diag_category").size().reindex(d_rate.index)
        fig, ax = dark_fig(6, 3.4)
        n    = len(d_rate)
        bc_d = [CORAL if i >= n-1 else AMBER if i >= n-3 else TEAL for i in range(n)]
        ax.barh(d_rate.index, d_rate.values, color=bc_d, height=0.62, zorder=2)
        ax.grid(axis="x", zorder=1)
        for i, (v, cnt) in enumerate(zip(d_rate.values, d_cnt.values)):
            ax.text(v + 0.1, i, f"{v:.1f}%  (n={cnt:,})",
                    va="center", fontsize=7.5, color="#5a8aaa")
        ax.set_xlabel("30-day readmission rate (%)", fontsize=9)
        ax.set_xlim(0, max(d_rate.max() * 1.6, 1))
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Row 2: Donut + LOS + Gender
    c3, c4, c5 = st.columns(3, gap="large")

    with c3:
        st.markdown('<div class="section-badge">RISK TIERS</div>',
                    unsafe_allow_html=True)
        st.markdown("**Patient Risk Distribution**")
        tier_vc = df["risk_tier"].value_counts().reindex(
            ["High","Medium","Low"]).fillna(0)
        valid_t = tier_vc[tier_vc > 0]
        cmap    = {"High":CORAL,"Medium":AMBER,"Low":TEAL}
        fig, ax = dark_fig(4, 3.2)
        _, _, ats = ax.pie(
            valid_t.values, labels=None,
            colors=[cmap[t] for t in valid_t.index],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.52, edgecolor="#0e1e35", linewidth=2),
            pctdistance=0.75,
        )
        for at in ats:
            at.set_fontsize(9)
            at.set_color("#e8f4ff")
        ax.text(0, 0, f"{total:,}\npatients",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#a8bed2")
        ax.legend(valid_t.index.tolist(), loc="lower center", ncol=3,
                  fontsize=8, frameon=False, bbox_to_anchor=(0.5,-0.05),
                  labelcolor=[cmap[t] for t in valid_t.index])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c4:
        st.markdown('<div class="section-badge">LENGTH OF STAY</div>',
                    unsafe_allow_html=True)
        st.markdown("**Rate vs Length of Stay**")
        los_r = df.groupby("time_in_hospital")["readmitted_30d"].mean() * 100
        fig, ax = dark_fig(4, 3.2)
        ax.plot(los_r.index, los_r.values, color=TEAL,
                marker="o", markersize=5, linewidth=2, zorder=3)
        ax.fill_between(los_r.index, los_r.values,
                        alpha=0.1, color=TEAL, zorder=2)
        ax.grid(zorder=1)
        ax.set_xlabel("Days in hospital", fontsize=9)
        ax.set_ylabel("Readmission rate (%)", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c5:
        st.markdown('<div class="section-badge">GENDER</div>',
                    unsafe_allow_html=True)
        st.markdown("**Readmission Rate by Gender**")
        g_rate = df.groupby("gender")["readmitted_30d"].mean() * 100
        g_cnt  = df.groupby("gender").size()
        fig, ax = dark_fig(4, 3.2)
        glist  = g_rate.index.tolist()
        gc     = [CORAL if g == "Female" else BLUE for g in glist]
        bars   = ax.bar(glist, g_rate.values, color=gc, width=0.45, zorder=2)
        ax.grid(axis="y", zorder=1)
        for bar, v, g in zip(bars, g_rate.values, glist):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.12,
                    f"{v:.1f}%\nn={g_cnt[g]:,}",
                    ha="center", fontsize=9, color="#7fa8c8")
        ax.set_ylabel("Readmission rate (%)", fontsize=9)
        ax.set_ylim(0, max(g_rate.max() * 1.4, 1))
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Row 3: Score distribution + High-risk table
    st.markdown("---")
    h1, h2 = st.columns([1, 1], gap="large")

    with h1:
        st.markdown('<div class="section-badge">SCORE DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        st.markdown("**Risk Score: Actual vs Predicted**")
        fig, ax = dark_fig(6, 3.2)
        ax.hist(df[df["readmitted_30d"]==0]["risk_score"], bins=35,
                alpha=0.7, color=TEAL, label="Not readmitted <30d",
                density=True, zorder=2)
        pos_df = df[df["readmitted_30d"]==1]["risk_score"]
        if len(pos_df) > 0:
            ax.hist(pos_df, bins=35, alpha=0.7, color=CORAL,
                    label="Readmitted <30d", density=True, zorder=3)
        ax.axvline(P50, color=AMBER, linewidth=1.5, linestyle="--",
                   label=f"Medium ({P50:.2f})", zorder=4)
        ax.axvline(P85, color=CORAL, linewidth=1.5, linestyle=":",
                   label=f"High ({P85:.2f})", zorder=4)
        ax.set_xlabel("Risk probability score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7.5, framealpha=0.2, labelcolor="#a8bed2")
        ax.grid(zorder=1)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with h2:
        st.markdown('<div class="section-badge">HIGH-RISK COHORT</div>',
                    unsafe_allow_html=True)
        st.markdown("**Top 25 Highest-Risk Patients**")
        show = [c for c in ["age_group","gender","diag_category",
                             "time_in_hospital","number_diagnoses",
                             "risk_score","risk_tier","readmitted_30d"]
                if c in df.columns]
        top25 = (df[df["risk_tier"]=="High"][show]
                   .sort_values("risk_score", ascending=False)
                   .head(25)
                   .reset_index(drop=True))
        rmap = {
            "age_group":"Age","gender":"Gender","diag_category":"Diagnosis",
            "time_in_hospital":"LOS","number_diagnoses":"Dx",
            "risk_score":"Risk Score","risk_tier":"Tier",
            "readmitted_30d":"Actual <30d",
        }
        top25 = top25.rename(columns={k:v for k,v in rmap.items() if k in top25.columns})
        if "Risk Score" in top25.columns:
            st.dataframe(
                top25.style
                     .format({"Risk Score": "{:.3f}"})
                     .background_gradient(subset=["Risk Score"], cmap="Reds"),
                use_container_width=True, height=240,
            )
        else:
            st.dataframe(top25, use_container_width=True, height=240)


# PAGE 3 — ABOUT

elif page == "ℹ️  About":

    st.markdown("""
    <div style='margin-bottom:22px'>
      <div class='header-brand'>ℹ️ DOCUMENTATION</div>
      <h1 class='header-title'>About MediRisk</h1>
      <p class='header-sub'>Technical details, model info, and dataset documentation.</p>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🩺 How to Use", "🤖 Model Details", "📁 Dataset & Disclaimer"])

    with t1:
        st.markdown(f"""
### Using the Risk Predictor
1. Go to **⚕️ Risk Predictor** in the sidebar
2. Fill in Demographics, Current Admission, and Prior History
3. Click **⚡ Generate Risk Assessment**
4. Review the risk score, gauge bar, and clinical action plan

### Risk Tier Definitions

| Tier | Threshold | Action |
|------|-----------|--------|
| 🔴 High   | ≥ {P85:.3f} | Intensive intervention — 24hr call, nurse educator, pharmacist review |
| 🟡 Medium | {P50:.3f} – {P85:.3f} | Enhanced follow-up — 72hr call, 14-day appointment |
| 🟢 Low    | < {P50:.3f} | Standard discharge — routine 30-day follow-up |

### Using the Population Dashboard
- Open **📊 Population Analytics**
- Use the **Filter Population** panel to slice by gender, tier, diagnosis, and age
- All charts and KPI cards update instantly
        """)

    with t2:
        st.markdown(f"""
### Machine Learning Model

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Estimators | 200 trees |
| Max depth | 12 |
| Min samples leaf | 20 |
| Class handling | `class_weight='balanced'` |
| Features | {len(FEATURE_NAMES)} engineered variables |
| Metric focus | **Recall** — minimise missed high-risk patients |

### Feature List
{", ".join(f"`{f}`" for f in FEATURE_NAMES)}

### Risk Thresholds
- Medium : ≥ **{P50:.4f}** (33rd percentile of scores)
- High   : ≥ **{P85:.4f}** (66th percentile of scores)
        """)

    with t3:
        st.markdown("""
### Dataset
**Diabetes 130-US Hospitals (1999–2008)**
- Source : UCI Machine Learning Repository
- Records: 101,766 patient encounters across 130 US hospitals
- Target : 30-day readmission (11.2% positive rate)

### Pipeline Steps
1. Data cleaning & imputation
2. Feature engineering (ICD-9 mapping, medication flags, visit history)
3. Target encoding (`readmitted` → binary 0/1)
4. Train/test split (80/20 stratified)
5. StandardScaler (fit on train only)
6. Random Forest training (`class_weight='balanced'`)
7. Risk scoring & tier assignment
8. Cost optimisation analysis

### ⚠️ Disclaimer
> This tool is for **research and educational purposes only**.
> It is **not** a substitute for clinical judgment or medical advice.
> All predictions must be reviewed by a qualified healthcare professional.
> The model was trained on data from 1999–2008 and may not reflect current practices.
        """)

    st.markdown("---")
    ca, cb, cc = st.columns(3)
    ca.info(f"**Tech Stack**\nPython 3.10\nScikit-learn · Pandas\nMatplotlib · Streamlit")
    cb.info(f"**Model path**\n`{MODEL_DIR}`")
    cc.info(f"**Data path**\n`{DATA_DIR}`")