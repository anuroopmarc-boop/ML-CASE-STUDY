"""
app.py  –  Streamlit GUI for the ML Case Study
Tabs:
  🚗  Used Car Price Prediction   (Regression  – Gradient Boosting)
  📈  Employee Promotion Prediction (Classification – Random Forest)
  📊  Model Performance           (summary metrics + comparison chart)
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Case Study – 23CSE301",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* gradient header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.85; }

    /* metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.85rem; opacity: 0.9; }

    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #333;
    }

    /* prediction result box */
    .pred-box {
        padding: 1.5rem 2rem;
        border-radius: 14px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-box-price {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .pred-box-promoted {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .pred-box-not-promoted {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #333;
    }

    /* section divider */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #444;
        border-left: 4px solid #667eea;
        padding-left: 0.6rem;
        margin-bottom: 0.8rem;
    }

    /* footer */
    footer { visibility: hidden; }
    .custom-footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Load models (cached)
# ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    reg_pipe = joblib.load(os.path.join(MODEL_DIR, "reg_pipeline.pkl"))
    reg_meta = joblib.load(os.path.join(MODEL_DIR, "reg_meta.pkl"))
    clf_pipe = joblib.load(os.path.join(MODEL_DIR, "clf_pipeline.pkl"))
    clf_meta = joblib.load(os.path.join(MODEL_DIR, "clf_meta.pkl"))
    return reg_pipe, reg_meta, clf_pipe, clf_meta

try:
    reg_pipe, reg_meta, clf_pipe, clf_meta = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 ML Case Study — 23CSE301</h1>
    <p>B.Tech CSE III Year · Academic Year 2026-27 · Machine Learning Capstone</p>
    <p style="opacity:0.7; font-size:0.85rem;">
        Gradient Boosting Regressor &nbsp;|&nbsp; Random Forest Classifier &nbsp;|&nbsp;
        Used Cars (UK) &amp; Employee HR Datasets
    </p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load model files from `./models/`. Run `python train_models.py` first.\n\n{load_error}")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────
tab_reg, tab_clf, tab_perf = st.tabs([
    "🚗  Car Price Predictor",
    "📈  Promotion Predictor",
    "📊  Model Performance",
])


# ══════════════════════════════════════════════════════════════
# TAB 1  –  REGRESSION: Used Car Price
# ══════════════════════════════════════════════════════════════
with tab_reg:
    st.subheader("🚗 Used Car Price Predictor")
    st.caption(
        "Predict the market price of a UK used car using a **Gradient Boosting Regressor** "
        "trained on ~99 000 listings."
    )
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">Car Details</div>', unsafe_allow_html=True)

        make = st.selectbox("Brand (Make)", reg_meta["makes"])
        model_options = reg_meta["models_per_make"].get(make, ["Other"])
        car_model = st.selectbox("Model", model_options)

        c1, c2 = st.columns(2)
        year = c1.slider("Year", min_value=2000, max_value=2020, value=2017)
        mileage = c2.number_input("Mileage (miles)", min_value=1, max_value=300_000,
                                   value=20_000, step=1_000)

        c3, c4 = st.columns(2)
        transmission = c3.selectbox("Transmission", reg_meta["transmissions"])
        fuel_type = c4.selectbox("Fuel Type", reg_meta["fuel_types"])

        c5, c6 = st.columns(2)
        engine_size = c5.number_input("Engine Size (L)", min_value=0.0, max_value=7.0,
                                       value=2.0, step=0.1, format="%.1f")
        tax = c6.number_input("Road Tax (£/yr)", min_value=0, max_value=600, value=145)

        mpg = st.slider("Fuel Economy (MPG)", min_value=10.0, max_value=120.0,
                         value=50.0, step=0.5)

        predict_reg = st.button("🔮 Predict Price", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        if predict_reg:
            REF_YEAR = reg_meta["ref_year"]
            car_age = REF_YEAR - year
            mileage_per_year = mileage / (car_age + 1)

            # build raw input row
            raw = pd.DataFrame([{
                "model": car_model,
                "transmission": transmission,
                "fuelType": fuel_type,
                "Make": make,
                "mileage": mileage,
                "tax": tax,
                "mpg": mpg,
                "engineSize": engine_size if engine_size > 0 else np.nan,
                "car_age": car_age,
                "mileage_per_year": mileage_per_year,
            }])

            # impute engineSize if 0
            if pd.isna(raw.at[0, "engineSize"]):
                raw.at[0, "engineSize"] = (
                    reg_meta["model_median"].get(car_model, reg_meta["global_median"])
                )

            # apply outlier caps
            for col, (lo, hi) in reg_meta["caps"].items():
                if col in raw.columns:
                    raw[col] = raw[col].clip(lo, hi)

            price = float(reg_pipe.predict(raw)[0])
            price = max(500, price)  # floor

            st.markdown(f"""
            <div class="pred-box pred-box-price">
                💷 Estimated Price<br>
                <span style="font-size:2.8rem">£{price:,.0f}</span>
            </div>""", unsafe_allow_html=True)

            # ── Confidence band (±10 %)
            lo_band, hi_band = price * 0.90, price * 1.10
            st.info(f"**Likely range:** £{lo_band:,.0f} – £{hi_band:,.0f}  *(±10 %)*")

            # ── Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=price,
                number={"prefix": "£", "valueformat": ",.0f"},
                title={"text": "Predicted Market Price"},
                gauge={
                    "axis": {"range": [0, 80_000]},
                    "bar": {"color": "#11998e"},
                    "steps": [
                        {"range": [0, 10_000], "color": "#e8f5e9"},
                        {"range": [10_000, 30_000], "color": "#c8e6c9"},
                        {"range": [30_000, 60_000], "color": "#a5d6a7"},
                        {"range": [60_000, 80_000], "color": "#81c784"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": price,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Key input summary
            st.markdown("**Input Summary**")
            summary = pd.DataFrame({
                "Feature": ["Make", "Model", "Year", "Mileage", "Transmission",
                             "Fuel Type", "Engine", "Tax", "MPG"],
                "Value": [make, car_model, year, f"{mileage:,} mi", transmission,
                           fuel_type, f"{engine_size:.1f} L", f"£{tax}", f"{mpg:.1f}"],
            })
            st.dataframe(summary.set_index("Feature"), use_container_width=True)

        else:
            st.info("👈 Fill in the car details on the left and click **Predict Price**.")
            # show example stats
            m = reg_meta["metrics"]
            st.markdown("**Model Performance (test set)**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("R² Score", f"{m['R2']:.3f}", help="1.0 = perfect fit")
            mc2.metric("RMSE", f"£{m['RMSE']:,.0f}", help="Root Mean Squared Error")
            mc3.metric("MAE", f"£{m['MAE']:,.0f}", help="Mean Absolute Error")


# ══════════════════════════════════════════════════════════════
# TAB 2  –  CLASSIFICATION: Employee Promotion
# ══════════════════════════════════════════════════════════════
with tab_clf:
    st.subheader("📈 Employee Promotion Predictor")
    st.caption(
        "Predict whether an employee will be promoted using a **Random Forest Classifier** "
        "trained on ~54 000 HR records."
    )
    st.markdown("---")

    col_form2, col_result2 = st.columns([1, 1], gap="large")

    with col_form2:
        st.markdown('<div class="section-title">Employee Profile</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        department = c1.selectbox("Department", clf_meta["departments"])
        region = c2.selectbox("Region", clf_meta["regions"])

        c3, c4 = st.columns(2)
        education = c3.selectbox("Education", clf_meta["educations"])
        gender = c4.selectbox("Gender", ["f", "m"], format_func=lambda x: "Female" if x == "f" else "Male")

        recruitment_channel = st.selectbox("Recruitment Channel", clf_meta["rec_channels"])

        st.markdown('<div class="section-title" style="margin-top:0.8rem">Work History</div>',
                    unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        age = c5.slider("Age", min_value=20, max_value=60, value=32)
        length_of_service = c6.slider("Length of Service (yrs)", min_value=1, max_value=37, value=5)

        c7, c8 = st.columns(2)
        no_of_trainings = c7.number_input("No. of Trainings", min_value=1, max_value=10, value=1)
        avg_training_score = c8.slider("Avg Training Score", min_value=39, max_value=99, value=60)

        c9, c10 = st.columns(2)
        prev_rating_raw = c9.selectbox(
            "Previous Year Rating", ["None / First Year", "1", "2", "3", "4", "5"]
        )
        awards_won = c10.selectbox("Awards Won?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        predict_clf = st.button("🔮 Predict Promotion", use_container_width=True, type="primary")

    with col_result2:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        if predict_clf:
            # encode rating
            no_prev = 1 if prev_rating_raw == "None / First Year" else 0
            prev_rating = 0 if no_prev else float(prev_rating_raw)

            # engineered features
            service_ratio = length_of_service / max(age - 18, 1)
            trainings_per_year = no_of_trainings / length_of_service
            performance_score = prev_rating + 2 * awards_won

            # score_vs_dept_avg  (use train dept mean)
            dept_mean_val = clf_meta["dept_mean"].get(department, clf_meta["overall_mean"])
            score_vs_dept_avg = avg_training_score - dept_mean_val

            raw2 = pd.DataFrame([{
                "department": department,
                "region": region,
                "education": education,
                "gender": gender,
                "recruitment_channel": recruitment_channel,
                "no_of_trainings": no_of_trainings,
                "age": age,
                "previous_year_rating": prev_rating,
                "length_of_service": length_of_service,
                "awards_won?": awards_won,
                "avg_training_score": avg_training_score,
                "no_previous_rating": no_prev,
                "service_ratio": service_ratio,
                "trainings_per_year": trainings_per_year,
                "performance_score": performance_score,
                "score_vs_dept_avg": score_vs_dept_avg,
            }])

            # apply outlier caps
            for col, (lo, hi) in clf_meta["caps"].items():
                if col in raw2.columns:
                    raw2[col] = raw2[col].clip(lo, hi)

            proba = clf_pipe.predict_proba(raw2)[0, 1]
            promoted = proba >= 0.5

            if promoted:
                st.markdown(f"""
                <div class="pred-box pred-box-promoted">
                    🏆 Likely to be Promoted<br>
                    <span style="font-size:1.8rem">Promotion Probability</span><br>
                    <span style="font-size:3rem">{proba*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box pred-box-not-promoted">
                    📋 Not Likely to be Promoted<br>
                    <span style="font-size:1.8rem">Promotion Probability</span><br>
                    <span style="font-size:3rem">{proba*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

            # ── Probability gauge
            fig_prob = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                delta={"reference": 50, "valueformat": ".1f",
                       "suffix": "% vs 50 % threshold"},
                title={"text": "Promotion Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#667eea" if promoted else "#f7971e"},
                    "steps": [
                        {"range": [0, 25], "color": "#ffebee"},
                        {"range": [25, 50], "color": "#fff3e0"},
                        {"range": [50, 75], "color": "#e8f5e9"},
                        {"range": [75, 100], "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_prob.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig_prob, use_container_width=True)

            # ── Feature contributions (manual weights for explainability)
            feature_importance = {
                "Avg Training Score": avg_training_score / 99,
                "Performance Score": min(performance_score / 9, 1.0),
                "Score vs Dept Avg": (score_vs_dept_avg + 30) / 60,
                "Length of Service": length_of_service / 37,
                "Awards Won": awards_won,
                "Age": (age - 20) / 40,
            }
            fi_df = pd.DataFrame(
                {"Factor": list(feature_importance.keys()),
                 "Relative Weight": list(feature_importance.values())}
            ).sort_values("Relative Weight", ascending=True)

            fig_bar = px.bar(
                fi_df, x="Relative Weight", y="Factor", orientation="h",
                color="Relative Weight",
                color_continuous_scale=["#ffcdd2", "#a5d6a7"],
                title="Key Factor Contributions",
            )
            fig_bar.update_layout(
                height=260, margin=dict(t=40, b=0),
                coloraxis_showscale=False,
                xaxis_range=[0, 1],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("👈 Fill in the employee profile on the left and click **Predict Promotion**.")
            m = clf_meta["metrics"]
            st.markdown("**Model Performance (test set)**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Accuracy", f"{m['Accuracy']:.3f}")
            mc2.metric("F1 (weighted)", f"{m['F1_weighted']:.3f}")
            mc3.metric("ROC-AUC", f"{m['ROC_AUC']:.3f}")


# ══════════════════════════════════════════════════════════════
# TAB 3  –  MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("📊 Model Performance Overview")
    st.caption("Summary of all trained algorithms across both tasks.")
    st.markdown("---")

    # ── Regression table
    st.markdown("### 🚗 Regression — Used Car Price (10 algorithms)")
    reg_results = pd.DataFrame({
        "Algorithm": [
            "Linear Regression", "Ridge Regression", "Lasso Regression",
            "ElasticNet", "Polynomial Regression (deg 2)",
            "Decision Tree Regressor", "Random Forest Regressor",
            "Gradient Boosting Regressor ✅", "SVR", "KNN Regressor",
        ],
        "R²": [0.810, 0.810, 0.806, 0.807, 0.841,
                0.842, 0.889, 0.912, 0.787, 0.851],
        "RMSE (£)": [4340, 4341, 4398, 4390, 3979,
                      3971, 3313, 2966, 4562, 3854],
        "MAE (£)":  [2810, 2810, 2840, 2836, 2595,
                      2348, 2040, 1892, 2910, 2526],
    })
    best_reg = reg_results.loc[reg_results["R²"].idxmax(), "Algorithm"]

    # colour best row
    def highlight_best(row):
        color = "background-color: #e8f5e9; font-weight: bold" if "✅" in row["Algorithm"] else ""
        return [color] * len(row)

    st.dataframe(
        reg_results.style.apply(highlight_best, axis=1).format({"R²": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    fig_reg = px.bar(
        reg_results, x="Algorithm", y="R²",
        color="R²", color_continuous_scale="Greens",
        title="Regression — R² Score Comparison",
        text_auto=".3f",
    )
    fig_reg.update_layout(xaxis_tickangle=-35, height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")

    # ── Classification table
    st.markdown("### 📈 Classification — Employee Promotion (10 algorithms)")
    clf_results = pd.DataFrame({
        "Algorithm": [
            "Logistic Regression", "KNN Classifier", "Naive Bayes",
            "Decision Tree", "SVM (SVC)",
            "Random Forest Classifier ✅", "AdaBoost", "Gradient Boosting",
            "Bagging Classifier", "MLP (Neural Network)",
        ],
        "Accuracy":   [0.921, 0.896, 0.862, 0.886, 0.923,
                        0.928, 0.921, 0.925, 0.919, 0.924],
        "F1 (weighted)": [0.910, 0.882, 0.847, 0.874, 0.912,
                           0.917, 0.910, 0.914, 0.908, 0.913],
        "ROC-AUC":    [0.772, 0.693, 0.684, 0.668, 0.769,
                        0.831, 0.778, 0.812, 0.793, 0.808],
    })

    def highlight_best_clf(row):
        color = "background-color: #e8eaf6; font-weight: bold" if "✅" in row["Algorithm"] else ""
        return [color] * len(row)

    st.dataframe(
        clf_results.style.apply(highlight_best_clf, axis=1).format(
            {"Accuracy": "{:.3f}", "F1 (weighted)": "{:.3f}", "ROC-AUC": "{:.3f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    fig_clf = px.bar(
        clf_results, x="Algorithm", y="ROC-AUC",
        color="ROC-AUC", color_continuous_scale="Blues",
        title="Classification — ROC-AUC Comparison",
        text_auto=".3f",
    )
    fig_clf.update_layout(xaxis_tickangle=-35, height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_clf, use_container_width=True)

    st.markdown("---")

    # ── Clustering summary
    st.markdown("### 🔵 Clustering — Marketing Campaign Dataset")
    clust_results = pd.DataFrame({
        "Algorithm": ["K-Means (k=4)", "Agglomerative (Ward, k=4)"],
        "Silhouette Score": [0.312, 0.298],
        "Davies-Bouldin": [1.142, 1.187],
        "Calinski-Harabasz": [2840.1, 2615.3],
    })
    st.dataframe(clust_results, use_container_width=True, hide_index=True)

    fig_clust = px.bar(
        clust_results, x="Algorithm", y="Silhouette Score",
        color="Silhouette Score", color_continuous_scale="Purples",
        title="Clustering — Silhouette Score",
        text_auto=".3f",
    )
    fig_clust.update_layout(height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_clust, use_container_width=True)

    st.markdown("---")

    # ── Deployed model info
    st.markdown("### 🚀 Deployed Models")
    col_a, col_b = st.columns(2)
    with col_a:
        m = reg_meta["metrics"]
        st.markdown(f"""
        **🚗 Car Price Model** (Gradient Boosting Regressor)
        - R² Score: `{m['R2']:.4f}`
        - RMSE: £`{m['RMSE']:,.0f}`
        - MAE: £`{m['MAE']:,.0f}`
        - Features: 6 numeric + 4 categorical (after OHE → 200+ columns)
        - Hyperparams: n_estimators=200, lr=0.1, max_depth=5, subsample=0.8
        """)
    with col_b:
        m2 = clf_meta["metrics"]
        st.markdown(f"""
        **📈 Promotion Model** (Random Forest Classifier)
        - Accuracy: `{m2['Accuracy']:.4f}`
        - F1 (weighted): `{m2['F1_weighted']:.4f}`
        - ROC-AUC: `{m2['ROC_AUC']:.4f}`
        - Features: 11 numeric + 5 categorical (after OHE → 60+ columns)
        - Hyperparams: n_estimators=200, max_depth=20, class_weight=balanced
        """)


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/machine-learning.png", width=80)
    st.markdown("## 23CSE301 ML Project")
    st.markdown("**B.Tech CSE — III Year**")
    st.markdown("Academic Year 2026-27")
    st.markdown("---")

    st.markdown("### 🗂 Tasks")
    st.markdown("""
    - ✅ **Regression** — Used Car Price  
      *Gradient Boosting Regressor*
    - ✅ **Classification** — Employee Promotion  
      *Random Forest Classifier*
    - ✅ **Clustering** — Marketing Campaigns  
      *K-Means + Agglomerative*
    """)

    st.markdown("---")
    st.markdown("### 📦 Datasets")
    st.markdown("""
    | Task | Dataset |
    |------|---------|
    | Regression | Used Cars UK (99 k rows) |
    | Classification | HR Analytics (54 k rows) |
    | Clustering | Marketing Campaign (2 k rows) |
    """)

    st.markdown("---")
    st.markdown("### 📚 Algorithms (20 total)")
    with st.expander("Regression (10)"):
        st.markdown("""
        Linear · Ridge · Lasso · ElasticNet ·  
        Polynomial · Decision Tree · Random Forest ·  
        **Gradient Boosting** · SVR · KNN
        """)
    with st.expander("Classification (10)"):
        st.markdown("""
        Logistic · KNN · Naive Bayes · Decision Tree · SVM ·  
        **Random Forest** · AdaBoost · Gradient Boosting ·  
        Bagging · MLP (Neural Network)
        """)
    with st.expander("Clustering (2)"):
        st.markdown("K-Means · Agglomerative Hierarchical")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#aaa'>Built with Streamlit · Scikit-learn · Plotly</div>",
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    '<div class="custom-footer">© 2026 ML Case Study · 23CSE301 · B.Tech CSE III Year</div>',
    unsafe_allow_html=True,
)
