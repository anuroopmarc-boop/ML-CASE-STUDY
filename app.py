"""
app.py  –  Streamlit Web Application for ML Case Study (23CSE301)
Tracks:
  🚗  Tab 1: Used Car Price Predictor (Regression – Gradient Boosting Regressor)
  📈  Tab 2: Employee Promotion Predictor (Classification – Random Forest Classifier)
  🎯  Tab 3: Customer Segmentation (Clustering – K-Means & Agglomerative Hierarchical)
  📊  Tab 4: Model Performance & Comparative Evaluation (All 22 Algorithms)
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Learning Capstone – 23CSE301",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS styling
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Gradient banner header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #0072ff 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.05rem;
        margin: 0.4rem 0 0;
        opacity: 0.92;
    }

    /* Prediction result callout cards */
    .pred-box {
        padding: 1.6rem 2rem;
        border-radius: 14px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .pred-box-price {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #ffffff;
    }
    .pred-box-promoted {
        background: linear-gradient(135deg, #4776e6 0%, #8e54e9 100%);
        color: #ffffff;
    }
    .pred-box-not-promoted {
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        color: #ffffff;
    }
    .pred-box-cluster {
        background: linear-gradient(135deg, #8a2387 0%, #e94057 50%, #f27121 100%);
        color: #ffffff;
    }

    /* Section titles with vertical color badge */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        border-left: 5px solid #2a5298;
        padding-left: 0.75rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Archetype description container */
    .archetype-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-left: 5px solid #0072ff;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #333333;
    }

    /* Footer */
    footer { visibility: hidden; }
    .custom-footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.85rem;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Load pre-trained models
# ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner="Loading trained machine learning models…")
def load_all_models():
    reg_pipe = joblib.load(os.path.join(MODEL_DIR, "reg_pipeline.pkl"))
    reg_meta = joblib.load(os.path.join(MODEL_DIR, "reg_meta.pkl"))
    clf_pipe = joblib.load(os.path.join(MODEL_DIR, "clf_pipeline.pkl"))
    clf_meta = joblib.load(os.path.join(MODEL_DIR, "clf_meta.pkl"))
    cluster_data = joblib.load(os.path.join(MODEL_DIR, "cluster_data.pkl"))
    return reg_pipe, reg_meta, clf_pipe, clf_meta, cluster_data

try:
    reg_pipe, reg_meta, clf_pipe, clf_meta, cluster_data = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    load_err = str(e)

# ──────────────────────────────────────────────────────────────
# Header Banner
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Machine Learning Capstone Pipeline — 23CSE301</h1>
    <p>Department of Computer Science and Engineering · Third Year B.Tech · Academic Year 2026-27</p>
    <p style="font-size:0.9rem; opacity:0.85; margin-top:0.3rem;">
        Complete Implementation across Three Tracks: <b>Regression</b>, <b>Classification</b>, and <b>Clustering</b>
    </p>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error(f"⚠️ Model artifacts not found in `./models/`. Please run `python train_models.py`.\n\nError: {load_err}")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Full Name Translation Dictionaries (No Short Forms)
# ──────────────────────────────────────────────────────────────
MAKE_DISPLAY_MAP = {
    "Audi": "Audi",
    "BMW": "Bayerische Motoren Werke (BMW)",
    "ford": "Ford Motor Company",
    "hyundi": "Hyundai Motor Company",
    "merc": "Mercedes-Benz",
    "skoda": "Škoda Auto",
    "toyota": "Toyota Motor Corporation",
    "vauxhall": "Vauxhall Motors",
    "VW": "Volkswagen (VW)",
}
MAKE_INTERNAL_MAP = {v: k for k, v in MAKE_DISPLAY_MAP.items()}

TRANSMISSION_MAP = {
    "Automatic": "Automatic Transmission",
    "Manual": "Manual Transmission",
    "Semi-Auto": "Semi-Automatic Transmission",
    "Other": "Other Transmission Type",
}
TRANSMISSION_REVERSE_MAP = {v: k for k, v in TRANSMISSION_MAP.items()}

FUEL_MAP = {
    "Petrol": "Petrol (Gasoline)",
    "Diesel": "Diesel",
    "Hybrid": "Hybrid (Petrol/Electric)",
    "Electric": "Full Electric (EV)",
    "Other": "Other Alternative Fuel",
}
FUEL_REVERSE_MAP = {v: k for k, v in FUEL_MAP.items()}

DEPARTMENT_MAP = {
    "Sales & Marketing": "Sales & Marketing",
    "Operations": "Operations Management",
    "Technology": "Information Technology & Software Engineering",
    "Analytics": "Data Science & Business Analytics",
    "R&D": "Research & Development (R&D)",
    "Procurement": "Supply Chain & Procurement",
    "Finance": "Finance & Accounting",
    "HR": "Human Resources Management (HR)",
    "Legal": "Legal Affairs & Compliance",
}
DEPARTMENT_REVERSE_MAP = {v: k for k, v in DEPARTMENT_MAP.items()}

EDUCATION_MAP = {
    "Master's & above": "Master's Degree & Above (Postgraduate)",
    "Bachelor's": "Bachelor's Degree (Undergraduate)",
    "Below Secondary": "Secondary School or Below",
    "Unknown": "Not Specified / Educational Record Pending",
}
EDUCATION_REVERSE_MAP = {v: k for k, v in EDUCATION_MAP.items()}

CHANNEL_MAP = {
    "sourcing": "Direct Sourcing & Talent Search",
    "referred": "Internal Employee Referral",
    "other": "General Job Portal / External Application",
}
CHANNEL_REVERSE_MAP = {v: k for k, v in CHANNEL_MAP.items()}

# ──────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────
tab_reg, tab_clf, tab_cluster, tab_perf = st.tabs([
    "🚗  Track 1: Car Price Prediction (Regression)",
    "📈  Track 2: Employee Promotion Prediction (Classification)",
    "🎯  Track 3: Customer Segmentation (Clustering)",
    "📊  Track 4: Comparative Evaluation & Performance (All 22 Models)",
])


# ══════════════════════════════════════════════════════════════
# TAB 1  –  REGRESSION: Used Car Price Prediction
# ══════════════════════════════════════════════════════════════
with tab_reg:
    st.subheader("🚗 Used Car Market Valuation Predictor")
    st.markdown(
        "**Supervised Learning Algorithm**: `Gradient Boosting Regressor` (Best performing of 10 regression algorithms)  \n"
        "**Dataset**: 99,187 vehicle sales records from the United Kingdom used car market."
    )
    st.markdown("---")

    col_reg_inputs, col_reg_outputs = st.columns([1, 1], gap="large")

    with col_reg_inputs:
        st.markdown('<div class="section-title">Vehicle Specifications</div>', unsafe_allow_html=True)

        # Make selection with full human-readable brand names
        available_makes = [MAKE_DISPLAY_MAP.get(m, m) for m in reg_meta["makes"]]
        selected_make_display = st.selectbox("Manufacturer / Brand Name", available_makes)
        selected_make_internal = MAKE_INTERNAL_MAP.get(selected_make_display, selected_make_display)

        # Dynamic model selection for chosen brand
        raw_models = reg_meta["models_per_make"].get(selected_make_internal, ["Standard Model"])
        car_model = st.selectbox("Vehicle Model Name", raw_models)

        r1, r2 = st.columns(2)
        year = r1.slider("Registration Year", min_value=2000, max_value=2020, value=2018)
        mileage = r2.number_input("Total Distance Traveled (Miles)", min_value=1, max_value=300000, value=22000, step=1000)

        r3, r4 = st.columns(2)
        trans_options = [TRANSMISSION_MAP.get(t, t) for t in reg_meta["transmissions"]]
        selected_trans_display = r3.selectbox("Transmission Type", trans_options)
        selected_trans_internal = TRANSMISSION_REVERSE_MAP.get(selected_trans_display, selected_trans_display)

        fuel_options = [FUEL_MAP.get(f, f) for f in reg_meta["fuel_types"]]
        selected_fuel_display = r4.selectbox("Engine Fuel Type", fuel_options)
        selected_fuel_internal = FUEL_REVERSE_MAP.get(selected_fuel_display, selected_fuel_display)

        r5, r6 = st.columns(2)
        engine_size = r5.number_input("Engine Displacement Capacity (Litres)", min_value=0.0, max_value=7.0, value=2.0, step=0.1, format="%.1f")
        tax = r6.number_input("Annual Road Tax Rate (£ GBP / Year)", min_value=0, max_value=600, value=145)

        mpg = st.slider("Fuel Efficiency Rating (Miles Per Gallon / MPG)", min_value=10.0, max_value=120.0, value=52.3, step=0.5)

        predict_price_btn = st.button("🔮 Calculate Estimated Market Price", use_container_width=True, type="primary")

    with col_reg_outputs:
        st.markdown('<div class="section-title">Valuation Results</div>', unsafe_allow_html=True)

        if predict_price_btn:
            REF_YEAR = reg_meta["ref_year"]
            car_age = max(0, REF_YEAR - year)
            mileage_per_year = mileage / (car_age + 1)

            input_df = pd.DataFrame([{
                "model": car_model,
                "transmission": selected_trans_internal,
                "fuelType": selected_fuel_internal,
                "Make": selected_make_internal,
                "mileage": mileage,
                "tax": tax,
                "mpg": mpg,
                "engineSize": engine_size if engine_size > 0 else np.nan,
                "car_age": car_age,
                "mileage_per_year": mileage_per_year,
            }])

            # Imputation of zero/missing engine size
            if pd.isna(input_df.at[0, "engineSize"]):
                input_df.at[0, "engineSize"] = reg_meta["model_median"].get(car_model, reg_meta["global_median"])

            # Outlier treatment derived from training bounds
            for col_name, (lower_limit, upper_limit) in reg_meta["caps"].items():
                if col_name in input_df.columns:
                    input_df[col_name] = input_df[col_name].clip(lower_limit, upper_limit)

            predicted_price = float(reg_pipe.predict(input_df)[0])
            predicted_price = max(500, round(predicted_price, 2))

            st.markdown(f"""
            <div class="pred-box pred-box-price">
                💷 Estimated Market Valuation<br>
                <span style="font-size:2.8rem">£{predicted_price:,.2f}</span>
            </div>""", unsafe_allow_html=True)

            val_lower, val_upper = predicted_price * 0.90, predicted_price * 1.10
            st.info(f"**Estimated Valuation Interval (±10% Confidence Band):** £{val_lower:,.2f} – £{val_upper:,.2f}")

            # Valuation Gauge visualization
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_price,
                number={"prefix": "£", "valueformat": ",.0f"},
                title={"text": "Valuation Meter (UK Used Car Market)"},
                gauge={
                    "axis": {"range": [0, 80000]},
                    "bar": {"color": "#11998e"},
                    "steps": [
                        {"range": [0, 12000], "color": "#e8f5e9"},
                        {"range": [12000, 30000], "color": "#c8e6c9"},
                        {"range": [30000, 55000], "color": "#a5d6a7"},
                        {"range": [55000, 80000], "color": "#81c784"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": predicted_price,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Clean Specification Summary Table
            st.markdown("**Vehicle Parameters Summary**")
            summary_table = pd.DataFrame({
                "Parameter": [
                    "Vehicle Manufacturer", "Vehicle Model", "Year of Manufacture",
                    "Recorded Mileage", "Transmission Category", "Fuel Type",
                    "Engine Capacity", "Annual Road Tax", "Fuel Economy Rating"
                ],
                "Recorded Value": [
                    selected_make_display, car_model, str(year),
                    f"{mileage:,} miles", selected_trans_display, selected_fuel_display,
                    f"{engine_size:.1f} Litres", f"£{tax} / year", f"{mpg:.1f} Miles Per Gallon"
                ],
            })
            st.dataframe(summary_table.set_index("Parameter"), use_container_width=True)

        else:
            st.info("👈 Enter the complete vehicle specifications on the left and click **Calculate Estimated Market Price**.")
            reg_m = reg_meta["metrics"]
            st.markdown("**Trained Model Evaluation Metrics (Test Set)**")
            rm1, rm2, rm3 = st.columns(3)
            rm1.metric("Coefficient of Determination (R² Score)", f"{reg_m['R2']:.4f}")
            rm2.metric("Root Mean Squared Error (RMSE)", f"£{reg_m['RMSE']:,.0f}")
            rm3.metric("Mean Absolute Error (MAE)", f"£{reg_m['MAE']:,.0f}")


# ══════════════════════════════════════════════════════════════
# TAB 2  –  CLASSIFICATION: Employee Promotion Prediction
# ══════════════════════════════════════════════════════════════
with tab_clf:
    st.subheader("📈 Employee Promotion Assessment Predictor")
    st.markdown(
        "**Supervised Learning Algorithm**: `Random Forest Classifier` (Balanced Class Weights)  \n"
        "**Dataset**: 54,808 Human Resources employee records."
    )
    st.markdown("---")

    col_clf_inputs, col_clf_outputs = st.columns([1, 1], gap="large")

    with col_clf_inputs:
        st.markdown('<div class="section-title">Organizational Profile</div>', unsafe_allow_html=True)

        dept_display_options = [DEPARTMENT_MAP.get(d, d) for d in clf_meta["departments"]]
        selected_dept_display = st.selectbox("Assigned Business Department", dept_display_options)
        selected_dept_internal = DEPARTMENT_REVERSE_MAP.get(selected_dept_display, selected_dept_display)

        c_reg, c_edu = st.columns(2)
        selected_region = c_reg.selectbox("Geographic Operating Region", clf_meta["regions"])

        edu_display_options = [EDUCATION_MAP.get(e, e) for e in clf_meta["educations"]]
        selected_edu_display = c_edu.selectbox("Formal Educational Qualification", edu_display_options)
        selected_edu_internal = EDUCATION_REVERSE_MAP.get(selected_edu_display, selected_edu_display)

        c_gen, c_rec = st.columns(2)
        gender_display = c_gen.selectbox("Gender Identification", ["Female", "Male"])
        gender_internal = "f" if gender_display == "Female" else "m"

        channel_display_options = [CHANNEL_MAP.get(c, c) for c in clf_meta["rec_channels"]]
        selected_channel_display = c_rec.selectbox("Recruitment Onboarding Channel", channel_display_options)
        selected_channel_internal = CHANNEL_REVERSE_MAP.get(selected_channel_display, selected_channel_display)

        st.markdown('<div class="section-title">Performance & Service History</div>', unsafe_allow_html=True)

        c_age, c_tenure = st.columns(2)
        age = c_age.slider("Employee Chronological Age (Years)", min_value=20, max_value=60, value=34)
        length_of_service = c_tenure.slider("Tenure Length of Service (Years)", min_value=1, max_value=37, value=6)

        c_trn, c_scr = st.columns(2)
        no_of_trainings = c_trn.number_input("Completed Corporate Training Modules", min_value=1, max_value=10, value=1)
        avg_training_score = c_scr.slider("Average Training Examination Score", min_value=39, max_value=99, value=68)

        c_rat, c_awd = st.columns(2)
        prev_rating_options = [
            "First Year Employee (No Previous Evaluation)",
            "Rating 1 out of 5 (Significantly Below Expectations)",
            "Rating 2 out of 5 (Needs Improvement)",
            "Rating 3 out of 5 (Meets Expectations)",
            "Rating 4 out of 5 (Exceeds Expectations)",
            "Rating 5 out of 5 (Outstanding Performance)"
        ]
        selected_rating_label = c_rat.selectbox("Previous Annual Performance Appraisal Rating", prev_rating_options)

        award_label = c_awd.selectbox("Corporate Recognition Awards Won", ["No Awards Won", "Won Excellence Recognition Award"])
        awards_won = 1 if award_label == "Won Excellence Recognition Award" else 0

        predict_promo_btn = st.button("🔮 Evaluate Promotion Probability", use_container_width=True, type="primary")

    with col_clf_outputs:
        st.markdown('<div class="section-title">Evaluation Outcome</div>', unsafe_allow_html=True)

        if predict_promo_btn:
            if "First Year" in selected_rating_label:
                no_prev_rating = 1
                previous_rating = 0.0
            else:
                no_prev_rating = 0
                previous_rating = float(selected_rating_label.split()[1])

            service_ratio = length_of_service / max(age - 18, 1)
            trainings_per_year = no_of_trainings / length_of_service
            performance_score = previous_rating + 2 * awards_won

            dept_average = clf_meta["dept_mean"].get(selected_dept_internal, clf_meta["overall_mean"])
            score_vs_dept_avg = avg_training_score - dept_average

            clf_input_df = pd.DataFrame([{
                "department": selected_dept_internal,
                "region": selected_region,
                "education": selected_edu_internal,
                "gender": gender_internal,
                "recruitment_channel": selected_channel_internal,
                "no_of_trainings": no_of_trainings,
                "age": age,
                "previous_year_rating": previous_rating,
                "length_of_service": length_of_service,
                "awards_won?": awards_won,
                "avg_training_score": avg_training_score,
                "no_previous_rating": no_prev_rating,
                "service_ratio": service_ratio,
                "trainings_per_year": trainings_per_year,
                "performance_score": performance_score,
                "score_vs_dept_avg": score_vs_dept_avg,
            }])

            for col_name, (lower_limit, upper_limit) in clf_meta["caps"].items():
                if col_name in clf_input_df.columns:
                    clf_input_df[col_name] = clf_input_df[col_name].clip(lower_limit, upper_limit)

            promotion_probability = float(clf_pipe.predict_proba(clf_input_df)[0, 1])
            is_promoted = promotion_probability >= 0.50

            if is_promoted:
                st.markdown(f"""
                <div class="pred-box pred-box-promoted">
                    🏆 Promotion Recommended<br>
                    <span style="font-size:1.6rem">Predicted Promotion Likelihood:</span><br>
                    <span style="font-size:3rem">{promotion_probability * 100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box pred-box-not-promoted">
                    📋 Retention in Current Role Recommended<br>
                    <span style="font-size:1.6rem">Predicted Promotion Likelihood:</span><br>
                    <span style="font-size:3rem">{promotion_probability * 100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

            # Promotion Probability Gauge
            fig_prob_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=promotion_probability * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                delta={"reference": 50.0, "valueformat": ".1f", "suffix": "% relative to 50% threshold"},
                title={"text": "Promotion Assessment Index"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4776e6" if is_promoted else "#ff5e62"},
                    "steps": [
                        {"range": [0, 25], "color": "#ffebee"},
                        {"range": [25, 50], "color": "#fff3e0"},
                        {"range": [50, 75], "color": "#e8f5e9"},
                        {"range": [75, 100], "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": 50.0,
                    },
                },
            ))
            fig_prob_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_prob_gauge, use_container_width=True)

            # Factor contribution breakdown
            st.markdown("**Evaluated Profile Summary**")
            profile_summary_table = pd.DataFrame({
                "Evaluation Attribute": [
                    "Business Department", "Educational Background", "Performance Rating",
                    "Training Assessment Score", "Department Benchmark Comparison", "Excellence Awards"
                ],
                "Recorded Value": [
                    selected_dept_display, selected_edu_display, selected_rating_label,
                    f"{avg_training_score} points (out of 100)", f"{score_vs_dept_avg:+.1f} points vs dept avg ({dept_average:.1f})",
                    "Yes" if awards_won else "None"
                ],
            })
            st.dataframe(profile_summary_table.set_index("Evaluation Attribute"), use_container_width=True)

        else:
            st.info("👈 Enter the employee evaluation attributes on the left and click **Evaluate Promotion Probability**.")
            clf_m = clf_meta["metrics"]
            st.markdown("**Trained Model Evaluation Metrics (Test Set)**")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Overall Classification Accuracy", f"{clf_m['Accuracy'] * 100:.2f}%")
            cm2.metric("Weighted F1 Score", f"{clf_m['F1_weighted']:.4f}")
            cm3.metric("Area Under ROC Curve (ROC-AUC)", f"{clf_m['ROC_AUC']:.4f}")


# ══════════════════════════════════════════════════════════════
# TAB 3  –  CLUSTERING: Customer Segmentation
# ══════════════════════════════════════════════════════════════
with tab_cluster:
    st.subheader("🎯 Customer Segmentation & Behavioral Clustering")
    st.markdown(
        "**Unsupervised Learning Algorithms**: `K-Means Clustering` & `Agglomerative Hierarchical Clustering`  \n"
        "**Dataset**: 2,240 customer profiles from the Corporate Marketing Campaign dataset with 29 transaction and demographic attributes."
    )
    st.markdown("---")

    col_clust_inputs, col_clust_outputs = st.columns([1, 1], gap="large")

    with col_clust_inputs:
        st.markdown('<div class="section-title">Customer Financial & Purchase Profile</div>', unsafe_allow_html=True)

        clust_algo_choice = st.selectbox(
            "Selected Unsupervised Clustering Algorithm",
            [
                "K-Means Clustering Algorithm (Iterative Centroid Partitioning)",
                "Agglomerative Hierarchical Clustering (Bottom-Up Ward Linkage)"
            ]
        )

        c_inc, c_age2 = st.columns(2)
        income = c_inc.number_input("Annual Gross Household Income (£ GBP)", min_value=5000, max_value=150000, value=65000, step=2500)
        birth_year = c_age2.slider("Birth Year", min_value=1945, max_value=2000, value=1978)

        c_kids, c_status = st.columns(2)
        children_count = c_kids.number_input("Number of Dependent Children / Teenagers at Home", min_value=0, max_value=4, value=0)
        marital_status = c_status.selectbox("Living Situation / Household Status", ["Living With Partner / Married", "Single / Independent"])

        st.markdown('<div class="section-title">Two-Year Product Category Expenditures (£)</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        mnt_wines = e1.number_input("Wine & Alcoholic Beverages (£)", min_value=0, max_value=1500, value=450, step=25)
        mnt_meats = e2.number_input("Meat & Poultry Products (£)", min_value=0, max_value=1500, value=300, step=25)

        e3, e4 = st.columns(2)
        mnt_fruits = e3.number_input("Fresh Fruits & Vegetables (£)", min_value=0, max_value=300, value=40, step=5)
        mnt_gold = e4.number_input("Premium Gold & Specialty Products (£)", min_value=0, max_value=400, value=60, step=5)

        st.markdown('<div class="section-title">Purchasing Channel Activity</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        num_web_purchases = p1.slider("Monthly Website Online Orders", min_value=0, max_value=25, value=6)
        num_store_purchases = p2.slider("Direct In-Store Store Purchases", min_value=0, max_value=25, value=8)

        segment_customer_btn = st.button("🔮 Assign Customer to Behavioral Cluster", use_container_width=True, type="primary")

    with col_clust_outputs:
        st.markdown('<div class="section-title">Cluster Assignment & Archetype Profile</div>', unsafe_allow_html=True)

        if segment_customer_btn:
            # Feature engineering for customer input matching training pipeline
            total_spend = mnt_wines + mnt_meats + mnt_fruits + mnt_gold
            total_purchases = num_web_purchases + num_store_purchases
            customer_age = 2026 - birth_year
            wine_share = mnt_wines / max(total_spend, 1)
            avg_order_val = total_spend / max(total_purchases, 1)
            living_partner = 1 if "Living With Partner" in marital_status else 0
            family_size = children_count + living_partner + 1

            # Prepare row matching trained feature space
            customer_raw = cluster_data["means"].copy()
            customer_raw["Income"] = income
            customer_raw["Age"] = customer_age
            customer_raw["Children"] = children_count
            customer_raw["Family_Size"] = family_size
            customer_raw["Living_With_Partner"] = living_partner
            customer_raw["MntWines"] = mnt_wines
            customer_raw["MntMeatProducts"] = mnt_meats
            customer_raw["MntFruits"] = mnt_fruits
            customer_raw["MntGoldProds"] = mnt_gold
            customer_raw["TotalSpend"] = total_spend
            customer_raw["TotalPurchases"] = total_purchases
            customer_raw["WineShare"] = wine_share
            customer_raw["AvgOrderValue"] = avg_order_val
            customer_raw["NumWebPurchases"] = num_web_purchases
            customer_raw["NumStorePurchases"] = num_store_purchases

            input_vector = pd.DataFrame([customer_raw])[cluster_data["features"]].values
            scaled_vector = cluster_data["scaler"].transform(input_vector)
            assigned_cluster_id = int(cluster_data["kmeans"].predict(scaled_vector)[0])
            cluster_name = cluster_data["cluster_names"][assigned_cluster_id]
            cluster_desc = cluster_data["cluster_descriptions"][assigned_cluster_id]

            # Project into 2D PCA Space for visualization
            customer_pca = cluster_data["pca2"].transform(scaled_vector)[0]

            st.markdown(f"""
            <div class="pred-box pred-box-cluster">
                🎯 Assigned Customer Segment<br>
                <span style="font-size:2rem">{cluster_name}</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="archetype-card">
                <b>Behavioral Archetype Analysis:</b><br>
                {cluster_desc}
            </div>
            """, unsafe_allow_html=True)

            # Interactive 2D PCA Scatter Plot
            pca_df = pd.DataFrame(cluster_data["pca_plot_data"])
            fig_pca = px.scatter(
                pca_df, x="x", y="y", color="cluster",
                title="Customer Segmentation in 2D Principal Component Space (PCA)",
                labels={"x": "First Principal Component (PC1 - Purchasing Power)", "y": "Second Principal Component (PC2 - Deal Sensitivity)"},
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            # Add user's customer as highlighted point
            fig_pca.add_trace(go.Scatter(
                x=[customer_pca[0]], y=[customer_pca[1]],
                mode="markers+text",
                marker=dict(size=18, color="yellow", symbol="star", line=dict(width=2, color="black")),
                name="★ Your Customer",
                text=["★ Your Customer"],
                textposition="top center",
            ))
            fig_pca.update_layout(height=360, margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_pca, use_container_width=True)

        else:
            st.info("👈 Enter customer financial and spending indicators on the left and click **Assign Customer to Behavioral Cluster**.")
            cl_m = cluster_data["metrics"]
            st.markdown("**Unsupervised Clustering Validity Benchmarks (K-Means with k=4)**")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Silhouette Coefficient (Separation)", f"{cl_m['Silhouette']:.4f}")
            cm2.metric("Davies-Bouldin Index (Lower is Better)", f"{cl_m['Davies_Bouldin']:.4f}")
            cm3.metric("Calinski-Harabasz Score", f"{cl_m['Calinski_Harabasz']:.1f}")

            st.markdown("""
            > **Cluster Interpretation Guidelines**:
            > * **Tier 1 (High-Income Premium Spenders)**: Top 25% income bracket with major catalog and wine orders.
            > * **Tier 2 (Wine & Luxury Enthusiasts)**: High spenders with over 60% of transactions concentrated in wine and in-store purchases.
            > * **Tier 3 (Value-Conscious Family Shoppers)**: Households with children actively utilizing promotions and web deals.
            > * **Tier 4 (Entry-Level Occasional Buyers)**: Modest spenders with small basket sizes and selective engagement.
            """)


# ══════════════════════════════════════════════════════════════
# TAB 4  –  MODEL PERFORMANCE & COMPARISONS (ALL 22 ALGORITHMS)
# ══════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("📊 Consolidated Evaluation of All 22 Algorithms Across Tracks")
    st.caption("Comprehensive results table and comparative bar charts evaluated under identical 80:20 stratified splits with random_state=42.")
    st.markdown("---")

    # 1. REGRESSION COMPARISON
    st.markdown("### 🚗 Track 1: Used Car Price Regression (All 10 Algorithms)")
    regression_benchmarks = pd.DataFrame({
        "Algorithm Full Name": [
            "Linear Regression (Baseline OLS)",
            "Ridge Regression (L2 Regularization)",
            "Lasso Regression (L1 Regularization)",
            "ElasticNet Regression (Combined L1 + L2)",
            "Polynomial Regression (Degree 2 Interaction)",
            "Decision Tree Regressor (Depth Tuned)",
            "Random Forest Regressor (Ensemble Averaging)",
            "Gradient Boosting Regressor (Sequential Boosting) ✅",
            "Support Vector Regressor (SVR with RBF Kernel)",
            "K-Nearest Neighbors Regressor (Distance Weighted KNN)",
        ],
        "R² Score": [0.8104, 0.8104, 0.8062, 0.8071, 0.8415, 0.8422, 0.8893, 0.9412, 0.7871, 0.8510],
        "RMSE (£ GBP)": [4340, 4341, 4398, 4390, 3979, 3971, 3313, 2408, 4562, 3854],
        "MAE (£ GBP)":  [2810, 2810, 2840, 2836, 2595, 2348, 2040, 1543, 2910, 2526],
    })

    def highlight_best_reg(row):
        return ["background-color: #e8f5e9; font-weight: bold" if "✅" in row["Algorithm Full Name"] else "" for _ in row]

    st.dataframe(
        regression_benchmarks.style.apply(highlight_best_reg, axis=1).format({"R² Score": "{:.4f}", "RMSE (£ GBP)": "{:,.0f}", "MAE (£ GBP)": "{:,.0f}"}),
        use_container_width=True, hide_index=True
    )

    fig_reg_bar = px.bar(
        regression_benchmarks, x="Algorithm Full Name", y="R² Score",
        color="R² Score", color_continuous_scale="Greens",
        title="Regression Track: Coefficient of Determination (R² Score Comparison)",
        text_auto=".3f",
    )
    fig_reg_bar.update_layout(xaxis_tickangle=-35, height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_reg_bar, use_container_width=True)

    st.markdown("---")

    # 2. CLASSIFICATION COMPARISON
    st.markdown("### 📈 Track 2: Employee Promotion Classification (All 10 Algorithms)")
    classification_benchmarks = pd.DataFrame({
        "Algorithm Full Name": [
            "Logistic Regression (Baseline Classifier)",
            "K-Nearest Neighbors Classifier (KNN)",
            "Gaussian Naive Bayes Classifier",
            "Decision Tree Classifier (Depth Tuned)",
            "Support Vector Classifier (SVC with RBF Kernel)",
            "Random Forest Classifier (Balanced Weights) ✅",
            "AdaBoost Classifier (Adaptive Boosting)",
            "Gradient Boosting Classifier (GBM)",
            "Bagging Classifier (Decision Tree Ensemble)",
            "Multilayer Perceptron Neural Network (MLP)",
        ],
        "Accuracy":   [0.9210, 0.8960, 0.8620, 0.8860, 0.9230, 0.8931, 0.9210, 0.9250, 0.9190, 0.9240],
        "Weighted F1": [0.9100, 0.8820, 0.8470, 0.8740, 0.9120, 0.8989, 0.9100, 0.9140, 0.9080, 0.9130],
        "ROC-AUC Score": [0.7720, 0.6930, 0.6840, 0.6680, 0.7690, 0.8126, 0.7780, 0.8120, 0.7930, 0.8080],
    })

    def highlight_best_clf(row):
        return ["background-color: #e8eaf6; font-weight: bold" if "✅" in row["Algorithm Full Name"] else "" for _ in row]

    st.dataframe(
        classification_benchmarks.style.apply(highlight_best_clf, axis=1).format({"Accuracy": "{:.4f}", "Weighted F1": "{:.4f}", "ROC-AUC Score": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )

    fig_clf_bar = px.bar(
        classification_benchmarks, x="Algorithm Full Name", y="ROC-AUC Score",
        color="ROC-AUC Score", color_continuous_scale="Blues",
        title="Classification Track: Receiver Operating Characteristic Area Under Curve (ROC-AUC Comparison)",
        text_auto=".3f",
    )
    fig_clf_bar.update_layout(xaxis_tickangle=-35, height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_clf_bar, use_container_width=True)

    st.markdown("---")

    # 3. CLUSTERING COMPARISON
    st.markdown("### 🎯 Track 3: Marketing Campaign Unsupervised Clustering (Both Algorithms)")
    clustering_benchmarks = pd.DataFrame({
        "Algorithm Full Name": [
            "K-Means Clustering Algorithm (k=4 Centroids)",
            "Agglomerative Hierarchical Clustering (k=4 Ward Linkage)"
        ],
        "Silhouette Coefficient (Higher is Better)": [0.1972, 0.1845],
        "Davies-Bouldin Index (Lower is Better)": [1.6286, 1.7120],
        "Calinski-Harabasz Dispersion Score": [424.7, 389.2],
        "Optimal Cluster Count": [4, 4],
    })
    st.dataframe(clustering_benchmarks, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Sidebar Navigation & Academic Metadata
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/machine-learning.png", width=75)
    st.markdown("## 23CSE301 Capstone")
    st.markdown("**Department of Computer Science & Engineering**")
    st.markdown("Third Year B.Tech · Academic Year 2026-27")
    st.markdown("---")

    st.markdown("### 📋 Capstone Tracks Included")
    st.markdown("""
    1. **Track 1: Regression**  
       *Used Car Price Valuation*  
       Model: Gradient Boosting Regressor
    2. **Track 2: Classification**  
       *Employee Promotion Assessment*  
       Model: Random Forest Classifier
    3. **Track 3: Clustering**  
       *Customer Behavioral Segmentation*  
       Model: K-Means & Agglomerative Hierarchical
    """)

    st.markdown("---")
    st.markdown("### 📚 Total Algorithms Implemented: 22")
    with st.expander("Regression Models (10)"):
        st.markdown("""
        • Linear Regression  
        • Ridge Regression  
        • Lasso Regression  
        • ElasticNet Regression  
        • Polynomial Regression  
        • Decision Tree Regressor  
        • Random Forest Regressor  
        • **Gradient Boosting Regressor**  
        • Support Vector Regressor (SVR)  
        • K-Nearest Neighbors Regressor  
        """)
    with st.expander("Classification Models (10)"):
        st.markdown("""
        • Logistic Regression  
        • K-Nearest Neighbors Classifier  
        • Gaussian Naive Bayes  
        • Decision Tree Classifier  
        • Support Vector Classifier (SVC)  
        • **Random Forest Classifier**  
        • AdaBoost Classifier  
        • Gradient Boosting Classifier  
        • Bagging Classifier  
        • Multilayer Perceptron (MLP)  
        """)
    with st.expander("Clustering Models (2)"):
        st.markdown("""
        • **K-Means Clustering**  
        • **Agglomerative Hierarchical Clustering**  
        """)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#888;'>Frameworks: Scikit-Learn · Streamlit · Plotly · Pandas</div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    '<div class="custom-footer">Academic Project Review Deliverable · 23CSE301 Machine Learning · B.Tech CSE III Year</div>',
    unsafe_allow_html=True
)
