from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Land Acquisition Analytics",
    page_icon="🏗️",
    layout="wide"
)

REQUIRED_COLUMNS = [
    "Project_ID",
    "Year",
    "District",
    "Documents_Complete",
    "Compensation_Paid",
    "Legal_Dispute",
    "Approval_Pending",
    "Delay_Days",
    "Risk",
]
BOOL_COLUMNS = [
    "Documents_Complete",
    "Compensation_Paid",
    "Legal_Dispute",
    "Approval_Pending",
]
RISK_LEVELS = ["Low", "Medium", "High"]


# Demo data only — replace with verified historical records later.
@st.cache_data
def create_demo_data():
    rng = np.random.default_rng(42)
    n = 250

    df = pd.DataFrame({
        "Project_ID": [f"LA-{i:03d}" for i in range(1, n + 1)],
        "Year": rng.integers(2022, 2027, n),
        "District": rng.choice(
            ["Chennai", "Salem", "Madurai", "Coimbatore", "Trichy"], n
        ),
        "Documents_Complete": rng.choice([True, False], n, p=[0.7, 0.3]),
        "Compensation_Paid": rng.choice([True, False], n, p=[0.6, 0.4]),
        "Legal_Dispute": rng.choice([True, False], n, p=[0.2, 0.8]),
        "Approval_Pending": rng.choice([True, False], n, p=[0.3, 0.7]),
    })

    # Synthetic outcome for demonstrating charts, not real-world evidence.
    df["Delay_Days"] = (
        (~df["Documents_Complete"]) * rng.integers(15, 70, n)
        + (~df["Compensation_Paid"]) * rng.integers(20, 90, n)
        + df["Legal_Dispute"] * rng.integers(30, 120, n)
        + df["Approval_Pending"] * rng.integers(10, 60, n)
        + rng.integers(0, 16, n)
    )

    df["Risk"] = pd.cut(
        df["Delay_Days"],
        bins=[-1, 40, 100, float("inf")],
        labels=RISK_LEVELS
    ).astype(str)

    return df


def to_bool(series):
    """Convert True/False, Yes/No, 1/0 text to booleans. Unknown -> NaN."""
    mapping = {
        "true": True, "false": False,
        "yes": True, "no": False,
        "1": True, "0": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)



BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "land_delay_model.joblib"


@st.cache_resource
def load_model():
    """Load the model saved by train_model.py (None if missing/unreadable)."""
    if not MODEL_FILE.exists():
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return None


model = load_model()

# ---------------------------------------------------------------
# Load data: demo by default, uploaded CSV if valid.
# This MUST run before filters, metrics and charts are built.
# ---------------------------------------------------------------
df = create_demo_data()
using_demo = True

st.sidebar.header("Project Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload project records (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        missing_columns = [
            c for c in REQUIRED_COLUMNS if c not in uploaded_df.columns
        ]

        if missing_columns:
            st.sidebar.error(
                "Missing columns: " + ", ".join(missing_columns)
                + ". Showing demo data instead."
            )
        elif uploaded_df.empty:
            st.sidebar.error(
                "The uploaded CSV contains no records. "
                "Showing demo data instead."
            )
        else:
            clean = uploaded_df[REQUIRED_COLUMNS].copy()
            problems = []

            # Yes/No style columns
            for col in BOOL_COLUMNS:
                converted = to_bool(clean[col])
                if converted.isna().any():
                    problems.append(f"{col} (use True/False)")
                else:
                    clean[col] = converted.astype(bool)

            # Numeric columns
            for col in ["Year", "Delay_Days"]:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")
                if clean[col].isna().any():
                    problems.append(f"{col} (must be numbers)")

            # Risk labels
            clean["Risk"] = clean["Risk"].astype(str).str.strip().str.title()
            if not clean["Risk"].isin(RISK_LEVELS).all():
                problems.append("Risk (use Low, Medium or High)")

            if problems:
                st.sidebar.error(
                    "Invalid values in: " + "; ".join(problems)
                    + ". Showing demo data instead."
                )
            else:
                clean["Year"] = clean["Year"].astype(int)
                clean["District"] = clean["District"].astype(str).str.strip()
                df = clean
                using_demo = False
                st.sidebar.success(f"Loaded {len(df)} project records.")

    except Exception as error:
        st.sidebar.error(
            f"Could not read CSV: {error}. Showing demo data instead."
        )

# ---------------------------------------------------------------
# Page header
# ---------------------------------------------------------------
st.title("🏗️ Land Acquisition Delay Prediction System")
st.caption("Project monitoring • Risk analysis • Historical trends")

if using_demo:
    st.warning(
        "DEMO MODE: All displayed project records are synthetic. "
        "Risk categories are rule-based examples, not validated AI predictions. "
        "Do not use them for actual land acquisition decisions."
    )
else:
    st.info(
        "Showing your uploaded project records. Risk values come from your "
        "file; this app has not validated them."
    )

# Sidebar filters (built from whichever dataset is active)
st.sidebar.header("Dashboard Filters")
years = sorted(df["Year"].unique())
selected_years = st.sidebar.multiselect(
    "Select years", years, default=years
)
districts = sorted(df["District"].unique())
selected_districts = st.sidebar.multiselect(
    "Select districts", districts, default=districts
)

filtered = df[
    df["Year"].isin(selected_years)
    & df["District"].isin(selected_districts)
]

if filtered.empty:
    st.info("No records match your filters. Select another year or district.")
else:
    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", len(filtered))
    c2.metric("High-risk projects", int((filtered["Risk"] == "High").sum()))
    c3.metric("Average delay", f"{filtered['Delay_Days'].mean():.1f} days")
    c4.metric(
        "Pending compensation",
        int((~filtered["Compensation_Paid"]).sum())
    )

    st.divider()
    st.subheader("📊 Project Analytics")

    left, right = st.columns(2)

    with left:
        risk_counts = (
            filtered["Risk"].value_counts()
            .reindex(RISK_LEVELS, fill_value=0)
            .rename_axis("Risk level")
            .reset_index(name="Projects")
        )
        fig = px.pie(
            risk_counts,
            names="Risk level",
            values="Projects",
            title="Risk Distribution",
            color="Risk level",
            color_discrete_map={
                "Low": "#16a34a",
                "Medium": "#eab308",
                "High": "#dc2626"
            }
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        yearly = (
            filtered.groupby("Year", as_index=False, observed=True)
            .agg(Average_Delay=("Delay_Days", "mean"))
        )
        fig = px.line(
            yearly,
            x="Year",
            y="Average_Delay",
            markers=True,
            title="Average Delay by Year"
        )
        fig.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)

    with left:
        reasons = pd.DataFrame({
            "Issue": [
                "Documents incomplete",
                "Compensation pending",
                "Legal dispute",
                "Approval pending"
            ],
            "Projects": [
                int((~filtered["Documents_Complete"]).sum()),
                int((~filtered["Compensation_Paid"]).sum()),
                int(filtered["Legal_Dispute"].sum()),
                int(filtered["Approval_Pending"].sum())
            ]
        })
        fig = px.bar(
            reasons,
            x="Issue",
            y="Projects",
            title="Recorded Issues by Category"
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        district_summary = (
            filtered.groupby("District", as_index=False)
            .agg(Average_Delay=("Delay_Days", "mean"))
        )
        fig = px.bar(
            district_summary,
            x="District",
            y="Average_Delay",
            title="Average Delay by District"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("📋 Project Records")
    st.dataframe(filtered, width="stretch", hide_index=True)

    st.download_button(
        "Download filtered records (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="land_acquisition_records.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("🔎 Test a Project's Risk Factors")

if model is None:
    st.info(
        "No trained model found. Run `python train_model.py` to create "
        "`land_delay_model.joblib`. Using the hand-built example score instead."
    )
else:
    st.write(
        "Enter project conditions to get a risk category from the trained model."
    )

col1, col2 = st.columns(2)
with col1:
    year_options = sorted(set(df["Year"].tolist()))
    project_year = st.selectbox(
        "Project year", year_options, index=len(year_options) - 1
    )
    documents_complete = st.selectbox(
        "Documents complete?", ["Yes", "No"]
    )
    compensation_paid = st.selectbox(
        "Compensation paid?", ["Yes", "No"]
    )
with col2:
    project_district = st.selectbox(
        "District", sorted(df["District"].unique())
    )
    legal_dispute = st.selectbox(
        "Legal dispute reported?", ["No", "Yes"]
    )
    approval_pending = st.selectbox(
        "Approval pending?", ["No", "Yes"]
    )

if st.button("Assess Risk", type="primary"):
    actions = {
        "High": "Prioritize human review of disputes, compensation, and approvals.",
        "Medium": "Review outstanding documents and payment status.",
        "Low": "Continue routine monitoring and verify records.",
    }

    if model is not None:
        row = pd.DataFrame([{
            "Year": int(project_year),
            "District": project_district,
            "Documents_Complete": int(documents_complete == "Yes"),
            "Compensation_Paid": int(compensation_paid == "Yes"),
            "Legal_Dispute": int(legal_dispute == "Yes"),
            "Approval_Pending": int(approval_pending == "Yes"),
        }])
        try:
            risk = str(model.predict(row)[0])
            probs = model.predict_proba(row)[0]
            classes = [str(c) for c in model.classes_]

            st.subheader(f"Predicted risk category: {risk}")
            prob_cols = st.columns(len(classes))
            for pc, name, p in zip(prob_cols, classes, probs):
                pc.metric(f"{name} probability", f"{p:.0%}")
            st.write("Suggested review action:", actions.get(risk, "Review manually."))
            st.caption(
                "Model output from land_delay_model.joblib. If it was trained "
                "on synthetic demo data, this only demonstrates the workflow. "
                "It is not a validated prediction or a legal determination."
            )
        except Exception as error:
            st.error(f"The model could not make a prediction: {error}")
    else:
        score = (
            (documents_complete == "No") * 20
            + (compensation_paid == "No") * 25
            + (legal_dispute == "Yes") * 35
            + (approval_pending == "Yes") * 20
        )
        risk = "High" if score >= 50 else "Medium" if score >= 25 else "Low"
        st.metric("Illustrative risk score", f"{score}/100")
        st.subheader(f"Risk category: {risk}")
        st.write("Suggested review action:", actions[risk])
        st.caption(
            "This is a hand-built scoring example, not a trained model, "
            "a probability of delay, or a legal determination."
        )

st.caption("Land Acquisition Analytics • Beginner portfolio prototype")