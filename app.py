import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="HR Analytics – Reduced Feature Model",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------
# Load Artifacts (ONLY PKL FILES)
# ---------------------------------
@st.cache_resource
def load_models():
    features = joblib.load("features.pkl")          # Selected features only
    scaler = joblib.load("scaler.pkl")
    clf_model = joblib.load("best_classifier.pkl")
    reg_model = joblib.load("best_regressor.pkl")
    return features, scaler, clf_model, reg_model

features, scaler, clf_model, reg_model = load_models()

# ---------------------------------
# UI Header
# ---------------------------------
st.title("📊 HR Analytics Prediction System")
st.markdown("""
This system predicts:
- **Employee Attrition (Classification)**
- **Monthly Income (Regression)**

The models are built using a reduced set of features selected via feature selection techniques.
""")

# ---------------------------------
# Sidebar – Model Information
# ---------------------------------
st.sidebar.header("🧠 Model Summary")


st.sidebar.markdown("""
### Classification Models
- Logistic Regression 84.01%✅ (Selected)
- KNN
- Decision Tree
- Random Forest
- Naive Bayes

### Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor 91.85%✅ (Selected)

### Feature Selection
- SelectKBest (F-classification, F-regression)
- Correlation Analysis
- Minimal optimal features retained
""")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected Features:** {len(features)}")
with st.sidebar.expander("📋 View Features"):
    for i, feat in enumerate(features, 1):
        st.text(f"{i}. {feat}")

# ---------------------------------
# Input Section - YOUR 6 FEATURES
# ---------------------------------
st.subheader("📝 Enter Employee Details")

input_data = {}
cols = st.columns(3)

# Feature 1: Age
input_data['Age'] = cols[0].number_input(
    "Age",
    min_value=18,
    max_value=60,
    value=35,
    step=1,
    help="Employee's age in years"
)

# Feature 2: YearsWithCurrManager
input_data['YearsWithCurrManager'] = cols[1].number_input(
    "YearsWithCurrManager",
    min_value=0,
    max_value=17,
    value=3,
    step=1,
    help="Years working with current manager"
)

# Feature 3: YearsAtCompany
input_data['YearsAtCompany'] = cols[2].number_input(
    "YearsAtCompany",
    min_value=0,
    max_value=40,
    value=5,
    step=1,
    help="Total years at the company"
)

# Feature 4: TotalWorkingYears
input_data['TotalWorkingYears'] = cols[0].number_input(
    "TotalWorkingYears",
    min_value=0,
    max_value=40,
    value=10,
    step=1,
    help="Total years of work experience"
)

# Feature 5: JobLevel
input_data['JobLevel'] = cols[1].selectbox(
    "JobLevel",
    options=[1, 2, 3, 4, 5],
    index=1,
    format_func=lambda x: f"{x} - {['Entry', 'Junior', 'Mid', 'Senior', 'Executive'][x-1]}",
    help="Job level in organization hierarchy"
)

# Feature 6: YearsInCurrentRole
input_data['YearsInCurrentRole'] = cols[2].number_input(
    "YearsInCurrentRole",
    min_value=0,
    max_value=18,
    value=3,
    step=1,
    help="Years in current role/position"
)

input_df = pd.DataFrame([input_data])

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔍 Predict"):
    try:
        # Ensure correct feature order
        input_df = input_df[features]
        
        # Scale
        X_scaled = scaler.transform(input_df)
        
        # Predictions
        attr_pred = clf_model.predict(X_scaled)[0]
        attr_prob = clf_model.predict_proba(X_scaled)[0][1]
        salary_pred = reg_model.predict(X_scaled)[0]
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if attr_pred == 1:
                st.error("⚠️ Employee is Likely to Leave")
            else:
                st.success("✅ Employee is Likely to Stay")
            st.metric("Attrition Probability", f"{attr_prob * 100:.2f}%")
        
        with col2:
            st.metric("Predicted Monthly Income", f"₹ {salary_pred:,.2f}")
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown("""
🎓 **Project – Supervised Machine Learning**  
📌 Feature Reduction using SelectKBest / Correlation  
🚀 Deployed with Streamlit
""")