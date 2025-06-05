import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from io import BytesIO

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib

st.set_page_config(page_title="Machine Learning Modeling", layout="wide")
st.title("Machine Learning Modeling App")

with st.sidebar:
    st.header("⚙️ Settings")
    use_sample = st.checkbox("Use sample data (Titanic)")

    if use_sample:
        df = sns.load_dataset("titanic").dropna(subset=["survived"])
    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

if 'df' in locals():
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    tab1, tab2, tab3 = st.tabs(["📋 Configuration", "🔍 Training", "📊 Model Evaluation"])

    with tab1:
        target_column = st.selectbox("🎯 Select target column:", df.columns)

        model_type = st.selectbox("🤖 Model:", ["Classification", "Regression"])

        # Deteksi kolom numerik & kategorikal awal
        default_num = df.select_dtypes(include=['int64', 'float64']).columns.difference([target_column])
        default_cat = df.select_dtypes(include=['object', 'category']).columns.difference([target_column])

        # Deteksi kolom biner (0 dan 1)
        binary_cols = [col for col in df.columns if df[col].dropna().nunique() == 2 and df[col].dropna().isin([0, 1]).all()]
        binary_cols = list(set(binary_cols) - {target_column})

        # Pisahkan binary features jika ingin treat sebagai numerik atau kategorikal
        numerical_binaries = binary_cols  # binary sebagai numerik
        categorical_binaries = []         # kosong jika tidak ingin ada binary treat sebagai kategori

        # Pilih fitur numerik dan kategorikal secara eksplisit
        num_features = st.multiselect(
            "🔢 Numerical features:", 
            df.columns, 
            default=list(default_num.union(numerical_binaries))
        )
        cat_features = st.multiselect(
            "🔤 Categorical features:", 
            df.columns.difference([target_column]), 
            default=list(default_cat.union(categorical_binaries))
        )

        if model_type == "Classification":
            model_name = st.selectbox("📚 Classification model", ["Logistic Regression", "Decision Trees", "Random Forest", "SVM", "XGBoost", "CatBoost"])
        else:
            model_name = st.selectbox("📚 Regression model", ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "XGBoost", "CatBoost"])

    with tab2:
        if st.button("🚀 Train Model"):
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            num_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            cat_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("numerical", num_transformer, num_features),
                ("categorical", cat_transformer, cat_features)
            ])

            if model_type == "Classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Trees": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC(),
                    "XGBoost": XGBClassifier(),
                    "CatBoost": CatBoostClassifier(verbose=0)
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "SVR": SVR(),
                    "XGBoost": XGBRegressor(),
                    "CatBoost": CatBoostRegressor(verbose=0)
                }

            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", models[model_name])
            ])

            if not num_features and not cat_features:
                st.error("⚠️ Please select at least 1 numerical or categorical feature.")
            else:
                model.fit(X_train, y_train)
                st.success("✅ Training model succesful")

                st.session_state["trained_model"] = model
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.session_state["model_name"] = model_name
                st.session_state["model_type"] = model_type

    with tab3:
        if "trained_model" in st.session_state:
            model = st.session_state["trained_model"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            model_name = st.session_state["model_name"]
            model_type = st.session_state["model_type"]
            y_pred = model.predict(X_test)

            st.subheader(f"📊 Model Evaluation: {model_name}")

            if model_type == "Classification":
                st.text("📄 Classification Report:")
                st.text(classification_report(y_test, y_pred))

                labels = sorted(list(set(y_test) | set(y_pred)))
                cm = confusion_matrix(y_test, y_pred, labels=labels)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", xticklabels=labels, yticklabels=labels)
                ax.set(
                    title=f"Confusion Matrix for {model_name}",
                    xlabel="Predicted Label",
                    ylabel="True Label"
                )
                    
                st.pyplot(fig)
            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                st.metric("📈 R²", f"{r2:.4f}")
                st.metric("📉 MAE", f"{mae:.4f}")
                st.metric("📉 MAPE (%)", f"{mape*100:.2f}")
                st.metric("📉 RMSE", f"{rmse:.4f}")

                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot scatter
                sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="royalblue", s=60, alpha=0.7, edgecolor="k")

                # Plot ideal line y = x
                min_val = min(min(y_test), min(y_pred))
                max_val = max(max(y_test), max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal")

                # Set judul dan label
                ax.set_title("🎯 Prediction vs Actual", fontsize=16)
                ax.set_xlabel("Actual", fontsize=12)
                ax.set_ylabel("Prediction", fontsize=12)

                # Anotasi R²
                ax.text(0.05, 0.95, f"$R^2$: {r2:.2f}", transform=ax.transAxes,
                        fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="gray"))

                # Tambah grid dan legend
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="upper left")

                # Format sumbu jika perlu
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

                # Tampilkan plot
                st.pyplot(fig)

            # Download model
            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)

            st.download_button(
                label="💾 Download Model",
                data=buffer,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )
else:
    st.warning("⚠️ Please upload a dataset atau gunakan sample data.")
