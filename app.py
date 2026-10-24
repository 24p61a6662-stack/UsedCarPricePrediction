
import streamlit as st
import pandas as pd
import numpy as np
import io

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Price Prediction")
st.write("Predict a used car's selling price using Linear Regression.")

# -------------------------------
# 2. Dataset upload
# -------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload your car dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your used car dataset CSV in the sidebar.")
    st.code(
        "Year, Selling_Price, Kms_Driven, Present_Price, "
        "Fuel_Type, Seller_Type, Transmission, Owner"
    )
    st.stop()

try:
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

# -------------------------------
# 3. Clean column names
# -------------------------------
df.columns = [
    str(col).strip().replace(" ", "_")
    for col in df.columns
]

column_aliases = {
    "name": "Car_Name",
    "car_name": "Car_Name",
    "year": "Year",
    "selling_price": "Selling_Price",
    "km_driven": "Kms_Driven",
    "kms_driven": "Kms_Driven",
    "present_price": "Present_Price",
    "fuel": "Fuel_Type",
    "fuel_type": "Fuel_Type",
    "seller_type": "Seller_Type",
    "transmission": "Transmission",
    "owner": "Owner"
}

df.rename(
    columns={
        col: column_aliases[col.lower()]
        for col in df.columns
        if col.lower() in column_aliases
    },
    inplace=True
)

if df.columns.duplicated().any():
    st.error("Duplicate columns found after cleaning column names.")
    st.stop()

if "Selling_Price" not in df.columns:
    st.error(
        "Selling_Price column not found. "
        "Please check your dataset."
    )
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# -------------------------------
# 4. Prepare the data
# -------------------------------
df["Selling_Price"] = pd.to_numeric(
    df["Selling_Price"], errors="coerce"
)

df = df.dropna(subset=["Selling_Price"])
df = df[df["Selling_Price"] >= 0].copy()

df.dropna(
    how="all",
    subset=[c for c in df.columns if c != "Selling_Price"],
    inplace=True
)

if len(df) < 5:
    st.error("At least 5 valid rows are required.")
    st.stop()

# Remove car name because it may be too specific
X = df.drop(
    columns=["Selling_Price", "Car_Name"],
    errors="ignore"
)

y = df["Selling_Price"]

if X.shape[1] == 0:
    st.error("No usable input features were found.")
    st.stop()

# Convert numeric-looking text columns where appropriate
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.9:
            X[col] = converted

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# -------------------------------
# 5. Display dataset
# -------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.write("Dataset shape:", df.shape)
st.write("Numerical features:", numeric_features)
st.write("Categorical features:", categorical_features)

with st.expander("View missing values"):
    st.dataframe(df.isnull().sum().rename("Missing Values"))

# -------------------------------
# 6. Preprocessing pipelines
# -------------------------------
transformers = []

if numeric_features:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

# -------------------------------
# 7. Split dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=45
)

# -------------------------------
# 8. Train Linear Regression
# -------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

try:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

# -------------------------------
# 9. Evaluate model
# -------------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float("nan")

st.subheader("Model Performance: Linear Regression")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3.metric("R² Score", f"{r2:.3f}")

st.caption(
    "Lower MAE and RMSE generally indicate smaller errors. "
    "R² closer to 1 generally indicates a better fit."
)

# -------------------------------
# 10. User input form
# -------------------------------
st.subheader("Predict a Used Car Price")
st.write("Enter the car details below.")

with st.form("car_prediction_form"):
    input_values = {}
    columns = st.columns(2)

    for i, col in enumerate(X.columns):
        with columns[i % 2]:
            if col in numeric_features:
                series = pd.to_numeric(
                    X[col], errors="coerce"
                ).dropna()

                median = float(series.median()) if not series.empty else 0.0
                minimum = float(series.min()) if not series.empty else 0.0
                maximum = float(series.max()) if not series.empty else 1.0

                if minimum == maximum:
                    maximum = minimum + 1.0

                default = min(max(median, minimum), maximum)

                step = (
                    1.0
                    if col.lower() in [
                        "year", "owner", "kms_driven", "km_driven"
                    ]
                    else 0.1
                )

                input_values[col] = st.number_input(
                    col.replace("_", " "),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    key=f"input_{col}"
                )
            else:
                options = sorted(
                    X[col].dropna().astype(str).unique().tolist()
                )

                if not options:
                    options = ["Unknown"]

                input_values[col] = st.selectbox(
                    col.replace("_", " "),
                    options,
                    key=f"input_{col}"
                )

    submitted = st.form_submit_button(
        "Predict Selling Price",
        type="primary"
    )

# -------------------------------
# 11. Show prediction
# -------------------------------
if submitted:
    new_car = pd.DataFrame(
        [input_values],
        columns=X.columns
    )

    try:
        predicted_price = float(model.predict(new_car)[0])

        st.success("Prediction completed!")
        st.metric(
            "Estimated Selling Price",
            f"{predicted_price:,.2f}"
        )

        st.dataframe(new_car, use_container_width=True)

        st.caption(
            "The price uses the same units as Selling_Price "
            "in your dataset. This is an estimate, not a guaranteed "
            "market price."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# -------------------------------
# 12. Actual vs predicted graph
# -------------------------------
st.subheader("Actual vs Predicted Prices")

chart_df = pd.DataFrame({
    "Actual Price": np.asarray(y_test),
    "Predicted Price": predictions
})

st.scatter_chart(
    chart_df,
    x="Actual Price",
    y="Predicted Price"
)

# -------------------------------
# 13. Project information
# -------------------------------
with st.expander("About this project"):
    st.write("""
    **Algorithm:** Linear Regression

    **Preprocessing:** Missing-value imputation,
    feature scaling and categorical encoding.

    **Evaluation metrics:** MAE, RMSE and R² Score.

    **Dataset target:** Selling_Price.

    The model is trained on the uploaded dataset and
    predicts prices using the same target-price units.
    """)
