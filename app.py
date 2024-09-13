import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import io

st.set_page_config(
    page_title="RegMod | www.anodra.uz",
    page_icon="🚀",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto"  # or "expanded" or "collapsed"
)
# DataFrame haqida ma'lumot olish funksiyasi
def get_df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue().split('\n')
    col_info = [x.split(maxsplit=4) for x in s[3:-2]]
    col_info_df = pd.DataFrame(col_info, columns=["#", "Columns", "Non-Null Count", "Dtype", "Details"])
    return col_info_df

# Yo'q bo'lib ketgan qiymatlarni to'ldirish funksiyasi
def handle_missing_values(df, strategy):
    if strategy == "Discard missing values":
        df = df.dropna()
    elif strategy == "Fill with average value":
        df = df.fillna(df.mean())
    elif strategy == "Fill with the median value":
        df = df.fillna(df.median())
    elif strategy == "Fill with the moda value":
        df = df.fillna(df.mode().iloc[0])
    return df

# Streamlit app
st.title('Regression model builder')



# Sidebarni yaratish
st.sidebar.title("Regression model builder")
st.sidebar.markdown("<h4 style='color: blue;'>Creator: <a href=`https://t.me/yagafarov`>Dinmukhammad Yagafarov</a></h4>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Ma'lumotlarni yuklash
uploaded_file = st.sidebar.file_uploader("Upload data (excel or csv)", type=["csv", "xlsx"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type == 'xlsx':
        df = pd.read_excel(uploaded_file)

    st.write("Uploaded data:")
    st.dataframe(df.head())

    st.write("About data:")
    df_info = get_df_info(df)
    st.table(df_info)

    # Yo'q bo'lib ketgan qiymatlarni to'ldirish
    missing_value_strategy = st.sidebar.selectbox("How to fill in missing values?", 
                                          ["Drop missing values", "Fill with mean value", "Fill with median value", "Fill with mode value"])
    df = handle_missing_values(df, missing_value_strategy)

    # Keraksiz ustunlarni olib tashlash
    columns_to_drop = st.sidebar.multiselect("Select the columns to delete", df.columns)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        st.write("Updated information:")
        st.dataframe(df.head())

    st.write("Updated information:")
    st.dataframe(df.head())

if 'df' in locals() and not df.empty:
    # Kodlash
    selected_columns = st.sidebar.multiselect("Select the columns to convert", df.columns)
    for column in selected_columns:
        if df[column].dtype == object:
            unique_values = df[column].unique()
            value_mapping = {val: idx for idx, val in enumerate(unique_values)}
            df[column] = df[column].map(value_mapping)

    # Bashorat qilinuvchi ustunni tanlash
    target_column = st.sidebar.selectbox("Select the predicted column", df.columns)

    # Model tanlash
    model_type = st.sidebar.selectbox("Select the model type", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest Regression","Decision Tree Regression"])

    # Ma'lumotlarni trenlash va test qismiga bo'lib bo'linishi
    train_set, test_set = train_test_split(df, test_size=0.15, random_state=28)
    x_train = train_set.drop(target_column, axis=1).values
    y_train = train_set[target_column].values
    x_test = test_set.drop(target_column, axis=1).values
    y_test = test_set[target_column].values

    # Tanlangan modelni trenlash
    if model_type == "Linear Regression":
        model = linear_model.LinearRegression()
    elif model_type == "Ridge Regression":
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
        model = linear_model.Ridge(alpha=alpha)
    elif model_type == "Lasso Regression":
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
        model = linear_model.Lasso(alpha=alpha)
    elif model_type == "Random Forest Regression":
        n_estimators = st.sidebar.slider("Estimatorlar soni", 10, 500, 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=28)
    elif model_type == "Decision Tree Regression":
        max_depth = st.sidebar.slider("Max depth", 1, 50, 10)
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=28)

    model.fit(x_train, y_train)

    # Bashorat qilinuvchi ustunni aniqlash
    y_predict = model.predict(x_test)

    # Modelni baholash
    MAE = mean_absolute_error(y_test, y_predict)
    RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)

    st.write(f"Coefficients: {model.coef_ if hasattr(model, 'coef_') else 'N/A (Random Forest)'}")
    st.write(f"MAE: {MAE}")
    st.write(f"RMSE: {RMSE}")
    st.write(f"R^2: {r2}")

    # Natijalarni chizish
    y_train_p = model.predict(x_train)
    y_test_p = model.predict(x_test)

    fig, ax = plt.subplots()
    ax.scatter(y_train, y_train_p, color='blue', label="Train data", alpha=0.6)
    ax.scatter(y_test, y_test_p, color='red', label="Test data", alpha=0.6)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='green', linestyle='--', linewidth=2, label="Prediction line")

    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title("Prediction line for selected model")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Modelni diskga saqlash
    filename = 'model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Modelni yuklab olish tugmasi
    with open(filename, 'rb') as f:
        st.download_button('Download the model(.pkl)', f, file_name=filename)
