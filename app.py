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
    page_title="Regression Model Builder | www.anodra.uz",
    page_icon="🚀",
    layout="centered", 
    initial_sidebar_state="auto"  
)


def get_df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue().split('\n')
    col_info = [x.split(maxsplit=4) for x in s[3:-2]]
    col_info_df = pd.DataFrame(col_info, columns=["#", "Columns", "Non-Null Count", "Dtype", "Details"])
    return col_info_df


def handle_missing_values(df, strategy):
    if strategy == "Drop missing values":
        df = df.dropna()
    elif strategy == "Fill with mean value":
        df = df.fillna(df.mean())
    elif strategy == "Fill with median value":
        df = df.fillna(df.median())
    elif strategy == "Fill with mode value":
        df = df.fillna(df.mode().iloc[0])
    return df


st.title('Regression Model Builder')


st.sidebar.title("Regression Model Builder")
st.sidebar.markdown("<h4 style='color: blue;'>Creator: <a href=`https://t.me/yagafarov`>Dinmukhammad Yagafarov</a></h4>", unsafe_allow_html=True)
st.sidebar.markdown("---")


uploaded_file = st.sidebar.file_uploader("Upload data (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type == 'xlsx':
        df = pd.read_excel(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(df.head())

    st.write("Data Information:")
    df_info = get_df_info(df)
    st.table(df_info)


    missing_value_strategy = st.sidebar.selectbox("How to handle missing values?", 
                                                  ["Drop missing values", "Fill with mean value", "Fill with median value", "Fill with mode value"])
    df = handle_missing_values(df, missing_value_strategy)


    columns_to_drop = st.sidebar.multiselect("Select columns to delete", df.columns)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        st.write("Updated Data:")
        st.dataframe(df.head())

if 'df' in locals() and not df.empty:

    selected_columns = st.sidebar.multiselect("Select columns to convert", df.columns)
    for column in selected_columns:
        if df[column].dtype == object:
            unique_values = df[column].unique()
            value_mapping = {val: idx for idx, val in enumerate(unique_values)}
            df[column] = df[column].map(value_mapping)


    target_column = st.sidebar.selectbox("Select the target column for prediction", df.columns)


    model_type = st.sidebar.selectbox("Select model type", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest Regression", "Decision Tree Regression"])


    train_set, test_set = train_test_split(df, test_size=0.15, random_state=28)
    x_train = train_set.drop(target_column, axis=1).values
    y_train = train_set[target_column].values
    x_test = test_set.drop(target_column, axis=1).values
    y_test = test_set[target_column].values


    if model_type == "Linear Regression":
        model = linear_model.LinearRegression()
    elif model_type == "Ridge Regression":
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
        model = linear_model.Ridge(alpha=alpha)
    elif model_type == "Lasso Regression":
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
        model = linear_model.Lasso(alpha=alpha)
    elif model_type == "Random Forest Regression":
        n_estimators = st.sidebar.slider("Number of estimators", 10, 500, 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=28)
    elif model_type == "Decision Tree Regression":
        max_depth = st.sidebar.slider("Max depth", 1, 50, 10)
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=28)

    model.fit(x_train, y_train)


    y_predict = model.predict(x_test)


    MAE = mean_absolute_error(y_test, y_predict)
    RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)

    st.write(f"Coefficients: {model.coef_ if hasattr(model, 'coef_') else 'N/A (Random Forest)'}")
    st.write(f"MAE: {MAE}")
    st.write(f"RMSE: {RMSE}")
    st.write(f"R^2: {r2}")


    y_train_p = model.predict(x_train)
    y_test_p = model.predict(x_test)

    fig, ax = plt.subplots()
    ax.scatter(y_train, y_train_p, color='blue', label="Train data", alpha=0.6)
    ax.scatter(y_test, y_test_p, color='red', label="Test data", alpha=0.6)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='green', linestyle='--', linewidth=2, label="Prediction line")

    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title("Prediction Line for Selected Model")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    



    st.write("### Test the model with new data")
    test_data = []
    for col in df.drop(columns=[target_column]).columns:
        user_input = st.number_input(f"Enter value for {col}", value=0.0)
        test_data.append(user_input)
    
    if st.button('Predict'):
        prediction = model.predict([test_data])
        st.write(f"The predicted value is: {prediction[0]}")


        user_test_MAE = mean_absolute_error([prediction[0]], [y_test.mean()]) 
        user_test_r2 = r2_score([prediction[0]], [y_test.mean()])

        st.write(f"Test MAE for the entered data: {user_test_MAE}")
        st.write(f"Test R² for the entered data: {user_test_r2}")

    

    if st.button('Save and Download the Model'):

        filename = 'model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        with open(filename, 'rb') as f:
            st.download_button('Download the model (.pkl)', f, file_name=filename)
