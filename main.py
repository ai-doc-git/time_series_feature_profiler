import streamlit as st
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from function import train_test_split, auto_reg_modelling, evaluate_model, load_lottieurl

# Page Congifuration
st.set_page_config(
    page_title='TemporalSpider Analytics', 
    page_icon='🕐', 
    layout='wide'
)

st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> TemporalSpider Analytics </h1>", unsafe_allow_html=True)

lottie_url = "https://lottie.host/7bab6b92-1b21-4d2c-8e42-29a46211da1c/G8O2EyloAI.json"
_,img_col, _ = st.columns((0.5,2,0.5))

with img_col:
    lottie_hello = load_lottieurl(lottie_url)
    st_lottie(lottie_hello, key="user")
    
st.sidebar.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> App Controls </h1>", unsafe_allow_html=True)
st.sidebar.markdown("----")
file = st.sidebar.file_uploader("Upload time-series data:")

if file is not None:
    df = pd.read_csv(file)
    val_col = df.columns[1]
    
    data = {}
    for i in range(20):
        col = f'col{i}'
        data[col]= range(10)
    df2 = pd.DataFrame(df)

    columns = st.sidebar.multiselect("Columns:",df2.columns)
    filter = st.sidebar.radio("Choose by:", ("inclusion","exclusion"))

    if filter == "exclusion":
        columns = [col for col in df2.columns if col not in columns]

    final_df = df2[columns]
    
    if final_df.shape[1] > 1:
        st.markdown("<h2 style='text-align: center; color: rgb(0, 0, 0);'> Input Data </h2>", unsafe_allow_html=True)
        st.markdown("----")
        st.dataframe(final_df,height=200,use_container_width=True)
        st.line_chart(data=final_df, height=300, y=val_col, use_container_width=True)

        st.markdown("<h2 style='text-align: center; color: rgb(0, 0, 0);'> Forecast </h2>", unsafe_allow_html=True)
        st.markdown("----")
        final_df['date'] = pd.to_datetime(final_df['date'])
        data, train, test = train_test_split(final_df)

        main_train = train[['date', 'data']].set_index('date')
        exog_train = train.drop(['data'], axis=1).set_index('date')

        main_test = test[['date', 'data']].set_index('date')
        exog_test = test.drop(['data'], axis=1).set_index('date')

        prediction = auto_reg_modelling(main_train, main_test, exog_train, exog_test)
        test_set = test.set_index('date')[['data']]
        accuracy, mae, mse, rmse = evaluate_model(test_set, prediction)

        full_data = main_train.append(main_test)
        full_data['train'] = main_train['data']
        full_data['test'] = main_test['data']
        full_data['forecast'] = prediction['data']
        full_data = full_data.drop(['data'], axis=1)
        st.line_chart(full_data)

        st.markdown("<h2 style='text-align: center; color: rgb(0, 0, 0);'> Evaluation Result </h2>", unsafe_allow_html=True)
        st.markdown("----")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        mcol1.metric("Accuracy", accuracy)
        mcol2.metric("MAE", mae)
        mcol3.metric("MSE", mse)
        mcol4.metric("RMSE", rmse)