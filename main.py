import streamlit as st
import pandas as pd
import numpy as np

from function import train_test_split, auto_reg_modelling, evaluate_model


st.title('Time Series Feature Profiler')

st.sidebar.write('App Controls')

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
    
    st.write('Data:')
    st.dataframe(final_df,height=200,use_container_width=True)
    st.line_chart(data=df, height=300, y=val_col, use_container_width=True)
    
    final_df['date'] = pd.to_datetime(final_df['date'])
    data, train, test = train_test_split(final_df)
    
    main_train = train[['date', 'data']].set_index('date')
    exog_train = train.drop(['data'], axis=1).set_index('date')
    
    main_test = test[['date', 'data']].set_index('date')
    exog_test = test.drop(['data'], axis=1).set_index('date')
    
    prediction = auto_reg_modelling(main_train, main_test, exog_train, exog_test)
    test_set = test.set_index('date')[['data']]
    accuracy, mae, mse, rmse = evaluate_model(test_set, prediction)
    
    st.write("Metrics:", accuracy, mae, mse, rmse)
    
    
    full_data = train.append(test)
    full_data['train'] = train[val_col]
    full_data['test'] = test[val_col]
    full_data = full_data.drop([val_col], axis=1)
    print(full_data)
    st.line_chart(full_data)