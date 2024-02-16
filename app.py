import tensorflow as tf
import pandas as pd
import numpy as np
import torch
import json
import joblib
from keras.models import load_model
from models.SCINet import SCINet

import streamlit as st
st.set_page_config(page_title="Decision Support System", page_icon=":bar_chart:", layout="wide")

def load_models():
    models = {}
    
    checkpoint = torch.load('saved_models/SCINet/checkpoint.pth')
    model = SCINet(output_len=1, input_len=4,input_dim=6, hid_size=1, num_levels=1, dropout=0.5)
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    models['SCINet'] = model
    print('SCINet loaded')

    # Load LSTM model
    modelLSTM = load_model('saved_models/LSTM/LSTM0.h5', compile=False)
    modelLSTM.compile(loss='mean_squared_error', optimizer='adam')
    
    models['LSTM'] = modelLSTM
    print('LSTM loaded')

    # Load RNN model
    modelRNN = load_model('saved_models/RNN/RNN1.h5', compile=False)
    modelRNN.compile(loss='mean_squared_error', optimizer='adam')
    models['RNN'] = modelRNN 
    print('RNN loaded')

    # Load RFR model
    modelRFR = joblib.load('saved_models/RFR/RFR.joblib')
    models['RFR'] = modelRFR
    print('RFR loaded')

    modelNN = tf.saved_model.load('saved_models/NN/NN1.3')
    modelNN = modelNN.signatures['serving_default']
    models['NN'] = modelNN
    print('NN loaded')

    # Load LR model
    # models['LR'] = joblib.load('saved_models/LR/model.pkl')

    # Load DTR model
    # models['DTR'] = joblib.load('saved_models/DTR/model.pkl')

    return models

# load model and data
if 'models' not in st.session_state:
    st.session_state.models = load_models()

if 'df' not in st.session_state:
    st.session_state.df = pd.read_json('datasets/product-data/Full.json')
    st.session_state.pids = st.session_state.df['pid'].unique()
    st.session_state.item = pd.read_csv('datasets/product-data/itemenc.csv')
    st.session_state.prediction = pd.DataFrame(columns=['Mã SP','SCINet', 'LSTM', 'RNN', 'NN', 'RFR', 'Trung bình'])

st.header("Dynamic price forecasting", divider="rainbow")

###################################################
def preprocess_SCINet(df):
    cols = ['rrp', 'competitorPrice', 'click', 'basket', 'order', 'price']
    ret  = df.iloc[-4:][cols].values.reshape((1, 4, 6))
    return ret

# define functions to preprocess and run the model
def predict_SCINet(input):
    x = torch.from_numpy(input).float().cuda()
    return float(models['SCINet'](x)[0][0][5])

def predict_LSTM(df, adFlag, availability):
    cols = ['adFlag', 'availability', 'competitorPrice', 'click', 'basket', 'order', 'price']
    x = df.iloc[-4:]
    x = x.assign(adFlag = float(adFlag), availability = float(availability))
    x = x[cols].values.reshape(1, 4, 7)
    with tf.device('/cpu:0'):
        ret = models['LSTM'].predict(x, verbose=0)[-1]
    return float(ret)

def predict_RNN(df):
    cols = ['rrp', 'competitorPrice', 'click', 'basket', 'order', 'price']
    x  = df.iloc[-8:][cols].values
    ret = models['RNN'].predict(x, verbose=0)[-1]
    return float(ret)

def predict_NN(df, adFlag, availability):
    x = dict(df.iloc[-1])
    details = dict(st.session_state.item[st.session_state.item.pid == x['pid']].iloc[0])
    x.update(details)
    x['adFlag'] = adFlag
    x['availability'] = availability
    x = pd.DataFrame(x, index=[0], dtype='float32')[
        ['pid', 'adFlag', 'availability', 'competitorPrice', 'manufacturer','group', 'content', 'unit', 'pharmForm', 'genericProduct', 'salesIndex', 'category', 'rrp']
    ].values.reshape(1, 13)
    with tf.device('/cpu:0'):
        input_data = tf.convert_to_tensor(x, dtype=tf.float32)
        ret = models['NN'](input_data)['dense_7'].numpy()[-1]
        return float(ret)



def predict_LR(df):
    # x = df.iloc[-1]
    # ret = models['LR'].predict(x)
    pass

def predict_DTR(df):
    # x = df.iloc[-1]
    # ret = models['DTR'].predict(x)
    pass

def predict_RFR(df, adFlag, availability):
    x = dict(df.iloc[-1])
    details = dict(st.session_state.item[st.session_state.item.pid == x['pid']].iloc[0])
    x.update(details)
    x['adFlag'] = adFlag
    x['availability'] = availability
    x = pd.DataFrame(x, index=[0])[
        ['pid', 'adFlag', 'availability', 'competitorPrice', 'manufacturer','group', 'content', 'unit', 'pharmForm', 'genericProduct', 'salesIndex', 'category', 'rrp']]
    return float(models['RFR'].predict(x))

###################################################

models = st.session_state.models

options = st.session_state.pids
selected_pid = st.selectbox('Chọn mã sản phẩm để hiển thị dữ liệu:', options, placeholder="Chọn pid")

st.write("Dữ liệu mới nhất:")

product_df = st.session_state.df[st.session_state.df.pid==selected_pid]
st.dataframe(product_df.tail(8), use_container_width=True)

st.write("Nhập dữ liệu để bổ sung 1 dòng vào dữ liệu:")
cols = st.columns(st.session_state.df.shape[1] + 2)
values = []
for i, col in enumerate(cols):
    with col:
        if i == st.session_state.df.shape[1]:
            values.append(st.text_input(label='adFlag', value=0))
        elif i == st.session_state.df.shape[1] + 1:
            values.append(st.text_input(label='availability', value=1))
        elif st.session_state.df.columns[i] == 'pid':
            values.append(st.text_input(label='pid', value=selected_pid, disabled=True))
        elif st.session_state.df.columns[i] == 'day':
            values.append(st.text_input(label='day', value=int(product_df.iloc[-1].day + 1)))
        else:
            values.append(st.text_input(label=st.session_state.df.columns[i]))

st.session_state.save = st.checkbox('Lưu bảng vào CSDL cho lần sử dụng sau?', value=False)

# Preprocess the input data and run the model for prediction
if st.button('Gợi ý giá cho ngày mai'):
    try:
        prediction = st.session_state.prediction
        new_row= {col: float(value) for col, value in zip(st.session_state.df.columns, values[:-2])}
        st.session_state.df.loc[len(st.session_state.df)] = new_row
    except ValueError:
        st.write('Bạn không nhập đủ đầu vào, không có dòng nào được chèn thêm.')

    current = len(prediction)
    if st.session_state.save:
        st.session_state.df.to_json('datasets/product-data/Full.json', orient='records', indent=4)

    product_df = st.session_state.df[st.session_state.df.pid==selected_pid]
    prediction.loc[current, 'Mã SP'] = selected_pid


    # Run model SCINet
    input_data = preprocess_SCINet(product_df)
    output_SCINet = predict_SCINet(input_data)
    prediction.loc[current, 'SCINet'] = output_SCINet

    # Run model LSTM
    output_LSTM = predict_LSTM(product_df, values[-2], values[-1])
    prediction.loc[current, 'LSTM'] = output_LSTM

    # Run model RNN
    output_RNN = predict_RNN(product_df)
    prediction.loc[current, 'RNN'] = output_RNN

    # Run model NN
    output_NN = predict_NN(product_df, values[-2], values[-1])
    prediction.loc[current, 'NN'] = output_NN

    # # Run model LR
    # output_LR = predict_LR(product_df)
    # prediction.loc[current, 'LR'] = output_LR

    # # Run model DTR
    # output_DTR = predict_DTR(product_df)
    # prediction.loc[current, 'DTR'] = output_DTR

    # Run model RFR
    output_RFR = predict_RFR(product_df, values[-2], values[-1])
    prediction.loc[current, 'RFR'] = output_RFR

    # Calculate mean
    prediction['Trung bình'] = prediction.drop(columns=['Mã SP']).mean(axis=1)

    # Display results
    st.write('Các giá được gợi ý cho ngày mai:')
    st.dataframe(prediction.tail()[::-1], use_container_width=True)



st.header("Hướng dẫn sử dụng")
st.markdown("""
### Hướng dẫn nhanh
- Bấm vào dropdown menu và chọn một product ID để xem dữ liệu tương ứng.
- Nhập thêm 1 dòng dữ liệu mới cho bảng (Các thông tin về sản phẩm).
- Tick chọn checkbox nếu muốn lưu dữ liệu vào CSDL.
- Bấm nút để xem giá gợi ý cho ngày mai bằng nhiều phương pháp.

### Các thông tin trong 1 dòng của bảng
        
- pid: Mã SP
- day: Ngày thứ (tự cộng 1 so với ngày cuối trong bảng)
- adFlag: Có nằm trong chiến dịch quảng cáo? (0,1)
- availability: Tình trạng (1,2,3,4).
- competitorPrice: Giá đối thủ cạnh tranh.
- rrp: Giá tham chiếu.
- click: Tổng số lượt khách click vào SP trong ngày.
- basket: Tổng số lượt cho vào giỏ hàng.
- order: Tổng số lượt đặt hàng.
- *price*: Giá bán.
""")