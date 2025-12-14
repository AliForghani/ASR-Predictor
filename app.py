import streamlit as st
import subprocess
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import numpy as np
import joblib

def normalize_input(input_values):
    input_norm = np.zeros_like(input_values)
    
    input_norm[:, 0] = (input_values[:, 0] - 43) / (230 - 43)
    input_norm[:, 1] = (input_values[:, 1] - 0.5) / (4 - 0.5)
    input_norm[:, 2] = (input_values[:, 2] - 0.1) / (0.5 - 0.1)
    input_norm[:, 3] = (input_values[:, 3] - 50) / (500 - 50)
    input_norm[:, 5] = (input_values[:, 5] - 1) / (3 - 1)

    if storage_option=='0':
        input_norm[:, 4] = (input_values[:, 4] - 0) / (0.01515 - 0)
        input_norm[:, 6] = (input_values[:, 6] - 5) / (50 - 5)
    else:
        input_norm[:, 4] = (input_values[:, 4] - 0) / (0.004 - 0)

    
    return input_norm

# Feedforward function without for loop (using vectorized operations)
def feedforward_vectorized(input_scenarios, W1, W2):
    # Add bias to all input scenarios (bias=1 as the first column)
    input_scenarios_with_bias = np.hstack([np.ones((input_scenarios.shape[0], 1)), input_scenarios])
    print(input_scenarios_with_bias)
    
    # Compute the hidden layer input (input_scenarios_with_bias (Nx8) * W1 (8x30))
    hidden_input = np.dot(input_scenarios_with_bias, W1)

    #Apply sigmoid activation function on hidden layer input
    hidden_output=1 / (1 + np.exp(-hidden_input))

    hidden_output_with_bias = np.hstack([np.ones((hidden_output.shape[0], 1)), hidden_output])


    output = np.dot(hidden_output_with_bias, W2.T)
    
    return output

def run_model(storage_option,k,inj,por,thk,chd,ext_inj,dsp):
    model_weights = joblib.load('model_weights.joblib') 
    
    input_values=[k,inj,por,thk,chd,ext_inj]

    if storage_option=='0':
        input_values.append(dsp)
        
    input_values = np.array([input_values])

    if storage_option=='0':
        W1 = model_weights['W1']
        W2 = model_weights['W2']

    else:
        W1 = model_weights['W1_1year'] 
        W2 = model_weights['W2_1year']


    input_normalized = normalize_input(input_values)
    RENs = feedforward_vectorized(input_normalized, W1, W2)

    required_outputs = {
        'Days': ['30 day',  '60 day',  '90 day', '120 day'],
        'Values': (np.round(100*RENs[0,[1,3,5,7]],1)).tolist()
    }
    required_df = pd.DataFrame(required_outputs)
    return required_df

def create_bar_chart(required_df,storage_option,inj_rate,ext_inj):
    inj_rate=float(inj_rate)
    ext_rate=float(ext_inj)*inj_rate
    if storage_option=='0':
        x_values =np.arange(1,7)
        plot_title='RENs During Extractions (for 2 Months of Injection with No Storage Duration)'
        inj_trace = go.Bar(x=x_values, y=[inj_rate,inj_rate,0,0,0,0], name="Injection", marker_color='blue', width=0.6)
        ext_trace = go.Bar(x=x_values, y=[0,0,-ext_rate,-ext_rate,-ext_rate,-ext_rate ], name="Extraction", marker_color='red', width=0.6)

    else:
        x_values =np.arange(1,19)
        plot_title='RENs During Extractions (for 2 Months of Injection Followed by 1 Year of Storage)'
        inj_trace = go.Bar(x=x_values, y=[inj_rate,inj_rate,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], name="Injection", marker_color='blue', width=0.6)
        ext_trace = go.Bar(x=x_values, y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-ext_rate,-ext_rate,-ext_rate,-ext_rate ], name="Extraction", marker_color='red', width=0.6)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],subplot_titles=(plot_title, "ASR Pumping Scenario"))

    ren_trace = go.Bar(x=required_df['Days'], y=required_df['Values'], text=required_df['Values'], textposition='inside', showlegend =False)
    fig.add_trace(ren_trace, row=1, col=1)
    
    fig.add_trace(inj_trace, row=1, col=2)
    fig.add_trace(ext_trace, row=1, col=2)



    fig.update_layout(
        title={
            # 'text': plot_title,
            'y': 0.95,  # Vertical position of the title (0.0 - bottom, 1.0 - top)
            'x': 0.5,  # Horizontal position (0.5 centers the title)
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'family': 'Arial', 'color': 'black'}
        }
        ,height=550
    )

    fig.update_xaxes(title_text="Extraction Durations (Days)", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=1, col=2,tickmode='array', tickvals=x_values, ticktext=x_values)

    # Update the y-axes titles for both subplots
    fig.update_yaxes(title_text='REN (%)', row=1, col=1,range=[0, 100])
    fig.update_yaxes(title_text='Pumping Rate (cfs)', row=1, col=2)
    return fig


# Configure page settings and hide Streamlit menu/icons
st.set_page_config(
    layout="wide",
    page_title="ASR Performance Predictor",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Enhanced CSS to hide ALL Streamlit branding elements
hide_streamlit_style = """
    <style>
    /* Hide main menu */
    #MainMenu {visibility: hidden !important;}
    
    /* Hide footer */
    footer {visibility: hidden !important;}
    
    /* Hide header */
    header {visibility: hidden !important;}
    
    /* Hide deploy button */
    .stDeployButton {display: none !important;}
    
    /* Hide toolbar */
    section[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    
    /* Hide GitHub icon and menu buttons */
    button[kind="header"] {display: none !important;}
    
    /* Hide the entire header toolbar */
    .stApp header {display: none !important;}
    
    /* Hide settings menu */
    #stDecoration {display: none !important;}
    
    /* Additional targeting for stubborn elements */
    [data-testid="stHeader"] {display: none !important;}
    
    /* Hide any buttons in the top-right */
    section[data-testid="stSidebar"] ~ section > div > div:first-child > div:first-child {
        display: none !important;
    }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Title of the app
st.title("Aquifer Storage and Recovery (ASR) Performance Predictor")

form_col, empty_col=st.columns([2.5,1.55])
with form_col:
    with st.form(key="sum_form"):
        col_radio,col1,col2,col3,col4,col5,col6,col7 = st.columns([2,1,1,1,1,1,1,1])
        with col_radio:
            storage_option = st.radio("Storage Duration:",('No Storage', '1 Year Storage'),horizontal =False,label_visibility='collapsed')
            if storage_option=='No Storage':
                storage_option='0'
            else:
                storage_option='1'

        with col1:
            k = st.slider("Hyd Conductivity", min_value=43.0, max_value=230.0,format="%0.1f", step=0.1, value=100.0 )
        with col2:
            inj = st.slider("Injection Rate",  min_value=0.5, max_value=4.0,step=0.1, value=2.0,format="%0.1f") 
        with col3:
            por=st.slider("Porosity",  min_value=0.1, max_value=0.5,step=0.01, value=0.3,format="%0.2f")
        with col4:
            thk=st.slider("Layer Thickness",  min_value=50.0, max_value=500.0,step=0.1, value=200.0,format="%0.1f")
        with col5:
            chd = st.slider("Gradient",  min_value=0.0, max_value=0.004, step=0.0001,format="%0.4f", value=0.002) 
        with col6:
            ext_inj = st.slider("EXT/Inj Ratio",  min_value=1.0, max_value=3.0,step=0.01, value=2.0,format="%0.1f") 
        with col7:
            dsp = st.slider("Long-Dispersivity",  min_value=5.0, max_value=50.0,step=0.1, value=25.0,format="%0.1f")

        submit_button = st.form_submit_button(label="Submit")

     
# After form submission
if submit_button:
    required_df=run_model(storage_option,k,inj,por,thk,chd,ext_inj,dsp)

    fig = create_bar_chart(required_df,storage_option,inj,ext_inj)
    st.plotly_chart(fig)




