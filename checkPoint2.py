import streamlit as st
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
import joblib

st.markdown("<h1 style = 'color: #008DDA; text-align: center; font-family: Helvetica'>FINANCIAL INCLUSION DATASET</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFB000; text-align: center; font-family: Brush Script MT'> Built By: ADEWALE JOLAYEMI</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (11).png', use_column_width=True)
st.header('Project Background Information', divider = True)
st.write("Financial inclusion aims to provide access to financial services for underserved populations. Machine learning plays a vital role in this by enabling credit scoring without traditional data, detecting fraud, segmenting customers, managing risks, offering personalized recommendations, monitoring transactions, facilitating alternative lending, and enhancing customer service through natural language processing. Collaboration and ethical considerations are key for successful implementation.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

df = pd.read_csv('Financial_inclusion_dataset.csv')
st.dataframe(df)

st.sidebar.image('pngwing.com (12).png', caption='Welcome User')

st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Decleare user input variables
st.sidebar.subheader('Input Variable', divider=True)
sel_cols = ['age_of_respondent', 'household_size', 'job_type', 'education_level',
            'marital_status', 'country', 'location_type', 'bank_account']
age = st.sidebar.number_input('age_of_respondent')
household = st.sidebar.number_input('household_size')
job = st.sidebar.selectbox('job_type', df['job_type'].unique())
education = st.sidebar.selectbox('education_level', df['education_level'].unique())
mariStatus = st.sidebar.selectbox('marital_status', df['marital_status'].unique())
country = st.sidebar.selectbox('country', df['country'].unique())
location = st.sidebar.selectbox('location_type', df['location_type'].unique())

# Display the users-input
input_var = pd.DataFrame()
input_var['age_of_respondent'] = [age]
input_var['household_size'] = [household]
input_var['job_type'] = [job]
input_var['education_level'] = [education]
input_var['marital_status'] = [mariStatus]
input_var['country'] = [country]
input_var['location_type'] = [location]

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('User Input Variables', divider=True)
st.dataframe(input_var, use_container_width=True)

# Importing Transformers
country = joblib.load('country_encoder.pkl')
education = joblib.load('education_level_encoder.pkl')
job = joblib.load('job_type_encoder.pkl')
location = joblib.load('location_type_encoder.pkl')
mariStatus = joblib.load('marital_status_encoder.pkl')

# Applying transformations to the user's
input_var['country'] = country.transform(input_var[['country']])
input_var['education_level'] = education.transform(input_var[['education_level']])
input_var['job_type'] = job.transform(input_var[['job_type']])
input_var['location_type'] = location.transform(input_var[['location_type']])
input_var['marital_status'] = mariStatus.transform(input_var[['marital_status']])


# st.dataframe(input_var)

model = joblib.load('Finance.pkl')
prediction = model.predict(input_var)

if st.button('ACTIVE BANK ACCOUNT'):
    if prediction[0] == 0:
        st.error('NO')
        
    else:
        st.success('YES') 