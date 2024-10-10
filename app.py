import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def setup_preprocessor(pre_process_df):
    global numeric_features, categorical_features, preprocessor, numeric_feature_names, categorical_feature_names

    # Define features
    numeric_features = list(pre_process_df.drop(['Агрофон'], axis=1)
                           .select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = ['Агрофон']

    # Create and fit preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Fit preprocessor
    preprocessor.fit(pre_process_df)

    # Get feature names
    numeric_feature_names = numeric_features
    categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

    return preprocessor

def initialize_model():
    global lgbfit, model_params
    
    pre_process_df = pd.read_csv('test.csv')  # or however you load your data
    
    # Setup preprocessor
    setup_preprocessor(pre_process_df)
    
    # Load model
    lgbfit = lgb.Booster(model_file=r'C:\Users\Dgebe N\.cursor-tutor\projects\python\Field_Analysis\lgbfit.txt')
    try:
        with open(r'C:\Users\Dgebe N\.cursor-tutor\projects\python\Field_Analysis\lgbfit_params.json', 'r') as f:
            model_params = json.load(f)
    except Exception as e:
        print("Error Loading Parameters")

initialize_model()

df_main = pd.read_csv(r'new_ml_data.csv')

def main():
    st.title('Yield Crop Prediction')
    html_temp='''
    <div style='background-color:red; padding:12px'>
    <h1 style='color:  #000000; text-align: center;'>Yield Crop Prediction Machine Learning Model</h1>
    </div>
    <h2 style='color:  red; text-align: center;'>Please Enter Input</h2>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    farm= st.selectbox("Type or Select a Farm_ID.",df_main['Подразделение'].unique()) 
    field= st.selectbox("Type or Select a Field_ID",df_main['Поле'].unique())
    Max_NDVI=st.number_input('Enter Max_NDVI May(MAX_NDVI).',value=None)
    July_NDVI=st.number_input('Enter July NDVI (7_NDVI).',value=None)
    relative_humidity_June=st.number_input('Enter Relative Humidity May(6_relative_humidity).',value=None)
    relative_humidity_July=st.number_input('Enter Relative Humidity July (7_relative_humidity).',value=None)
    min_temperature_July = st.number_input('Minimum Temperature July (7_temperature_2m_min).',value=None)
    total_rainfall_May=st.number_input('Enter Total Rainfall May(5_total_precipitation_sum).',value=None)
    total_rainfall_July=st.number_input('Enter Total Rainfall July (7_total_precipitation_sum).',value=None)
    wind_speed_May = st.number_input('Enter Wind Speed May (5_v_component_of_wind_10m).',value=None)
    VPD_May = st.number_input('Enter VPD May (5_vapor_pressure_deficit).',value=None)
    VPD_June = st.number_input('Enter VPD June (6_vapor_pressure_deficit).',value=None)
    sowing_date = st.number_input('Enter Sowing Date (DOY_min).',value=None)
    cec = st.number_input('Enter Cec (cec).',value=None)
    clay = st.number_input('Enter Clay (clay).',value=None)
    sand = st.number_input('Enter Sand (sand).',value=None)
    silt = st.number_input('Enter Silt (silt).',value=None)
    NDVI_197 = st.number_input('Enter 2nd Week of July NDVI (193_NDVI)',value=None)
    NDVI_201 = st.number_input('Enter 3rd Week of July NDVI (201_NDVI).',value=None)
    NDVI_209 = st.number_input('Enter 4th Week of July NDVI (209_NDVI).',value=None)
    agrofon= st.selectbox("Type or Select a Field_ID",df_main['Агрофон'].unique())
        # Collect user inputs into a horizontal list for predictions
    input = [farm, field, Max_NDVI, July_NDVI, relative_humidity_June, relative_humidity_July, 
             min_temperature_July, total_rainfall_May, total_rainfall_July, wind_speed_May, 
             VPD_May, VPD_June, sowing_date, cec, clay, sand, silt, NDVI_197, NDVI_201, 
             NDVI_209, agrofon]

    result=''
    if st.button('Predict',''):
        result=prediction(input)
    temp='''
     <div style='background-color:navy; padding:8px'>
     <h1 style='color: gold  ; text-align: center;'>{}</h1>
     </div>
     '''.format(result)
    st.markdown(temp,unsafe_allow_html=True)


def prediction(input_data):
    # Convert input list to a dictionary
    input_dict = {
        'MAX_NDVI': input_data[2],
        '7_NDVI': input_data[3],
        '6_relative_humidity': input_data[4],
        '7_relative_humidity': input_data[5],
        '7_temperature_2m_min': input_data[6],
        '5_total_precipitation_sum': input_data[7],
        '7_total_precipitation_sum': input_data[8],
        '5_v_component_of_wind_10m': input_data[9],
        '5_vapor_pressure_deficit': input_data[10],
        '6_vapor_pressure_deficit': input_data[11],
        'DOY_min': input_data[12],
        'cec': input_data[13],
        'clay': input_data[14],
        'sand': input_data[15],
        'silt': input_data[16],
        '193_NDVI': input_data[17],
        '201_NDVI': input_data[18],
        '209_NDVI': input_data[19],
        'Агрофон': input_data[20]
    }
    
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    try:
        # Transform the input data using the preprocessor
        processed_input = preprocessor.transform(input_df)
        
        # Convert to DataFrame with correct column names
        processed_df = pd.DataFrame(
            processed_input, 
            columns=numeric_feature_names + categorical_feature_names.tolist()
        )
        
        # Make prediction
        prediction = lgbfit.predict(processed_df)[0]
        
        return f"Predicted Yield: {prediction:.2f}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

if __name__=='__main__':
    main()
