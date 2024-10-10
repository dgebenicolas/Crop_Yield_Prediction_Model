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
    lgbfit = lgb.Booster(model_file=r'lgbfit.txt')
    try:
        with open(r'lgbfit_params.json', 'r') as f:
            model_params = json.load(f)
    except Exception as e:
        print("Error Loading Parameters")

initialize_model()

df_main = pd.read_csv(r'new_ml_data.csv')

def main():
    st.title('Yield Crop Prediction')
    html_temp = '''
    <div style='background-color:red; padding:12px'>
    <h1 style='color: #000000; text-align: center;'>Yield Crop Prediction Machine Learning Model</h1>
    </div>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Explanation for users
    st.markdown("""
    ### How to Use:
    1. Paste your list of values in the text area below
    2. The list should contain 19 numeric values in the following order:
        - MAX_NDVI, 7_NDVI, 6_relative_humidity, 7_relative_humidity, 7_temperature_2m_min
        - 5_total_precipitation_sum, 7_total_precipitation_sum, 5_v_component_of_wind_10m
        - 5_vapor_pressure_deficit, 6_vapor_pressure_deficit, DOY_min, cec, clay, sand, silt
        - 193_NDVI, 201_NDVI, 209_NDVI
    3. Select the Agrofon value from the dropdown
    4. Click 'Predict' to get your result!
    """)

    # Example for users
    st.markdown("""
    #### Example input:
    ```
    0.76, 0.65, 70.5, 65.2, 15.3, 45.2, 50.1, 2.3, 0.8, 1.2, 125, 15.6, 25.3, 40.2, 34.5, 0.62, 0.58, 0.55
    ```
    """)

    # Text area for list input
    input_text = st.text_area("Enter your values (comma-separated):", height=100)
    
    # Dropdown for Agrofon
    agrofon = st.selectbox("Select Agrofon:", df_main['Агрофон'].unique())
    
    # Process input when Predict button is clicked
    if st.button('Predict'):
        try:
            # Parse input text to list
            input_values = [float(x.strip()) for x in input_text.split(',') if x.strip()]
            
            if len(input_values) != 18:  # 18 because we're not counting farm and field IDs
                st.error(f"Please enter exactly 18 values. You entered {len(input_values)} values.")
                return
            
            # Add placeholder values for farm and field (indices 0 and 1)
            full_input = ['placeholder', 'placeholder'] + input_values + [agrofon]
            
            result = prediction(full_input)
            
            # Display result
            result_html = f'''
            <div style='background-color:navy; padding:8px'>
            <h1 style='color: gold; text-align: center;'>{result}</h1>
            </div>
            '''
            st.markdown(result_html, unsafe_allow_html=True)
            
        except ValueError:
            st.error("Please ensure all values are numbers and are correctly formatted (comma-separated).")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add a section showing the expected order of variables
    with st.expander("See detailed description of input values"):
        st.markdown("""
        ### Expected order of values:
        1. MAX_NDVI (May)
        2. July NDVI (7_NDVI)
        3. Relative Humidity June (6_relative_humidity)
        4. Relative Humidity July (7_relative_humidity)
        5. Minimum Temperature July (7_temperature_2m_min)
        6. Total Rainfall May (5_total_precipitation_sum)
        7. Total Rainfall July (7_total_precipitation_sum)
        8. Wind Speed May (5_v_component_of_wind_10m)
        9. VPD May (5_vapor_pressure_deficit)
        10. VPD June (6_vapor_pressure_deficit)
        11. Sowing Date (DOY_min)
        12. Cec (cec)
        13. Clay (clay)
        14. Sand (sand)
        15. Silt (silt)
        16. 2nd Week of July NDVI (193_NDVI)
        17. 3rd Week of July NDVI (201_NDVI)
        18. 4th Week of July NDVI (209_NDVI)
        """)


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
