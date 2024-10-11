import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.express as px
from utils import (
    setup_preprocessor, check_csv_format, process_data, 
    map_agrofon_to_group, REQUIRED_COLUMNS, COLUMN_DTYPES
)

def load_model():
    try:
        model = lgb.Booster(model_file='lgbfit.txt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title('Crop Yield Prediction')
    
    # Load model and preprocessor
    model = load_model()
    if model is None:
        return
    
    preprocessor, numeric_features, categorical_features = setup_preprocessor(None)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Check CSV format
        is_valid, result = check_csv_format(uploaded_file)
        if not is_valid:
            st.error(result)
            return
        
        df = result  # result is the DataFrame if validation passed
        
        # Process data
        id_columns, process_df = process_data(df)
        
        # Map Agrofon to groups
        process_df = map_agrofon_to_group(process_df)
        
        # Preprocess data using fit_transform
        processed_data = preprocessor.transform(process_df)
        
        # Create DataFrame with processed data
        feature_names = (numeric_features + 
                        preprocessor.named_transformers_['cat'].get_feature_names(categorical_features).tolist())
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
        
        # Make predictions
        predictions = model.predict(processed_df)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Farm ID': id_columns['Подразделение'],
            'Field ID': id_columns['Поле'],
            'Predicted_Yield': predictions
        })
        
        # Display results table
        st.subheader("Predictions Table")
        st.dataframe(results_df)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(results_df['Predicted_Yield'].describe())
        
        # Histogram
        st.subheader("Prediction Distribution")
        fig = px.histogram(results_df, x='Predicted_Yield', 
                          title='Distribution of Predicted Yields')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()

