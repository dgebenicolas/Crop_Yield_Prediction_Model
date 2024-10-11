import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.express as px
import json
from utils import (
    setup_preprocessor, check_csv_format, process_data, 
    map_agrofon_to_group, REQUIRED_COLUMNS, COLUMN_DTYPES,
    process_data_yield
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
        has_yield = 'Yield' in df.columns
        
        if has_yield:
            id_columns, process_df = process_data_yield(df)
        else:
            id_columns, process_df = process_data(df)
        
        # Map Agrofon to groups
        process_df = map_agrofon_to_group(process_df)
        
        # Preprocess data using fit_transform
        processed_data = preprocessor.transform(process_df)
        
        # Create DataFrame with processed data
        feature_names = (numeric_features + 
                        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
        
        # Make predictions
        predictions = model.predict(processed_df)
        
        # Create results DataFrame
        results_df = id_columns.copy()
        results_df['Predicted_Yield'] = predictions
        
# Display results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predictions Table")
            st.dataframe(
                results_df.style.format({'Predicted_Yield': '{:.2f}'}),
                height=400,  # Adjust height as needed
                use_container_width=True
            )
        
        with col2:
            st.subheader("Summary Statistics")
            stats_df = pd.DataFrame(results_df['Predicted_Yield'].describe())
            stats_df.columns = ['Value']
            stats_df['Value'] = stats_df['Value'].apply(lambda x: f'{x:.2f}')
            
            # Style the statistics table
            st.dataframe(
                stats_df.style.set_properties(**{
                    'background-color': '#f0f2f6',
                    'color': 'black',
                    'border': '1px solid darkgrey',
                    'padding': '12px'
                }),
                use_container_width=True
            )
        
        # Histogram
        st.subheader("Prediction Distribution")
        fig = px.histogram(
            results_df, 
            x='Predicted_Yield',
            nbins=100,
            title='Distribution of Predicted Yields',
            color_discrete_sequence=['#3498db'],
            template='simple_white'
        )
        
        fig.update_layout(
            xaxis_title="Predicted Yield",
            yaxis_title="Count",
            showlegend=False,
            xaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
            title=dict(font=dict(size=16))
        )
        
        mean_yield = results_df['Predicted_Yield'].mean()
        fig.add_vline(
            x=mean_yield, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {mean_yield:.2f}",
            annotation_position="top"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Choropleth Map
        st.subheader("Predicted Yield Map")
        
        try:
            with open('With_Holes_FIELDS_Geo_Boundaries__2024.geojson', 'r') as f:
                geojson_data = json.load(f)
                
            map_data = results_df.copy()
            map_data = map_data[['Подразделение', 'Field_ID', 'Predicted_Yield']]
            
            fig_map = px.choropleth_mapbox(
                map_data, 
                geojson=geojson_data, 
                locations='Field_ID',
                featureidkey="properties.Field_ID",
                color='Predicted_Yield',
                color_continuous_scale="RdYlGn",
                range_color=(map_data['Predicted_Yield'].min(), map_data['Predicted_Yield'].max()),
                mapbox_style="carto-positron",
                zoom=8,
                center={"lat": 53.95, "lon": 63.48},
                opacity=0.7,
                labels={'Predicted_Yield': 'Predicted Yield'}
            )
            
            fig_map.update_layout(
                margin={"r":0,"t":30,"l":0,"b":0},
                height=600,
                title=dict(text='Predicted Yield by Field', x=0.5)
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")

        if has_yield:
            # Calculate residuals
            results_df['Residuals'] = results_df['Yield'] - results_df['Predicted_Yield']
            
            # Add residuals visualization
            st.subheader("Residuals Distribution")
            fig_residuals = px.histogram(
                results_df,
                x='Residuals',
                nbins=100,
                title='Distribution of Residuals',
                color_discrete_sequence=['#e74c3c'],
                template='simple_white'
            )
            
            mean_residual = results_df['Residuals'].mean()
            std_residual = results_df['Residuals'].std()
            mae_residual = np.abs(results_df['Residuals']).mean()
            mean_predicted_yield = results_df['Predicted_Yield'].mean()
            mean_yield = results_df['Yield'].mean()
            
            fig_residuals.add_vline(x=mean_residual, line_dash="dash", line_color="blue",
                                   annotation_text=f"Mean: {mean_residual:.2f}")
            fig_residuals.add_vline(x=mean_residual + std_residual, line_dash="dot", line_color="gray",
                                   annotation_text=f"+1 Std: {std_residual:.2f}")
            fig_residuals.add_vline(x=mean_residual - std_residual, line_dash="dot", line_color="gray",
                                   annotation_text=f"-1 Std: {std_residual:.2f}")
            
            fig_residuals.update_layout(
                xaxis_title="Residuals",
                yaxis_title="Count",
                annotations=[
                    dict(x=0.8, y=1.05, xref="paper", yref="paper",
                        text=f"MAE: {mae_residual:.2f}", showarrow=False),
                    dict(x=0.8, y=1.00, xref="paper", yref="paper",
                        text=f"Mean Predicted: {mean_predicted_yield:.2f}", showarrow=False),
                    dict(x=0.8, y=0.95, xref="paper", yref="paper",
                        text=f"Mean Actual: {mean_yield:.2f}", showarrow=False),
                    dict(x=0.8, y=0.90, xref="paper", yref="paper",
                        text=f"Std Residual: {std_residual:.2f}", showarrow=False),
                    dict(x=0.8, y=0.85, xref="paper", yref="paper",
                        text=f"Mean Residual: {mean_residual:.2f}", showarrow=False)
                ]
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Update the predictions table to include Yield and Residuals
            st.subheader("Predictions Table")
            display_df = results_df.copy()
            display_columns = ['Подразделение', 'Поле', 'Field_ID', 'Yield', 'Predicted_Yield', 'Residuals']
            format_dict = {
                'Yield': '{:.2f}',
                'Predicted_Yield': '{:.2f}',
                'Residuals': '{:.2f}'
            }
            
            st.dataframe(
                display_df[display_columns].style.format(format_dict),
                height=400,
                use_container_width=True
            )



if __name__ == '__main__':
    main()
