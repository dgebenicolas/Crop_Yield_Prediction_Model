import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.express as px
import json
import sys
import os
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from long_term_utils import (
    setup_preprocessor, check_csv_format, process_data, 
    map_agrofon_to_group, process_data_yield, rename_product_groups
)

current_dir = os.path.dirname(os.path.abspath(__file__))
def load_model():
    try:
        model = lgb.Booster(model_file = os.path.join(current_dir, 'long_term_lgbm.txt'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title('Crop Yield Prediction')
    st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <h3 style='margin-top: 0;'>Model Input Features:</h3>
    <p><strong>Required columns for prediction:</strong></p>
    <ul>
        <li><strong>Field Information:</strong> Подразделение, Поле, Field_ID</li>
        <li><strong>Historical Yield Data:</strong> Previous_Years_Yield, Previous_Year_Mean_Region</li>
        <li><strong>Fertilizer and Pesticide Usage:</strong> Macro Total/ha, Micro Total/ha, Fung Total/ha, Pest Total/ha</li>
        <li><strong>Soil Properties:</strong> bdod, cec, clay, phh2o, sand, silt, soc</li>
        <li><strong>Weather Data (May-August):</strong> 
            <ul>
                <li>Temperature: 5_temperature_2m, 6_temperature_2m, 7_temperature_2m, 8_temperature_2m</li>
                <li>Precipitation: 5_total_precipitation_sum, 8_total_precipitation_sum</li>
            </ul>
        </li>
        <li><strong>Agricultural Practice:</strong> Агрофон, Product Group</li>
        <li><strong>Other:</strong> Year</li>
    </ul>
    <p><em>Optional: 'Yield' column for comparing predictions with actual yields</em></p>
</div>

    """, unsafe_allow_html=True)
    
    # Load model and preprocessor
    model = load_model()
    if model is None:
        return
    csv_path = os.path.join(current_dir, 'long_term_test.csv')
    pre_process_df = pd.read_csv(csv_path)
    preprocessor, numeric_features, categorical_features = setup_preprocessor(pre_process_df)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Check CSV format
        is_valid, result = check_csv_format(uploaded_file)
        if not is_valid:
            st.error(result)
            return
        has_yield = 'Yield' in result.columns
        
        if has_yield:
            id_columns, process_df = process_data_yield(result)
        else:
            id_columns, process_df = process_data(result)
        
        # Map Agrofon to groups
        process_df = map_agrofon_to_group(process_df)
        process_df = rename_product_groups(process_df)
        # Preprocess data using fit_transform
        processed_data = preprocessor.transform(process_df)
        
        # Create DataFrame with processed data
        feature_names = (numeric_features + 
                        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
        if 'Культура_others' in processed_df.columns:
            processed_df = processed_df.drop(columns=['Культура_others'])
        # Make predictions
        predictions = model.predict(processed_df)
        
        # Create results DataFrame
        results_df = id_columns.copy()
        results_df['Predicted_Yield'] = predictions
        

        st.subheader("Predictions Table")
        st.dataframe(
            results_df.style.format({'Predicted_Yield': '{:.2f}'}),
            height=400,  # Adjust height as needed
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            mae = mean_absolute_error(results_df['Yield'], results_df['Predicted_Yield'])
            st.metric(
                label="Mean Absolute Error (MAE)", 
                value=f"{mae:.2f} ц/га",
                help="Average absolute difference between actual and predicted yields By Farm"
            )
        with col2:
            mean_yield = results_df['Yield'].mean()
            mae = mean_absolute_error(results_df['Yield'], results_df['Predicted_Yield'])
            percentage_error = (mae / mean_yield) * 100
            st.metric(
                label="Mean Absolute Percentage Error", 
                value=f"{percentage_error:.1f}%",
                help="MAE divided by mean yield - shows average error as percentage of mean yield"
            )
        st.subheader('Yield Prediction By Farm')
        agg_columns = ['Predicted_Yield'] + (['Yield'] if 'Yield' in results_df.columns else [])
        farm_results = results_df.groupby('Подразделение')[agg_columns].mean().reset_index()

        col1, col2 = st.columns(2)
        with col1:
            mae = mean_absolute_error(farm_results['Yield'], farm_results['Predicted_Yield'])
            st.metric(
                label="Mean Absolute Error (MAE)", 
                value=f"{mae:.2f} ц/га",
                help="Average absolute difference between actual and predicted yields By Farm"
            )
        with col2:
            mean_yield = farm_results['Yield'].mean()
            mae = mean_absolute_error(farm_results['Yield'], farm_results['Predicted_Yield'])
            percentage_error = (mae / mean_yield) * 100
            st.metric(
                label="Mean Absolute Percentage Error", 
                value=f"{percentage_error:.1f}%",
                help="MAE divided by mean yield - shows average error as percentage of mean yield by Farm"
            )
        st.subheader('Summary Statistics')
        summary_results = pd.DataFrame({'Mean': results_df[agg_columns].mean(), 
                                    'Std': results_df[agg_columns].std()}).T
        st.dataframe(summary_results)

        if has_yield:
            st.subheader("Yield Distribution Comparison")

            fig_kde = go.Figure()
            colors = ['rgba(31, 119, 180, {})', 'rgba(255, 127, 14, {})']

            for idx, (col, label) in enumerate([('Yield', 'Actual Yield'), ('Predicted_Yield', 'Predicted Yield')]):
                kde = gaussian_kde(results_df[col], bw_method='scott')
                x_grid = np.linspace(results_df[col].min(), results_df[col].max(), 1000)
                fig_kde.add_trace(go.Scatter(
                    x=x_grid, y=kde(x_grid), name=label, mode='lines',
                    line=dict(color=colors[idx].format(1)),
                    fill='tozeroy', fillcolor=colors[idx].format(0.3)
                ))

            fig_kde.update_layout(
                title='Distribution Comparison: Actual vs Predicted Yield',
                xaxis_title='Yield', yaxis_title='Density', template='simple_white'
            )

            st.plotly_chart(fig_kde, use_container_width=True)
        else:
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

            geojson_filepath = os.path.join(current_dir,'All Fields Polygons.geojson')
            
            if not os.path.exists(geojson_filepath):
                st.error(f"Missing GEOJSON")
                return
                
            with open(geojson_filepath, 'r') as f:
                geojson_data = json.load(f)
                
            selected_divisions = st.multiselect(
                "Filter by Подразделение:",
                options=sorted(results_df['Подразделение'].unique()),
                default=sorted(results_df['Подразделение'].unique())
            )

            # Filter the data
            map_data = results_df[results_df['Подразделение'].isin(selected_divisions)].copy()
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
            # Assuming results_df is already defined and contains the necessary data
            results_df['Residuals'] = results_df['Yield'] - results_df['Predicted_Yield']

            # Add residuals visualization
            st.subheader("Residuals Distribution")

            # Calculate statistics
            mean_residual = results_df['Residuals'].mean()
            std_residual = results_df['Residuals'].std()
            mae_residual = np.abs(results_df['Residuals']).mean()
            mean_predicted_yield = results_df['Predicted_Yield'].mean()
            mean_yield = results_df['Yield'].mean()

            # Create histogram trace
            hist_trace = go.Histogram(
                x=results_df['Residuals'],
                nbinsx=100,
                name='Residuals',
                marker_color='#e74c3c'
            )

            # Create layout
            layout = go.Layout(
                title='Distribution of Residuals',
                xaxis_title="Residuals",
                yaxis_title="Count",
                template='simple_white',
                showlegend=False,
                annotations=[
                    dict(x=0.95, y=0.95, xref="paper", yref="paper",
                        text=f"Mean Residual: {mean_residual:.2f}", showarrow=False, align='left'),
                    dict(x=0.95, y=0.90, xref="paper", yref="paper",
                        text=f"Std Dev: {std_residual:.2f}", showarrow=False, align='left'),
                    dict(x=0.95, y=0.85, xref="paper", yref="paper",
                        text=f"MAE: {mae_residual:.2f}", showarrow=False, align='left'),
                    dict(x=0.95, y=0.80, xref="paper", yref="paper",
                        text=f"Mean Predicted: {mean_predicted_yield:.2f}", showarrow=False, align='left'),
                    dict(x=0.95, y=0.75, xref="paper", yref="paper",
                        text=f"Mean Actual: {mean_yield:.2f}", showarrow=False, align='left'),
                ]
            )

            # Create figure
            fig_residuals = go.Figure(data=[hist_trace], layout=layout)

            # Add vertical lines
            fig_residuals.add_vline(x=mean_residual, line_dash="dash", line_color="blue",
                                    annotation_text="Mean", annotation_position="top left")
            fig_residuals.add_vline(x=mean_residual + std_residual, line_dash="dot", line_color="gray",
                                    annotation_text="+1 Std", annotation_position="top left")
            fig_residuals.add_vline(x=mean_residual - std_residual, line_dash="dot", line_color="gray",
                                    annotation_text="-1 Std", annotation_position="top left")

            # Display the plot
            st.plotly_chart(fig_residuals, use_container_width=True)



if __name__ == '__main__':
    main()