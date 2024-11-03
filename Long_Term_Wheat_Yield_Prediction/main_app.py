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
    map_agrofon_to_group, process_data_yield, rename_product_groups, predict_yields, process_data_other, map_crop_name
)

current_dir = os.path.dirname(os.path.abspath(__file__))
def load_model(model_type):
    """
    Load model based on user selection
    Args:
        model_type (str): Either 'wheat' or 'other_crops'
    Returns:
        object: Loaded model or None if error
    """
    model_paths = {
        'wheat': 'long_term_lgbm.txt',
        'other_crops': 'other_crops_lgbm.txt'
    }
    
    try:
        model_path = os.path.join(current_dir, model_paths[model_type])
        model = lgb.Booster(model_file=model_path)
        return model
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        return None

def get_prep_path(model_type):
    """
    Get preprocessing data path based on model type
    """
    prep_paths = {
        'wheat': 'long_term_test.csv',
        'other_crops': 'other_crop_set_up.csv'
    }
    return prep_paths[model_type]

def main():
    st.title('Crop Yield Prediction')
    st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
    <h3 style="margin-top: 0;">Model Input Features:</h3>
    <p><strong>Required columns for prediction:</strong></p>
    <ul>
        <li><strong>General Information:</strong> Year, Агрофон, Культура</li>
        <li><strong>Fertilizer and Pesticide Application:</strong>
            Macro Total/ha, Fung Total/ha, Pest Total/ha
        </li>
        <li><strong>Soil Properties:</strong>
            bdod (bulk density), phh2o (soil pH), sand, silt, soc (soil organic carbon)
        </li>
        <li><strong>Weather Data (May-August):</strong>
            <ul>
                <li>Relative Humidity</li>
                <li>Solar Radiation: </li>
                <li>Max Temperature: </li>
                <li>Min Temperature:</li>
                <li>Total Precipitation: </li>
                <li>Wind Component (V): </li>
                <li>Vapor Pressure Deficit: </li>
            </ul>
        </li>
    </ul>
    <p><em>Note: Optional 'Yield' column can be used to compare predicted vs. actual yields.</em></p>
</div>


    """, unsafe_allow_html=True)
    # Add model selection dropdown
    model_type = st.selectbox(
        "Select crop type for prediction",
        options=['wheat', 'other_crops'],
        index=0,
        help="Choose 'wheat' for wheat predictions or 'other_crops' for other crop types"
    )
    
    model = load_model(model_type)
    if model is None:
        return None, "Failed to load model"
    
    # Get prep path based on model type
    prep_path = get_prep_path(model_type)
   
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Check CSV format
        is_valid, result = check_csv_format(uploaded_file)
        if not is_valid:
            st.error(result)
            return
        has_yield = 'Yield' in result.columns

        if model_type == 'wheat':
            if has_yield:
                id_columns, process_df = process_data_yield(result)
                result_df, error = predict_yields(id_columns, process_df, current_dir,  model, prep_path, model_type)
            else:
                id_columns, process_df = process_data(result)
                result_df, error = predict_yields(id_columns, process_df, current_dir,  model, prep_path, model_type)
        elif model_type == 'other_crops':
            id_columns, process_df = process_data_other(result)
            result_df, error = predict_yields(id_columns, process_df, current_dir,  model, prep_path, model_type)

        def create_visualizations(results_df, model_type):
            """
            Create visualizations based on the model type and results dataframe structure
            
            Args:
                results_df (pd.DataFrame): Results dataframe
                model_type (str): Type of model ('wheat' or 'other_crops')
            """
            # Determine dataframe structure
            has_yield = 'Yield' in results_df.columns
            is_other_crops = model_type == 'other_crops'
            if is_other_crops:
                results_df = map_crop_name(results_df)
            
            # Configure visualization parameters based on model type
            viz_config = {
                'yield_col': 'Yield' if has_yield else None,
                'pred_col': 'Predicted_Yield',
                'group_col': 'Подразделение',
                'field_id_col': 'Field_ID',
                'unit': 'ц/га' if not is_other_crops else '%',
                'color_scale': "RdYlGn" if not is_other_crops else "Viridis",
                'map_center': {"lat": 53.95, "lon": 63.48},
                'crop_col': 'Культура' if is_other_crops else None
            }

            # 1. Predictions Table
            st.subheader("Predictions Table")
            st.dataframe(
                results_df.style.format({viz_config['pred_col']: '{:.2f}'}),
                height=400,
                use_container_width=True
            )

            # 2. Error Metrics (only if actual yields are available)
            if has_yield:
                display_error_metrics(results_df, viz_config)

            # 3. Group-level Analysis
            display_group_analysis(results_df, viz_config)

            # 4. Summary Statistics
            display_summary_statistics(results_df, viz_config)

            # 5. Distribution Plots
            display_distribution_plots(results_df, viz_config)

            # 6. Choropleth Map
            display_choropleth_map(results_df, viz_config)

            # 7. Residuals Analysis (only if actual yields are available)
            if has_yield:
                display_residuals_analysis(results_df, viz_config)


        def display_error_metrics(results_df, config):
            """Display error metrics in two columns"""
            col1, col2 = st.columns(2)
            with col1:
                mae = mean_absolute_error(results_df[config['yield_col']], 
                                        results_df[config['pred_col']])
                st.metric(
                    label="Mean Absolute Error (MAE)", 
                    value=f"{mae:.2f} {config['unit']}",
                    help="Average absolute difference between actual and predicted values"
                )
            with col2:
                mean_actual = results_df[config['yield_col']].mean()
                percentage_error = (mae / mean_actual) * 100
                st.metric(
                    label="Mean Absolute Percentage Error", 
                    value=f"{percentage_error:.1f}%",
                    help="MAE as percentage of mean actual value"
                )

        def display_group_analysis(results_df, config):
            """Display group-level analysis with crop grouping for other_crops"""
            groupby_cols = [config['group_col']]
            if config['crop_col']:
                groupby_cols.append(config['crop_col'])
                st.subheader(f'Predictions By Farm and Crop')
            else:
                st.subheader(f'Predictions By Farm')
            
            agg_columns = [config['pred_col']]
            if config['yield_col']:
                agg_columns.append(config['yield_col'])
            
            group_results = results_df.groupby(groupby_cols)[agg_columns].mean().reset_index()
            st.dataframe(group_results)
            
            if config['yield_col']:
                display_error_metrics(group_results, config)

        def display_summary_statistics(results_df, config):
            """Display summary statistics with optional grouping by crop type"""
            st.subheader('Summary Statistics')
            cols_to_summarize = [config['pred_col']]
            if config['yield_col']:
                cols_to_summarize.append(config['yield_col'])
            
            # Check if it's other_crops model type and crop_col exists
            if config.get('crop_col'):
                # Group by crop type and calculate statistics
                summary_results = results_df.groupby(config['crop_col'])[cols_to_summarize].agg([
                    ('Mean', 'mean'),
                    ('Std', 'std'),
                    ('Min', 'min'),
                    ('Max', 'max')
                ]).round(2)
                
                # Flatten column multi-index for better display
                summary_results.columns = [f'{col[0]} {col[1]}' for col in summary_results.columns]
            else:
                # Original summary statistics without grouping
                summary_results = pd.DataFrame({
                    'Mean': results_df[cols_to_summarize].mean(),
                    'Std': results_df[cols_to_summarize].std(),
                    'Min': results_df[cols_to_summarize].min(),
                    'Max': results_df[cols_to_summarize].max()
                }).T
            
            st.dataframe(summary_results)

        def display_distribution_plots(results_df, config):
            """Display distribution plots with crop-specific handling"""
            if config['yield_col']:
                st.subheader("Value Distribution Comparison")
                
                if config['crop_col']:
                    # Create distribution plot for each crop
                    for crop in results_df[config['crop_col']].unique():
                        crop_data = results_df[results_df[config['crop_col']] == crop]
                        
                        fig_kde = go.Figure()
                        colors = ['rgba(31, 119, 180, {})', 'rgba(255, 127, 14, {})']
                        
                        for idx, (col, label) in enumerate([
                            (config['yield_col'], 'Actual Value'),
                            (config['pred_col'], 'Predicted Value')
                        ]):
                            kde = gaussian_kde(crop_data[col], bw_method='scott')
                            x_grid = np.linspace(crop_data[col].min(), crop_data[col].max(), 1000)
                            fig_kde.add_trace(go.Scatter(
                                x=x_grid, y=kde(x_grid), name=label, mode='lines',
                                line=dict(color=colors[idx].format(1)),
                                fill='tozeroy', fillcolor=colors[idx].format(0.3)
                            ))
                        
                        fig_kde.update_layout(
                            title=f'Distribution Comparison for {crop} ({config["unit"]})',
                            xaxis_title=f'Value ({config["unit"]})',
                            yaxis_title='Density',
                            template='simple_white'
                        )
                        st.plotly_chart(fig_kde, use_container_width=True)
                else:
                    # Original distribution plot code for non-crop cases
                    fig_kde = go.Figure()
                    colors = ['rgba(31, 119, 180, {})', 'rgba(255, 127, 14, {})']
                    
                    for idx, (col, label) in enumerate([
                        (config['yield_col'], 'Actual Value'),
                        (config['pred_col'], 'Predicted Value')
                    ]):
                        kde = gaussian_kde(results_df[col], bw_method='scott')
                        x_grid = np.linspace(results_df[col].min(), results_df[col].max(), 1000)
                        fig_kde.add_trace(go.Scatter(
                            x=x_grid, y=kde(x_grid), name=label, mode='lines',
                            line=dict(color=colors[idx].format(1)),
                            fill='tozeroy', fillcolor=colors[idx].format(0.3)
                        ))
                    
                    fig_kde.update_layout(
                        title=f'Distribution Comparison ({config["unit"]})',
                        xaxis_title=f'Value ({config["unit"]})',
                        yaxis_title='Density',
                        template='simple_white'
                    )
                    st.plotly_chart(fig_kde, use_container_width=True)
            else:
                st.subheader("Prediction Distribution")
                if config['crop_col']:
                    fig = px.box(results_df, x=config['crop_col'], y=config['pred_col'],
                                title=f'Distribution of Predicted Values by Crop ({config["unit"]})',
                                color=config['crop_col'])
                else:
                    fig = px.histogram(results_df, x=config['pred_col'],
                                    title=f'Distribution of Predicted Values ({config["unit"]})',
                                    color_discrete_sequence=['#3498db'])
                    

                
                st.plotly_chart(fig, use_container_width=True)

        def display_choropleth_map(results_df, config):
            """Display choropleth map with additional crop filtering for other_crops"""
            st.subheader("Predicted Value Map")
            try:
                geojson_filepath = os.path.join(current_dir, 'All Fields Polygons.geojson')
                if not os.path.exists(geojson_filepath):
                    st.error("Missing GEOJSON file")
                    return

                with open(geojson_filepath, 'r') as f:
                    geojson_data = json.load(f)

                # Filters
                filters = {
                    config['group_col']: st.multiselect(
                        f"Filter by {config['group_col']}:",
                        options=sorted(results_df[config['group_col']].unique()),
                        default=sorted(results_df[config['group_col']].unique())
                    )
                }
                
                if config['crop_col']:
                    filters[config['crop_col']] = st.multiselect(
                        "Filter by Crop:",
                        options=sorted(results_df[config['crop_col']].unique()),
                        default=sorted(results_df[config['crop_col']].unique())[0:1]  # Default to first crop
                    )

                # Apply filters
                map_data = results_df.copy()
                for col, selected_values in filters.items():
                    map_data = map_data[map_data[col].isin(selected_values)]

                # Prepare hover data
                hover_data = {config['pred_col']: ':.2f'}
                if config['crop_col']:
                    hover_data[config['crop_col']] = True

                fig_map = px.choropleth_mapbox(
                    map_data,
                    geojson=geojson_data,
                    locations=config['field_id_col'],
                    featureidkey="properties.Field_ID",
                    color=config['pred_col'],
                    color_continuous_scale=config['color_scale'],
                    range_color=(map_data[config['pred_col']].min(), 
                                map_data[config['pred_col']].max()),
                    mapbox_style="carto-positron",
                    zoom=8,
                    center=config['map_center'],
                    opacity=0.7,
                    hover_data=hover_data,
                    labels={config['pred_col']: f'Predicted Value ({config["unit"]})'}
                )

                fig_map.update_layout(
                    margin={"r":0,"t":30,"l":0,"b":0},
                    height=600,
                    title=dict(text=f'Predicted Values by Field ({config["unit"]})', x=0.5)
                )
                st.plotly_chart(fig_map, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating map: {str(e)}")

        def display_residuals_analysis(results_df, config):
            """Display residuals analysis"""
            results_df['Residuals'] = results_df[config['yield_col']] - results_df[config['pred_col']]
            
            st.subheader("Residuals Distribution")
            
            stats = {
                'Mean Residual': results_df['Residuals'].mean(),
                'Std Dev': results_df['Residuals'].std(),
                'MAE': np.abs(results_df['Residuals']).mean(),
                'Mean Predicted': results_df[config['pred_col']].mean(),
                'Mean Actual': results_df[config['yield_col']].mean()
            }
            
            fig_residuals = go.Figure(data=[
                go.Histogram(
                    x=results_df['Residuals'],
                    nbinsx=100,
                    name='Residuals',
                    marker_color='#e74c3c'
                )
            ])
            
            fig_residuals.update_layout(
                title='Distribution of Residuals',
                xaxis_title=f"Residuals ({config['unit']})",
                yaxis_title="Count",
                template='simple_white',
                showlegend=False,
                annotations=[
                    dict(x=0.95, y=0.95-0.05*i, xref="paper", yref="paper",
                        text=f"{k}: {v:.2f}", showarrow=False, align='left')
                    for i, (k, v) in enumerate(stats.items())
                ]
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
        create_visualizations(result_df, model_type)


        # csv_path = os.path.join(current_dir, 'long_term_test.csv')
        # pre_process_df = pd.read_csv(csv_path)
        # preprocessor, numeric_features, categorical_features = setup_preprocessor(pre_process_df)
        # feature_names = (numeric_features + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
        
        # # Map Agrofon to groups
        # process_df = map_agrofon_to_group(process_df)
        # process_df = rename_product_groups(process_df)
        # # Preprocess data using fit_transform
        # processed_data = preprocessor.transform(process_df)
        
        # # Create DataFrame with processed data
        # processed_df = pd.DataFrame(processed_data, columns=feature_names)
        # if 'Культура_others' in processed_df.columns:
        #     processed_df = processed_df.drop(columns=['Культура_others'])
        # # Make predictions
        # predictions = model.predict(processed_df)
        
        # # Create results DataFrame
        # results_df = id_columns.copy()
        # results_df['Predicted_Yield'] = predictions
        

        # st.subheader("Predictions Table")
        # st.dataframe(
        #     results_df.style.format({'Predicted_Yield': '{:.2f}'}),
        #     height=400,  # Adjust height as needed
        #     use_container_width=True
        # )

        # col1, col2 = st.columns(2)
        # with col1:
        #     mae = mean_absolute_error(results_df['Yield'], results_df['Predicted_Yield'])
        #     st.metric(
        #         label="Mean Absolute Error (MAE)", 
        #         value=f"{mae:.2f} ц/га",
        #         help="Average absolute difference between actual and predicted yields By Farm"
        #     )
        # with col2:
        #     mean_yield = results_df['Yield'].mean()
        #     mae = mean_absolute_error(results_df['Yield'], results_df['Predicted_Yield'])
        #     percentage_error = (mae / mean_yield) * 100
        #     st.metric(
        #         label="Mean Absolute Percentage Error", 
        #         value=f"{percentage_error:.1f}%",
        #         help="MAE divided by mean yield - shows average error as percentage of mean yield"
        #     )
        # st.subheader('Yield Prediction By Farm')
        # agg_columns = ['Predicted_Yield'] + (['Yield'] if 'Yield' in results_df.columns else [])
        # farm_results = results_df.groupby('Подразделение')[agg_columns].mean().reset_index()
        # st.dataframe(farm_results)

        # col1, col2 = st.columns(2)
        # with col1:
        #     mae = mean_absolute_error(farm_results['Yield'], farm_results['Predicted_Yield'])
        #     st.metric(
        #         label="Mean Absolute Error (MAE)", 
        #         value=f"{mae:.2f} ц/га",
        #         help="Average absolute difference between actual and predicted yields By Farm"
        #     )
        # with col2:
        #     mean_yield = farm_results['Yield'].mean()
        #     mae = mean_absolute_error(farm_results['Yield'], farm_results['Predicted_Yield'])
        #     percentage_error = (mae / mean_yield) * 100
        #     st.metric(
        #         label="Mean Absolute Percentage Error", 
        #         value=f"{percentage_error:.1f}%",
        #         help="MAE divided by mean yield - shows average error as percentage of mean yield by Farm"
        #     )
        # st.subheader('Summary Statistics')
        # summary_results = pd.DataFrame({'Mean': results_df[agg_columns].mean(), 
        #                             'Std': results_df[agg_columns].std()}).T
        # st.dataframe(summary_results)

        # if has_yield:
        #     st.subheader("Yield Distribution Comparison")

        #     fig_kde = go.Figure()
        #     colors = ['rgba(31, 119, 180, {})', 'rgba(255, 127, 14, {})']

        #     for idx, (col, label) in enumerate([('Yield', 'Actual Yield'), ('Predicted_Yield', 'Predicted Yield')]):
        #         kde = gaussian_kde(results_df[col], bw_method='scott')
        #         x_grid = np.linspace(results_df[col].min(), results_df[col].max(), 1000)
        #         fig_kde.add_trace(go.Scatter(
        #             x=x_grid, y=kde(x_grid), name=label, mode='lines',
        #             line=dict(color=colors[idx].format(1)),
        #             fill='tozeroy', fillcolor=colors[idx].format(0.3)
        #         ))

        #     fig_kde.update_layout(
        #         title='Distribution Comparison: Actual vs Predicted Yield',
        #         xaxis_title='Yield', yaxis_title='Density', template='simple_white'
        #     )

        #     st.plotly_chart(fig_kde, use_container_width=True)
        # else:
        #     st.subheader("Prediction Distribution")
        #     fig = px.histogram(
        #         results_df, 
        #         x='Predicted_Yield',
        #         nbins=100,
        #         title='Distribution of Predicted Yields',
        #         color_discrete_sequence=['#3498db'],
        #         template='simple_white'
        #     )
            
        #     fig.update_layout(
        #         xaxis_title="Predicted Yield",
        #         yaxis_title="Count",
        #         showlegend=False,
        #         xaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
        #         yaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
        #         title=dict(font=dict(size=16))
        #     )
            
        #     mean_yield = results_df['Predicted_Yield'].mean()
        #     fig.add_vline(
        #         x=mean_yield, 
        #         line_dash="dash", 
        #         line_color="red",
        #         annotation_text=f"Mean: {mean_yield:.2f}",
        #         annotation_position="top"
        #     )
            
        #     st.plotly_chart(fig, use_container_width=True)
        
        # # Choropleth Map
        # st.subheader("Predicted Yield Map")        
        # try:

        #     geojson_filepath = os.path.join(current_dir,'All Fields Polygons.geojson')
            
        #     if not os.path.exists(geojson_filepath):
        #         st.error(f"Missing GEOJSON")
        #         return
                
        #     with open(geojson_filepath, 'r') as f:
        #         geojson_data = json.load(f)
                
        #     selected_divisions = st.multiselect(
        #         "Filter by Подразделение:",
        #         options=sorted(results_df['Подразделение'].unique()),
        #         default=sorted(results_df['Подразделение'].unique())
        #     )

        #     # Filter the data
        #     map_data = results_df[results_df['Подразделение'].isin(selected_divisions)].copy()
        #     map_data = map_data[['Подразделение', 'Field_ID', 'Predicted_Yield']]
            
        #     fig_map = px.choropleth_mapbox(
        #         map_data, 
        #         geojson=geojson_data, 
        #         locations='Field_ID',
        #         featureidkey="properties.Field_ID",
        #         color='Predicted_Yield',
        #         color_continuous_scale="RdYlGn",
        #         range_color=(map_data['Predicted_Yield'].min(), map_data['Predicted_Yield'].max()),
        #         mapbox_style="carto-positron",
        #         zoom=8,
        #         center={"lat": 53.95, "lon": 63.48},
        #         opacity=0.7,
        #         labels={'Predicted_Yield': 'Predicted Yield'}
        #     )
            
        #     fig_map.update_layout(
        #         margin={"r":0,"t":30,"l":0,"b":0},
        #         height=600,
        #         title=dict(text='Predicted Yield by Field', x=0.5)
        #     )
            
        #     st.plotly_chart(fig_map, use_container_width=True)
            
        # except Exception as e:
        #     st.error(f"Error creating map: {str(e)}")

        # if has_yield:
        #     # Calculate residuals
        #     # Assuming results_df is already defined and contains the necessary data
        #     results_df['Residuals'] = results_df['Yield'] - results_df['Predicted_Yield']

        #     # Add residuals visualization
        #     st.subheader("Residuals Distribution")

        #     # Calculate statistics
        #     mean_residual = results_df['Residuals'].mean()
        #     std_residual = results_df['Residuals'].std()
        #     mae_residual = np.abs(results_df['Residuals']).mean()
        #     mean_predicted_yield = results_df['Predicted_Yield'].mean()
        #     mean_yield = results_df['Yield'].mean()

        #     # Create histogram trace
        #     hist_trace = go.Histogram(
        #         x=results_df['Residuals'],
        #         nbinsx=100,
        #         name='Residuals',
        #         marker_color='#e74c3c'
        #     )

        #     # Create layout
        #     layout = go.Layout(
        #         title='Distribution of Residuals',
        #         xaxis_title="Residuals",
        #         yaxis_title="Count",
        #         template='simple_white',
        #         showlegend=False,
        #         annotations=[
        #             dict(x=0.95, y=0.95, xref="paper", yref="paper",
        #                 text=f"Mean Residual: {mean_residual:.2f}", showarrow=False, align='left'),
        #             dict(x=0.95, y=0.90, xref="paper", yref="paper",
        #                 text=f"Std Dev: {std_residual:.2f}", showarrow=False, align='left'),
        #             dict(x=0.95, y=0.85, xref="paper", yref="paper",
        #                 text=f"MAE: {mae_residual:.2f}", showarrow=False, align='left'),
        #             dict(x=0.95, y=0.80, xref="paper", yref="paper",
        #                 text=f"Mean Predicted: {mean_predicted_yield:.2f}", showarrow=False, align='left'),
        #             dict(x=0.95, y=0.75, xref="paper", yref="paper",
        #                 text=f"Mean Actual: {mean_yield:.2f}", showarrow=False, align='left'),
        #         ]
        #     )

        #     # Create figure
        #     fig_residuals = go.Figure(data=[hist_trace], layout=layout)

        #     # Add vertical lines
        #     fig_residuals.add_vline(x=mean_residual, line_dash="dash", line_color="blue",
        #                             annotation_text="Mean", annotation_position="top left")
        #     fig_residuals.add_vline(x=mean_residual + std_residual, line_dash="dot", line_color="gray",
        #                             annotation_text="+1 Std", annotation_position="top left")
        #     fig_residuals.add_vline(x=mean_residual - std_residual, line_dash="dot", line_color="gray",
        #                             annotation_text="-1 Std", annotation_position="top left")

        #     # Display the plot
        #     st.plotly_chart(fig_residuals, use_container_width=True)



if __name__ == '__main__':
    main()