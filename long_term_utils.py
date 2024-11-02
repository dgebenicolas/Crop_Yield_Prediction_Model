import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



REQUIRED_COLUMNS = ['Подразделение', 'Поле', 'Field_ID', 'Year', 'Агрофон',
       'Культура', 'Macro Total/ha', 'Fung Total/ha', 'Pest Total/ha', 'bdod',
       'phh2o', 'sand', 'silt', 'soc', 'DOY_min', '5_relative_humidity',
       '6_relative_humidity', '7_relative_humidity', '8_relative_humidity',
       '5_surface_solar_radiation_downwards_sum',
       '6_surface_solar_radiation_downwards_sum',
       '7_surface_solar_radiation_downwards_sum',
       '8_surface_solar_radiation_downwards_sum', '5_temperature_2m_max',
       '6_temperature_2m_max', '7_temperature_2m_max', '8_temperature_2m_max',
       '5_temperature_2m_min', '6_temperature_2m_min', '7_temperature_2m_min',
       '8_temperature_2m_min', '5_total_precipitation_sum',
       '6_total_precipitation_sum', '7_total_precipitation_sum',
       '8_total_precipitation_sum', '5_v_component_of_wind_10m',
       '6_v_component_of_wind_10m', '7_v_component_of_wind_10m',
       '8_v_component_of_wind_10m', '5_vapor_pressure_deficit',
       '6_vapor_pressure_deficit', '7_vapor_pressure_deficit',
       '8_vapor_pressure_deficit'
]

COLUMN_DTYPES = {
    'Year': 'int64', 'Агрофон': 'object', 'Культура': 'object',
    'Macro Total/ha': 'float64', 'Fung Total/ha': 'float64', 'Pest Total/ha': 'float64',
    'bdod': 'float64', 'phh2o': 'float64', 'sand': 'float64', 'silt': 'float64', 'soc': 'float64',
    'DOY_min': 'int64', '5_relative_humidity': 'float64', '6_relative_humidity': 'float64',
    '7_relative_humidity': 'float64', '8_relative_humidity': 'float64',
    '5_surface_solar_radiation_downwards_sum': 'float64', '6_surface_solar_radiation_downwards_sum': 'float64',
    '7_surface_solar_radiation_downwards_sum': 'float64', '8_surface_solar_radiation_downwards_sum': 'float64',
    '5_temperature_2m_max': 'float64', '6_temperature_2m_max': 'float64', '7_temperature_2m_max': 'float64',
    '8_temperature_2m_max': 'float64', '5_temperature_2m_min': 'float64', '6_temperature_2m_min': 'float64',
    '7_temperature_2m_min': 'float64', '8_temperature_2m_min': 'float64',
    '5_total_precipitation_sum': 'float64', '6_total_precipitation_sum': 'float64',
    '7_total_precipitation_sum': 'float64', '8_total_precipitation_sum': 'float64',
    '5_v_component_of_wind_10m': 'float64', '6_v_component_of_wind_10m': 'float64',
    '7_v_component_of_wind_10m': 'float64', '8_v_component_of_wind_10m': 'float64',
    '5_vapor_pressure_deficit': 'float64', '6_vapor_pressure_deficit': 'float64',
    '7_vapor_pressure_deficit': 'float64', '8_vapor_pressure_deficit': 'float64'
}



def setup_preprocessor(pre_process_df):
    numeric_features = list(pre_process_df.drop(['Агрофон', 'Культура'], axis=1).select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = ['Агрофон', 'Культура']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    preprocessor.fit(pre_process_df)
    return preprocessor, numeric_features, categorical_features

def check_csv_format(file):
    if not file.name.endswith('.csv'):
        return False, "File must be a CSV"
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}"
    
    return True, df

def process_data(df):
    # Store IDs before processing
    id_columns = df[['Подразделение', 'Поле', 'Field_ID']].copy()
    
    # Drop ID columns and reorder remaining columns
    process_cols = [col for col in REQUIRED_COLUMNS if col not in ['Подразделение', 'Поле', 'Field_ID']]
    process_df = df[process_cols].copy()
    
    # Enforce data types
    for col, dtype in COLUMN_DTYPES.items():
        if col in process_df.columns:
            process_df[col] = process_df[col].astype(dtype)
    
    return id_columns, process_df

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def process_data_yield(df):
    # Store IDs and Yield before processing
    df_cleaned = df.copy()
    #df_cleaned = remove_outliers_iqr(df_cleaned, 'Yield')
    id_columns = df_cleaned[['Подразделение', 'Поле', 'Field_ID', 'Yield']].copy()
    
    # Drop ID columns and Yield, and reorder remaining columns
    process_cols = [col for col in REQUIRED_COLUMNS if col not in ['Подразделение', 'Поле', 'Field_ID']]
    process_df = df_cleaned[process_cols].copy()
    
    # Enforce data types
    for col, dtype in COLUMN_DTYPES.items():
        if col in process_df.columns:
            process_df[col] = process_df[col].astype(dtype)
    
    return id_columns, process_df

def map_agrofon_to_group(df):
    """
    Maps the 'Агрофон' column in the given DataFrame to predefined product groups.

    The function categorizes various agrofon types into broader groups such as 
    'Stubble', 'Fallow', and 'Deep Tillage' based on the presence of specific 
    keywords in the agrofon names. If an agrofon does not match any of the 
    predefined categories, it is labeled as 'others'.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing an 'Агрофон' column.

    Returns:
    pd.DataFrame: The modified DataFrame with the 'Агрофон' column updated to 
    reflect the mapped product groups.
    """
    mapped_df = df.copy()
    def map_product_name(product_name):
        product_name = product_name.lower()  # Convert to lower case

        if "стерня" in product_name:
            return "Stubble"
        elif "пар" in product_name:
            return "Fallow"
        elif "глубокая" in product_name or "глубокое" in product_name:
            return "Deep Tillage"
        return 'others' 
    
    mapped_df['Агрофон'] = mapped_df['Агрофон'].apply(map_product_name)
    
    return mapped_df

def rename_product_groups(df):
    """
    Maps product names in the given DataFrame to standardized product groups.
    
    The function categorizes various wheat (пшеница) varieties into their respective
    groups based on the presence of specific variety names in the product names.
    The matching is case-insensitive and handles variations in product name formats.
    If a product doesn't match any predefined variety, it is labeled as 'others'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'Product Name' column.
    
    Returns:
    pd.DataFrame: The modified DataFrame with an additional 'Product Group' column 
                 containing the mapped product groups.
    """
    mapped_df = df.copy()
    
    def map_product_name(product_name):
        # Convert to lower case for case-insensitive matching
        product_name = str(product_name).lower()
        
        # Dictionary of variety keywords and their standardized group names
        variety_groups = {
            'астана': 'Астана',
            'шортандинская': 'Шортандинская',
            'ликамеро': 'Ликамеро',
            'тобольская': 'Тобольская',
            'тризо': 'Тризо',
            'радуга': 'Радуга',
            'гранни': 'Гранни',
            'урало-сибирская': 'Урало-Сибирская',
            'уралосибирская': 'Урало-Сибирская',  # Handle variation in spelling
            'айна': 'Айна'
        }
        
        # Check for each variety keyword in the product name
        for keyword, group in variety_groups.items():
            if keyword in product_name:
                return group
                
        return 'others'
    
    mapped_df['Культура'] = mapped_df['Культура'].apply(map_product_name)
    
    return mapped_df