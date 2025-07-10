import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def extract_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def transform_data(df):
    """Preprocess data by handling missing values, encoding categorical features, and scaling numerical data."""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Pipelines for transformations
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    transformed_data = preprocessor.fit_transform(df)
    return transformed_data, preprocessor

def load_data(transformed_data, output_file):
    """Save the processed data to a CSV file."""
    pd.DataFrame(transformed_data).to_csv(output_file, index=False)

def main():
    input_file = r'C:\Users\rishi\OneDrive\Desktop\code tech\sampledata'
    output_file = 'processed_data.csv'
    
    df = extract_data(input_file)
    transformed_data, _ = transform_data(df)
    load_data(transformed_data, output_file)
    print(f"Data processing complete. Output saved to {output_file}")

if __name__ == "_main_":
    main()