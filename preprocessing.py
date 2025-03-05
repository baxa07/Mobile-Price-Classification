import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath: str):
    """
    Loads the dataset, splits into features and target, 
    and returns train/test splits with standardized features.
    
    Parameters:
        filepath (str): Path to the CSV data file.
    
    Returns:
        df (pd.DataFrame): The loaded dataset.
        X_train (pd.DataFrame): Training features (original scale).
        X_test (pd.DataFrame): Testing features (original scale).
        X_train_scaled (np.ndarray): Standardized training features.
        X_test_scaled (np.ndarray): Standardized testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
    """
    print("Loading data from:", filepath)
    df = pd.read_csv(filepath)
    
    # Separate features and target variable (assuming "price_range" as target)
    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    
    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data loaded and preprocessed successfully.")
    return df, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
