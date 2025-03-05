from preprocessing import load_and_preprocess_data
from EDA import perform_eda
from feature_engineering import feature_engineering
from model_training import train_and_evaluate

def main():
    # Path to your dataset (ensure "train.csv" is in your working directory)
    filepath = "train.csv"
    
    # Step 1: Preprocessing
    df, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data(filepath)
    
    # Step 2: Exploratory Data Analysis (EDA)
    perform_eda(df)
    
    # Step 3: Feature Engineering
    feature_names = X_train.columns.tolist()
    X_train_combined, X_test_combined = feature_engineering(
        X_train_scaled, X_test_scaled, feature_names, y_train, clustering_method="GMM"
    )
    
    # Step 4: Model Training and Evaluation
    train_and_evaluate(X_train_combined, X_test_combined, y_train, y_test, clustering_method="GMM")

if __name__ == "__main__":
    main()
