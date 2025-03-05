import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Performs exploratory data analysis (EDA) on the dataset,
    outputs key statistics, and saves plots for the report.
    
    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("Performing Exploratory Data Analysis (EDA)...")
    
    # Display dataset overview
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    print("\nDataset info:")
    df.info()
    
    # Plot and save the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.show()
    
    # Plot and save the distribution of the target variable 'price_range'
    if "price_range" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x="price_range", palette="viridis")
        plt.title("Distribution of Price Range")
        plt.tight_layout()
        plt.savefig("price_range_distribution.png")
        plt.show()
    
    print("EDA completed and plots saved.")
