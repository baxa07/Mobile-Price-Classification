from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(X_train_combined, X_test_combined, y_train, y_test, clustering_method="GMM"):
    """
    Trains a Random Forest classifier with an expanded hyperparameter grid to reduce overfitting,
    calculates both training and test accuracy, and outputs evaluation plots.
    
    Parameters:
        X_train_combined (np.ndarray): Combined training features.
        X_test_combined (np.ndarray): Combined testing features.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Testing target values.
        clustering_method (str): Clustering method used ('GMM' or 'DBSCAN').
    """
    print("Starting hyperparameter tuning using GridSearchCV with extended parameter grid...")
    
    # Expanded hyperparameter grid to enforce regularization and limit overfitting.
    param_grid = {
        'max_depth': [5, 7, 10, 15],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 3, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    base_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    grid_search = GridSearchCV(base_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_combined, y_train)
    
    print("Best parameters found:", grid_search.best_params_)
    best_clf = grid_search.best_estimator_
    
    # Evaluate on training data
    y_pred_train = best_clf.predict(X_train_combined)
    train_acc = accuracy_score(y_train, y_pred_train)
    
    # Evaluate on test data
    y_pred_test = best_clf.predict(X_test_combined)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTraining Accuracy ({clustering_method}): {train_acc * 100:.2f}%")
    print(f"Test Accuracy ({clustering_method}): {test_acc * 100:.2f}%")
    
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_pred_test))
    
    # Plot and save the confusion matrix for test data
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
