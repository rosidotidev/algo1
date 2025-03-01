import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import data.data_enricher as de
from stock.ticker import forward_looking_label
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def norm_data(df):
    #dfx=df = pd.read_csv("")
    print(df["Target"].value_counts())
    min_value = df["Target"].value_counts().min()
    print(f"min value {min_value}")
    sample_df=df.groupby('Target').sample(min_value)
    return min_value,sample_df


def train_rf(df, features=None, target="Target", test_size=0.2,
             n_estimators=10000, random_state=42, model_filename="rf_model.pkl", scaler_filename="scaler.pkl"):
    """
    Trains a RandomForest model using the given dataset, normalizes the data,
    and saves the trained model and scaler.

    Parameters:
    - dataset_path (str): Path to the dataset CSV file.
    - features (list): List of feature column names to use for training.
    - target (str): Target column name.
    - test_size (float): Proportion of data to use for testing (default 0.2).
    - n_estimators (int): Number of trees in the RandomForestClassifier (default 100).
    - random_state (int): Random state for reproducibility.
    - model_filename (str): Name of the output file for the trained model.
    - scaler_filename (str): Name of the output file for the scaler.

    Returns:
    - None (saves trained model and scaler to disk).
    """

    # Load dataset
    #df = pd.read_csv(dataset_path)

    df_source=df.copy()
    min,df=norm_data(df)
    # Use default features if none are specified
    if features is None:
        features = ['Close', 'Volume', 'rsi_14', 'MACD', 'BB_Upper', 'BB_Lower']

    # Extract feature matrix (X) and target vector (y)
    X = df[features]
    y = df[target]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    split_index = int(len(X_scaled) * (1-test_size))

    # Suddivisione sequenziale: i primi 80% sono train, gli ultimi 20% sono test
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Stampa per confermare la separazione
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Save the trained model and scaler
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"Model saved as {model_filename}")
    print(f"Scaler saved as {scaler_filename}")

    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Creiamo un DataFrame con i valori affiancati
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_results.to_csv("results.csv")
    # Stampiamo le prime righe per verificare
    #print(df_results.head(100))

def evaluate_model(model_filename="rf_model.pkl", scaler_filename="scaler.pkl", df=None, features=None, target="Target"):
    """
    Loads a trained model and scaler, applies them to a given dataset, and evaluates performance.

    Parameters:
    - model_filename (str): Path to the trained model file.
    - scaler_filename (str): Path to the scaler file.
    - df (pd.DataFrame): DataFrame containing the new data to evaluate.
    - features (list): List of feature column names to use.
    - target (str): Target column name for comparison (if available).

    Returns:
    - predictions (pd.Series): Predicted values.
    - df_results (pd.DataFrame): DataFrame with actual vs predicted values (if target column is present).
    """
    try:
        # Load the trained model and scaler
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)

        if df is None or df.empty:
            raise ValueError("Error: The input DataFrame is empty or None.")

        # Ensure features are specified
        if features is None:
            features = ['Close', 'Volume', 'rsi_14', 'MACD', 'BB_Upper', 'BB_Lower']

        # Ensure all required features exist in the dataset
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")

        # Extract feature matrix (X)
        X = df[features]

        # Normalize features using the loaded scaler
        X_scaled = scaler.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)

        # If the target column exists, evaluate performance
        if target in df.columns:
            y_actual = df[target]

            accuracy = accuracy_score(y_actual, predictions)
            classification_rep = classification_report(y_actual, predictions)
            confusion_mat = confusion_matrix(y_actual, predictions)

            print(f"Test Accuracy: {accuracy:.4f}")
            print("Classification Report:\n", classification_rep)
            print("Confusion Matrix:\n", confusion_mat)

            # Return results as DataFrame
            df_results = pd.DataFrame({'Actual': y_actual, 'Predicted': predictions})
            return predictions, df_results
        else:
            print("No target column found, returning predictions only.")
            return predictions

    except Exception as e:
        print(f"Error during model evaluation: {repr(e)}")
        return None

def evaluate_xgboost(model_filename="xgb_model.pkl", scaler_filename="scaler.pkl", df=None, features=None):
    """
    Loads a trained XGBoost model and evaluates a given dataset.
    """
    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)

        if df is None or df.empty:
            raise ValueError("Error: The input DataFrame is empty or None.")

        if features is None:
            features = ['Close', 'Volume', 'rsi_14', 'MACD', 'BB_Upper', 'BB_Lower']

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")

        X = scaler.transform(df[features])
        predictions = model.predict(X)
        df["Predicted"] = predictions

        return df

    except Exception as e:
        print(f"Error during model evaluation: {repr(e)}")
        return None

def train_xgboost(df, features=None, target="Target", test_size=0.2,
                  n_estimators=10000, learning_rate=0.001, max_depth=30, model_filename="xgb_model.pkl",
                  scaler_filename="scaler.pkl"):
    """
    Trains an XGBoost model, normalizes the data, and saves the trained model and scaler.

    Parameters:
    - dataset_path (str): Path to the dataset CSV file.
    - features (list): List of feature column names.
    - target (str): Target column name.
    - test_size (float): Proportion of data for testing.
    - n_estimators (int): Number of boosting rounds (default 500).
    - learning_rate (float): Step size shrinkage (default 0.05).
    - max_depth (int): Depth of trees (default 5).
    - model_filename (str): Name of the saved model file.
    - scaler_filename (str): Name of the saved scaler file.
    """

    try:
        # Use default features if none are provided
        if features is None:
            features = ['Close', 'Volume', 'rsi_14', 'MACD', 'BB_Upper', 'BB_Lower']

        # Ensure required features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")

        # Extract feature matrix (X) and target vector (y)
        X = df[features]
        y = df[target]

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split dataset (last 20% as test)
        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Compute class weight (balance class distribution)
        class_counts = y_train.value_counts().to_dict()
        total_samples = len(y_train)
        scale_pos_weight = {cls: total_samples / count for cls, count in class_counts.items()}

        # Train XGBoost model
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                              max_depth=max_depth, scale_pos_weight=scale_pos_weight)
        model.fit(X_train, y_train)

        # Save model and scaler
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)

        print(f"Model saved as {model_filename}")
        print(f"Scaler saved as {scaler_filename}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_rep)
        print("Confusion Matrix:\n", confusion_mat)

        return model, scaler

    except Exception as e:
        print(f"Error during XGBoost training: {repr(e)}")
        return None, None




# Example usage
if __name__ == "__main__":
    df=de.read_df("../../data/ETH-USD.csv")
    df=de.add_indicators(df)
    df=forward_looking_label(df, look_ahead=25, threshold_buy=0.08, threshold_sell=0.04)
    split_index = int(len(df) * 0.9)
    # split for final test with never seen data
    X_train, X_test = df[:split_index], df[split_index:]

    #train_xgboost(X_train)
    train_rf(X_train,test_size=0.1)
    predictions, df_results = evaluate_model(df=X_test)
