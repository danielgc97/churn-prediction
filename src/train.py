import pandas as pd 
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn

def load_data(path: str) -> pd.DataFrame:
    '''Carga los datos de entrenamiento y prueba.'''
    df = pd.read_csv(path)
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    return df

def build_pipeline() -> Pipeline:
    """Construye el pipeline de preprocessing + modelo."""
    
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                      'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                      'EstimatedSalary']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def train(threshold: float = 0.3):
    """Entrena el modelo y registra el experimento en MLflow."""
    
    # Cargar datos
    df = load_data('data/Churn_Modelling.csv')
    
    # Split
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Construir pipeline
    pipeline = build_pipeline()
    
    # Tuning
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 15],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5,
        scoring='roc_auc', n_jobs=1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Evaluar con threshold
    best_pipeline = grid_search.best_estimator_
    y_proba = best_pipeline.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-prediction")
    
    with mlflow.start_run(run_name="pipeline_rf_tuned_script"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("recall_churn", report['1']['recall'])
        mlflow.log_metric("precision_churn", report['1']['precision'])
        mlflow.log_metric("f1_churn", report['1']['f1-score'])
        mlflow.sklearn.log_model(best_pipeline, name="pipeline_rf_churn")
    
    print(f"Threshold: {threshold}")
    print(f"AUC-ROC: {auc:.3f}")
    print(f"Recall churn: {report['1']['recall']:.3f}")
    print("Entrenamiento completado ✓")
    
if __name__ == "__main__":
    train()

