# 🏦 Churn Prediction — Bank Customer

Modelo de Machine Learning para predecir la fuga de clientes bancarios, desarrollado como proyecto de ML Engineering.

## 📌 Objetivo
Predecir qué clientes tienen mayor probabilidad de abandonar el banco para permitir acciones preventivas del área comercial.

## 📊 Dataset
- **Fuente:** [Bank Customer Churn Prediction — Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)
- 10,000 clientes con 14 variables
- Variable objetivo: `Exited` (`1` = churn, `0` = no churn)
- Balance de clases: aproximadamente 80% no churn / 20% churn

## 🧠 Avance actual
- Exploración y análisis de datos en `notebooks/01_eda.ipynb`
- Ingeniería de features en `notebooks/02_features.ipynb`
- Modelos base y evaluación en `notebooks/03_baseline.ipynb`
- Pipeline completo con preprocessing y entrenamiento en `notebooks/04_pipeline.ipynb`
- Ajuste de threshold y experimentos con SMOTE en `notebooks/05_threshold_smote.ipynb`
- Script de entrenamiento reproducible en `src/train.py`
- Tracking de experimentos con MLflow y almacenamiento local

## 🤖 Solución actual
- Pipeline con `StandardScaler` para numéricos y `OneHotEncoder` para categorías
- Modelo final: `RandomForestClassifier`
- Búsqueda de hiperparámetros con `GridSearchCV`
- Evaluación usando AUC-ROC y métricas de churn (recall, precision, f1)
- Registro de modelo y métricas en MLflow

## ⚙️ Stack
- Python 3.11
- pandas, numpy, scikit-learn
- MLflow
- Jupyter Notebook

## 📁 Estructura

churn-prediction/
├── data/                  # Dataset
├── mlruns/                # Experimentos MLflow
├── notebooks/             # Notebooks de análisis y modelado
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_pipeline.ipynb
│   └── 05_threshold_smote.ipynb
├── src/
│   └── train.py           # Script de entrenamiento reproducible
├── requirements.txt
└── README.md

## 🚀 Cómo ejecutar
```bash
git clone https://github.com/danielgc090997/churn-prediction.git
cd churn-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```

## 📈 Resultados esperados
El pipeline actual registra un experimento MLflow con métricas como:
- `auc_roc`
- `recall_churn`
- `precision_churn`
- `f1_churn`

## 🛠️ Próximos pasos
- Probar umbrales diferentes para priorizar recall
- Balancear clases con SMOTE en el pipeline
- Convertir el modelo a una API con FastAPI
- Empaquetar el pipeline como un servicio reproducible
