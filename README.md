# 🏦 Churn Prediction — Bank Customer

Modelo de Machine Learning para predecir la fuga de clientes bancarios, desarrollado como parte de un portafolio de proyectos orientado a ML Engineer.

## 📌 Objetivo
Predecir qué clientes tienen mayor probabilidad de abandonar el banco, permitiendo al equipo comercial intervenir proactivamente.

## 📊 Dataset
- **Fuente:** [Bank Customer Churn Prediction — Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)
- 10,000 clientes con 14 variables
- Variable objetivo: `Exited` (1 = se fue, 0 = se quedó)
- Distribución: 80% no churn / 20% churn

## 🔍 Hallazgos del EDA
- Clientes en **Alemania** tienen mayor tasa de churn
- **Mujeres** desertan más que hombres
- Clientes con **3-4 productos** tienen alta fuga (contra-intuitivo)
- **Inactividad** es el predictor más claro de churn
- **Edad** es la variable numérica con mayor correlación (0.29)

## 🤖 Modelos
| Modelo | AUC-ROC | Recall Churn | Precision Churn |
|--------|---------|--------------|-----------------|
| Logistic Regression | 0.75 | 0.19 | 0.59 |
| Random Forest (base) | 0.854 | 0.45 | 0.77 |
| Random Forest (tuned) | 0.864 | 0.45 | 0.82 |

## ⚙️ Stack
- Python 3.11
- pandas, numpy, scikit-learn
- MLflow (experiment tracking)
- Jupyter Notebooks

## 📁 Estructura

churn-prediction/
├── data/                  # Dataset (no incluido en repo)
├── notebooks/
│   ├── 01_eda.ipynb       # Análisis exploratorio
│   ├── 02_features.ipynb  # Feature engineering
│   └── 03_baseline.ipynb  # Modelos y tuning
├── .gitignore
└── README.md

## 🚀 Cómo reproducir
```bash
git clone https://github.com/danielgc090997/churn-prediction.git
cd churn-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 📈 Próximos pasos
- Ajustar threshold para maximizar recall
- Aplicar SMOTE para balancear clases
- Deployar modelo como API con FastAPI