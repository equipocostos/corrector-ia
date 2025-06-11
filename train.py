
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv('dataset.csv')
X = df['nombre_erroneo']
y = df['nombre_corregido']

modelo = make_pipeline(TfidfVectorizer(), LogisticRegression())
modelo.fit(X, y)

joblib.dump(modelo, 'modelo.pkl')
print("Modelo entrenado y guardado como modelo.pkl")
