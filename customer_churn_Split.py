from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Читаем DF из файла

# Перевод ошибочного obj в numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Удаление ненужных данных
df = df.drop(["customerID"], axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # для сохранения баланса классов
)
