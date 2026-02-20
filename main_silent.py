"""
Главный модуль пайплайна обработки данных для Customer Churn.
Сравнение качества модели на сырых и обработанных данных.
Последовательность обработки: выбросы → пропуски → фичи → кодирование → отбор → дисбаланс
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Импорт модулей проекта
from outliers import *
from missing_values import *
from split import *
from encoding_features import *
from Feature_Engineering import *
from feature_selection import *
from imbalanced_classes import *
from models import *

# ============================================
# 1. ПОДГОТОВКА ДАННЫХ
# ============================================

# Кодирование целевой переменной (один раз для всех экспериментов)
y_train, y_test, encoder = encode_target(y_train, y_test, return_encoder=True, verbose=False)

# Сохраняем исходные признаки для первого запуска
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

print("=" * 60)
print("ЭКСПЕРИМЕНТ 1: МОДЕЛЬ НА СЫРЫХ ДАННЫХ")
print("=" * 60)

# Определяем категориальные признаки для CatBoost
cat_features_raw = X_train_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

# Обучение и оценка на сырых данных
model_raw, _ = train_model(
    get_catboost(cat_features=cat_features_raw, auto_class_weights='Balanced'),
    X_train_raw, y_train,
    verbose=False
)
metrics_raw = evaluate_model(model_raw, X_test_raw, y_test)

# ============================================
# 2. ПОЛНЫЙ ПАЙПЛАЙН ОБРАБОТКИ
# ============================================

print("\n" + "=" * 60)
print("ОБРАБОТКА ДАННЫХ")
print("=" * 60)

# Копируем исходные данные для обработки
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

# 2.1 Обработка выбросов
# mask, lower_bounds, upper_bounds = detection_MAD(X_train)
# X_train = deleting_outliers_train(X_train, mask)
# y_train = y_train[~mask]  # синхронизируем целевую переменную
#
# X_test = capping_outliers_test(
#     X_train, X_test, lower_bounds, upper_bounds,
#     exclude_columns=["tenure", "MonthlyCharges", "TotalCharges"]
# )

if detection_categorical_outliers(X_train)[0].sum():
    X_train, X_test, _ = process_categorical_outliers(X_train, X_test, strategy="combined")

# 2.2 Обработка пропусков
X_train, X_test, _ = impute_numeric_knn(X_train, X_test)

# 2.3 Feature engineering
X_train, X_test = manual_combinations(X_train, X_test, verbose=False)
X_train, X_test = add_interactions(X_train, X_test, verbose=False)

# 2.4 Кодирование категориальных признаков (One-Hot)
X_train, X_test, ohe = one_hot_encode_split(X_train, X_test, verbose=False)

# 2.5 Отбор признаков
X_train, X_test, selected_features = select_features(X_train, X_test, y_train, verbose=False)

# 2.6 Обработка дисбаланса классов
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

method_name, method_func = auto_select_method(
    X_train_split, y_train_split, X_val, y_val,
    quick_mode=True,
    verbose=False
)

X_train, y_train = method_func(X_train, y_train)

print(f"\nВыбранный метод борьбы с дисбалансом: {method_name}")

# ============================================
# 3. ЭКСПЕРИМЕНТ 2: МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ
# ============================================

print("\n" + "=" * 60)
print("ЭКСПЕРИМЕНТ 2: МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ")
print("=" * 60)

# После One-Hot все признаки числовые, cat_features не нужен
model_processed, _ = train_model(
    get_catboost(),
    X_train, y_train,
    verbose=False
)
metrics_processed = evaluate_model(model_processed, X_test, y_test)

# ============================================
# 4. СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================

print("\n" + "=" * 60)
print("СРАВНЕНИЕ МЕТРИК")
print("=" * 60)

comparison = pd.DataFrame({
    'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    'Сырые данные': [
        metrics_raw['accuracy'],
        metrics_raw['precision'],
        metrics_raw['recall'],
        metrics_raw['f1'],
        metrics_raw.get('roc_auc', None)
    ],
    'Обработанные данные': [
        metrics_processed['accuracy'],
        metrics_processed['precision'],
        metrics_processed['recall'],
        metrics_processed['f1'],
        metrics_processed.get('roc_auc', None)
    ]
})

print(comparison.round(4).to_string(index=False))
