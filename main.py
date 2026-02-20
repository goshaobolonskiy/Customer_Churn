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

print("=" * 80)
print("ПАЙПЛАЙН ОБРАБОТКИ ДАННЫХ CUSTOMER CHURN".center(80))
print("=" * 80)

# Кодирование целевой переменной (один раз для всех экспериментов)
print("\n1. ПОДГОТОВКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print("-" * 60)
y_train, y_test, encoder = encode_target(y_train, y_test, return_encoder=True, verbose=True)

# Сохраняем исходные признаки для первого запуска
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ 1: МОДЕЛЬ НА СЫРЫХ ДАННЫХ".center(80))
print("=" * 80)

# Определяем категориальные признаки для CatBoost
cat_features_raw = X_train_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
print(f"\nКатегориальные признаки: {cat_features_raw}")

# Обучение и оценка на сырых данных
print("\nОбучение CatBoost на сырых данных...")
model_raw, _ = train_model(
    get_catboost(cat_features=cat_features_raw, auto_class_weights='Balanced'),
    X_train_raw, y_train,
    verbose=True
)
print("\nОценка модели на сырых данных:")
metrics_raw = evaluate_model(model_raw, X_test_raw, y_test, verbose=True)

# ============================================
# 2. ПОЛНЫЙ ПАЙПЛАЙН ОБРАБОТКИ
# ============================================

print("\n" + "=" * 80)
print("ЭТАП 1: ОБРАБОТКА ВЫБРОСОВ".center(80))
print("=" * 80)

# Копируем исходные данные для обработки
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

# 2.1 Обработка выбросов
print("\nОбнаружение выбросов методом MAD...")
mask, lower_bounds, upper_bounds = detection_MAD(X_train)
print(f"Обнаружено выбросов: {mask.sum()} строк ({mask.sum() / len(mask) * 100:.1f}%)")

print("\nУдаление выбросов из train...")
X_train = deleting_outliers_train(X_train, mask)
y_train = y_train[~mask]
print(f"Train после удаления: {X_train.shape}")

print("\nCapping выбросов в test (бизнес-логика: высокие сборы - лояльные клиенты)...")
X_test = capping_outliers_test(
    X_train, X_test, lower_bounds, upper_bounds,
    exclude_columns=["tenure", "MonthlyCharges", "TotalCharges"]
)
print(f"Test после capping: {X_test.shape}")

print("\nОбработка категориальных выбросов...")
if detection_categorical_outliers(X_train)[0].sum():
    X_train, X_test, _ = process_categorical_outliers(X_train, X_test, strategy="combined")
    print("Категориальные выбросы обработаны")
else:
    print("Категориальные выбросы не обнаружены")

print("\n" + "=" * 80)
print("ЭТАП 2: ОБРАБОТКА ПРОПУСКОВ".center(80))
print("=" * 80)

# 2.2 Обработка пропусков
missing_before = X_train.isna().sum().sum()
print(f"\nПропусков до обработки: {missing_before}")
X_train, X_test, _ = impute_numeric_knn(X_train, X_test)
missing_after = X_train.isna().sum().sum()
print(f"Пропусков после обработки: {missing_after}")

print("\n" + "=" * 80)
print("ЭТАП 3: FEATURE ENGINEERING".center(80))
print("=" * 80)

# 2.3 Feature engineering
print("\nСоздание ручных комбинаций признаков...")
X_train, X_test = manual_combinations(X_train, X_test, verbose=True)

print("\nСоздание автоматических взаимодействий...")
X_train, X_test = add_interactions(X_train, X_test, verbose=True)

print(f"\nРазмер после FE: train {X_train.shape}, test {X_test.shape}")

print("\n" + "=" * 80)
print("ЭТАП 4: КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ".center(80))
print("=" * 80)

# 2.4 Кодирование категориальных признаков (One-Hot)
print("\nOne-Hot Encoding категориальных признаков...")
X_train, X_test, ohe = one_hot_encode_split(X_train, X_test, verbose=True)

print("\n" + "=" * 80)
print("ЭТАП 5: ОТБОР ПРИЗНАКОВ".center(80))
print("=" * 80)

# 2.5 Отбор признаков
X_train, X_test, selected_features = select_features(X_train, X_test, y_train, verbose=True)
print(f"\nОтобрано признаков: {len(selected_features)}")

print("\n" + "=" * 80)
print("ЭТАП 6: ОБРАБОТКА ДИСБАЛАНСА КЛАССОВ".center(80))
print("=" * 80)

# 2.6 Обработка дисбаланса классов
print("\nСоздание валидационной выборки...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)
print(f"Train для выбора метода: {X_train_split.shape}")
print(f"Val для оценки: {X_val.shape}")

print("\nАвтоматический выбор метода борьбы с дисбалансом...")
method_name, method_func = auto_select_method(
    X_train_split, y_train_split, X_val, y_val,
    quick_mode=True,
    verbose=True
)

print(f"\nПрименение метода {method_name} ко всем данным...")
X_train, y_train = method_func(X_train, y_train)
print(f"Размер после обработки дисбаланса: {X_train.shape}")

# ============================================
# 3. ЭКСПЕРИМЕНТ 2: МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ
# ============================================

print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ 2: МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ".center(80))
print("=" * 80)

# После One-Hot все признаки числовые, cat_features не нужен
print("\nОбучение CatBoost на обработанных данных...")
model_processed, _ = train_model(
    get_catboost(),
    X_train, y_train,
    verbose=True
)
print("\nОценка модели на обработанных данных:")
metrics_processed = evaluate_model(model_processed, X_test, y_test, verbose=True)

# ============================================
# 4. СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================

print("\n" + "=" * 80)
print("СРАВНЕНИЕ МЕТРИК".center(80))
print("=" * 80)

comparison = pd.DataFrame({
    'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    'Сырые данные': [
        f"{metrics_raw['accuracy']:.4f}",
        f"{metrics_raw['precision']:.4f}",
        f"{metrics_raw['recall']:.4f}",
        f"{metrics_raw['f1']:.4f}",
        f"{metrics_raw.get('roc_auc', 0):.4f}"
    ],
    'Обработанные данные': [
        f"{metrics_processed['accuracy']:.4f}",
        f"{metrics_processed['precision']:.4f}",
        f"{metrics_processed['recall']:.4f}",
        f"{metrics_processed['f1']:.4f}",
        f"{metrics_processed.get('roc_auc', 0):.4f}"
    ]
})

print("\n")
print(comparison.to_string(index=False))

# Подсчет разницы
print("\n" + "-" * 60)
print("ИЗМЕНЕНИЕ МЕТРИК")
print("-" * 60)

metrics_diff = {
    'Accuracy': metrics_processed['accuracy'] - metrics_raw['accuracy'],
    'Precision': metrics_processed['precision'] - metrics_raw['precision'],
    'Recall': metrics_processed['recall'] - metrics_raw['recall'],
    'F1': metrics_processed['f1'] - metrics_raw['f1']
}

for metric, diff in metrics_diff.items():
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "="
    print(f"{metric}: {arrow} {abs(diff):.4f}")

print("\n" + "=" * 80)
