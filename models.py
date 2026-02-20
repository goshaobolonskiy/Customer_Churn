"""
Модуль с моделями машинного обучения для Customer Churn.
Содержит функции для обучения, сравнения и сохранения моделей.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import warnings

warnings.filterwarnings('ignore')


# ============================================
# 1. БАЗОВЫЕ МОДЕЛИ (SKLEARN)
# ============================================

def get_logistic_regression(**kwargs):
    """Логистическая регрессия"""
    from sklearn.linear_model import LogisticRegression

    params = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    params.update(kwargs)

    return LogisticRegression(**params)


def get_random_forest(**kwargs):
    """Случайный лес"""
    from sklearn.ensemble import RandomForestClassifier

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    params.update(kwargs)

    return RandomForestClassifier(**params)


def get_gradient_boosting(**kwargs):
    """Градиентный бустинг (sklearn)"""
    from sklearn.ensemble import GradientBoostingClassifier

    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    params.update(kwargs)

    return GradientBoostingClassifier(**params)


# ============================================
# 2. БУСТИНГИ
# ============================================

def get_xgboost(**kwargs):
    """XGBoost"""
    from xgboost import XGBClassifier

    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    params.update(kwargs)

    return XGBClassifier(**params)


def get_lightgbm(**kwargs):
    """LightGBM"""
    from lightgbm import LGBMClassifier

    params = {
        'n_estimators': 100,
        'max_depth': -1,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    params.update(kwargs)

    return LGBMClassifier(**params)


def get_catboost(**kwargs):
    """CatBoost"""
    from catboost import CatBoostClassifier

    params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'random_seed': 42,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }
    params.update(kwargs)

    return CatBoostClassifier(**params)


# ============================================
# 3. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================

def train_model(model, X_train, y_train, X_val=None, y_val=None, verbose=True):
    """
    Обучение модели с опциональной валидацией.

    Parameters:
    -----------
    model : sklearn-совместимая модель
    X_train, y_train : обучающие данные
    X_val, y_val : валидационные данные (опционально)
    verbose : bool, печатать ли информацию

    Returns:
    --------
    model : обученная модель
    history : dict с историей обучения (если есть)
    """
    history = {}

    # Для моделей с поддержкой early_stopping
    if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            if hasattr(model, 'evals_result_'):
                history = model.evals_result_
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    if verbose:
        print(f"  Модель {model.__class__.__name__} обучена")

    return model, history


# ============================================
# 4. ОЦЕНКА МОДЕЛЕЙ
# ============================================

def evaluate_model(model, X_test, y_test, verbose=True):
    """
    Оценка модели на тестовых данных.

    Returns:
    --------
    dict со всеми метриками
    """
    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Основные метрики
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # ROC-AUC (если есть вероятности)
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['roc_auc'] = None

    # Матрица ошибок
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    if verbose:
        print(f"\n📊 Оценка модели {model.__class__.__name__}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        if metrics.get('roc_auc'):
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\n  Матрица ошибок:")
        print(f"  {metrics['confusion_matrix']}")

    return metrics


# ============================================
# 5. СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================

def compare_models(models_dict, X_train, y_train, X_test, y_test, verbose=True):
    """
    Сравнение нескольких моделей.

    Parameters:
    -----------
    models_dict : dict
        Словарь вида {'название': модель}

    Returns:
    --------
    DataFrame с результатами сравнения
    """
    results = []

    for name, model in models_dict.items():
        if verbose:
            print(f"\n--- {name} ---")

        # Обучение
        model, _ = train_model(model, X_train, y_train, verbose=verbose)

        # Оценка
        metrics = evaluate_model(model, X_test, y_test, verbose=verbose)

        # Сохраняем результаты
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'ROC_AUC': metrics.get('roc_auc', None)
        })

    # Сортируем по F1
    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)

    if verbose:
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ МОДЕЛЕЙ (по убыванию F1)")
        print("=" * 80)
        print(results_df.to_string(index=False))

    return results_df


# ============================================
# 6. ПОИСК ЛУЧШЕЙ МОДЕЛИ
# ============================================

def find_best_model(X_train, y_train, X_test, y_test,
                    models_to_try='all', verbose=True):
    """
    Поиск лучшей модели из предопределенного набора.

    Parameters:
    -----------
    models_to_try : str or list
        'all' - все модели
        'sklearn' - только sklearn модели
        'boosting' - только бустинги
        list - список названий
    """

    # Определяем набор моделей
    all_models = {
        'LogisticRegression': get_logistic_regression(),
        'RandomForest': get_random_forest(),
        'GradientBoosting': get_gradient_boosting(),
        'XGBoost': get_xgboost(),
        'LightGBM': get_lightgbm(),
        'CatBoost': get_catboost()
    }

    if models_to_try == 'all':
        models = all_models
    elif models_to_try == 'sklearn':
        models = {k: v for k, v in all_models.items()
                  if k in ['LogisticRegression', 'RandomForest', 'GradientBoosting']}
    elif models_to_try == 'boosting':
        models = {k: v for k, v in all_models.items()
                  if k in ['XGBoost', 'LightGBM', 'CatBoost']}
    elif isinstance(models_to_try, list):
        models = {k: all_models[k] for k in models_to_try if k in all_models}
    else:
        models = all_models

    # Сравниваем модели
    results = compare_models(models, X_train, y_train, X_test, y_test, verbose=verbose)

    # Возвращаем лучшую модель
    best_model_name = results.iloc[0]['Model']
    best_model = all_models[best_model_name]

    if verbose:
        print(f"\n🏆 Лучшая модель: {best_model_name} (F1 = {results.iloc[0]['F1']:.4f})")

    return best_model_name, best_model, results
