"""
Модуль для обработки дисбаланса классов.
Все функции принимают уже разделенные X_train, X_test, y_train
и возвращают обработанные версии.
"""

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings

warnings.filterwarnings('ignore')


def analyze_class_balance(y_train, y_test=None, verbose=True):
    """
    Анализ баланса классов в тренировочных и тестовых данных.

    Parameters:
    -----------
    y_train : array-like
        Целевая переменная тренировочных данных
    y_test : array-like, optional
        Целевая переменная тестовых данных
    verbose : bool
        Печатать ли результат

    Returns:
    --------
    dict со статистикой баланса классов
    """
    if verbose:
        print("=" * 60)
        print("АНАЛИЗ БАЛАНСА КЛАССОВ")
        print("=" * 60)

    # Преобразуем в Series для удобства
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    # Статистика по train
    train_counts = y_train.value_counts()
    train_percents = y_train.value_counts(normalize=True) * 100

    if verbose:
        print(f"\nTRAIN данные ({len(y_train)} samples):")
        for cls in sorted(train_counts.index):
            print(f"  Класс {cls}: {train_counts[cls]} ({train_percents[cls]:.1f}%)")

    # Коэффициент дисбаланса
    majority = train_counts.max()
    minority = train_counts.min()
    imbalance_ratio = majority / minority

    if verbose:
        print(f"\n  Коэффициент дисбаланса: {imbalance_ratio:.2f} : 1")

        if imbalance_ratio > 10:
            print(f"  ⚠ Сильный дисбаланс (>10:1)")
        elif imbalance_ratio > 3:
            print(f"  ⚠ Умеренный дисбаланс (3-10:1)")
        else:
            print(f"  ✓ Приемлемый баланс (<3:1)")

    # Статистика по test (если есть)
    test_counts_dict = None
    test_percents_dict = None

    if y_test is not None:
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test)

        test_counts = y_test.value_counts()
        test_percents = y_test.value_counts(normalize=True) * 100
        test_counts_dict = test_counts.to_dict()
        test_percents_dict = test_percents.to_dict()

        if verbose:
            print(f"\nTEST данные ({len(y_test)} samples):")
            for cls in sorted(test_counts.index):
                print(f"  Класс {cls}: {test_counts[cls]} ({test_percents[cls]:.1f}%)")

            # Сравнение распределений
            print(f"\n  Сравнение train/test:")
            for cls in sorted(train_counts.index):
                train_pct = train_percents.get(cls, 0)
                test_pct = test_percents.get(cls, 0)
                diff = abs(train_pct - test_pct)
                print(f"    Класс {cls}: train={train_pct:.1f}%, test={test_pct:.1f}% (разница={diff:.1f}%)")

    return {
        'train_counts': train_counts.to_dict(),
        'train_percents': train_percents.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'test_counts': test_counts_dict,
        'test_percents': test_percents_dict
    }


def random_undersample(X_train, y_train, sampling_strategy='auto', random_state=42, verbose=True):
    """
    Random Under-Sampling: случайное удаление примеров мажоритарного класса.

    Parameters:
    -----------
    sampling_strategy : str or float
        'auto' - баланс 1:1
        float - соотношение (0.5 = миноритарный класс 50% от мажоритарного)
    verbose : bool
        Печатать ли информацию
    """
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        replacement=False
    )

    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    if verbose:
        original_counts = pd.Series(y_train).value_counts()
        resampled_counts = pd.Series(y_resampled).value_counts()
        print(f"RandomUnderSampler:")
        print(f"  Было: {dict(original_counts)}")
        print(f"  Стало: {dict(resampled_counts)}")

    return X_resampled, y_resampled, rus


def tomek_links(X_train, y_train, sampling_strategy='auto', verbose=True):
    """
    Tomek Links: удаление пар противоположных классов, которые являются ближайшими соседями.
    Очищает границу между классами, удаляя шум.
    """
    tl = TomekLinks(sampling_strategy=sampling_strategy)

    X_resampled, y_resampled = tl.fit_resample(X_train, y_train)

    if verbose:
        removed = len(y_train) - len(y_resampled)
        print(f"TomekLinks: удалено {removed} шумовых примеров")

    return X_resampled, y_resampled, tl


def enn(X_train, y_train, sampling_strategy='auto', n_neighbors=3, verbose=True):
    """
    Edited Nearest Neighbours: удаляет примеры, которые не согласуются с соседями.
    """
    enn = EditedNearestNeighbours(
        sampling_strategy=sampling_strategy,
        n_neighbors=n_neighbors,
        kind_sel='all'
    )

    X_resampled, y_resampled = enn.fit_resample(X_train, y_train)

    if verbose:
        removed = len(y_train) - len(y_resampled)
        print(f"EditedNearestNeighbours: удалено {removed} примеров")

    return X_resampled, y_resampled, enn


def smote(X_train, y_train, sampling_strategy='auto', k_neighbors=5, random_state=42, verbose=True):
    """
    SMOTE (Synthetic Minority Over-sampling Technique):
    Создает синтетические примеры миноритарного класса путем интерполяции.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    if verbose:
        original_counts = pd.Series(y_train).value_counts()
        resampled_counts = pd.Series(y_resampled).value_counts()
        print(f"SMOTE:")
        print(f"  Было: {dict(original_counts)}")
        print(f"  Стало: {dict(resampled_counts)}")
        print(f"  Создано синтетических: {resampled_counts.max() - original_counts.min()}")

    return X_resampled, y_resampled, smote


def borderline_smote(X_train, y_train, sampling_strategy='auto',
                     k_neighbors=5, m_neighbors=10, kind='borderline-1',
                     random_state=42, verbose=True):
    """
    Borderline-SMOTE: SMOTE, фокусирующийся на граничных примерах.

    kind:
        'borderline-1' - использует ближайших соседей из любого класса
        'borderline-2' - использует ближайших соседей только из миноритарного класса
    """
    border_smote = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        m_neighbors=m_neighbors,
        kind=kind,
        random_state=random_state
    )

    X_resampled, y_resampled = border_smote.fit_resample(X_train, y_train)

    if verbose:
        print(f"Borderline-SMOTE ({kind}): создано синтетических примеров на границе классов")

    return X_resampled, y_resampled, border_smote


def adasyn(X_train, y_train, sampling_strategy='auto', n_neighbors=5, random_state=42, verbose=True):
    """
    ADASYN: адаптивный SMOTE, создает больше примеров там, где класс труднее классифицировать.
    """
    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        n_neighbors=n_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    if verbose:
        print(f"ADASYN: создает больше примеров в сложных областях")

    return X_resampled, y_resampled, adasyn


def smote_enn(X_train, y_train, sampling_strategy='auto',
              smote_k=3, enn_k=3, random_state=42, verbose=True):
    """
    SMOTE + ENN: SMOTE для увеличения, затем ENN для очистки от шума.
    """
    smote_enn = SMOTEENN(
        sampling_strategy=sampling_strategy,
        smote=SMOTE(k_neighbors=smote_k, random_state=random_state),
        enn=EditedNearestNeighbours(n_neighbors=enn_k),
        random_state=random_state
    )

    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    if verbose:
        original_counts = pd.Series(y_train).value_counts()
        resampled_counts = pd.Series(y_resampled).value_counts()
        print(f"SMOTE + ENN:")
        print(f"  Было: {dict(original_counts)}")
        print(f"  Стало: {dict(resampled_counts)}")

    return X_resampled, y_resampled, smote_enn


def smote_tomek(X_train, y_train, sampling_strategy='auto',
                smote_k=5, random_state=42, verbose=True):
    """
    SMOTE + Tomek Links: SMOTE для увеличения, затем Tomek Links для очистки границ.
    """
    smote_tomek = SMOTETomek(
        sampling_strategy=sampling_strategy,
        smote=SMOTE(k_neighbors=smote_k, random_state=random_state),
        tomek=TomekLinks(),
        random_state=random_state
    )

    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

    if verbose:
        original_counts = pd.Series(y_train).value_counts()
        resampled_counts = pd.Series(y_resampled).value_counts()
        print(f"SMOTE + Tomek:")
        print(f"  Было: {dict(original_counts)}")
        print(f"  Стало: {dict(resampled_counts)}")

    return X_resampled, y_resampled, smote_tomek


def compare_methods(X_train, y_train, X_val, y_val, methods=None, scoring='f1', verbose=True):
    """
    Сравнение нескольких методов борьбы с дисбалансом.

    Parameters:
    -----------
    X_val, y_val : валидационная выборка для оценки
    methods : list of tuples
        Список методов в формате [(name, function, kwargs), ...]
        Если None, сравниваются все основные методы
    scoring : str
        Метрика для сравнения ('f1', 'recall', 'precision', 'roc_auc')
    verbose : bool
        Печатать ли результаты

    Returns:
    --------
    DataFrame с результатами сравнения
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

    if methods is None:
        methods = [
            ('Original', None, {}),  # без обработки
            ('RandomUnderSampler', random_undersample, {}),
            ('SMOTE', smote, {'k_neighbors': 5}),
            ('Borderline-SMOTE', borderline_smote, {'kind': 'borderline-1'}),
            ('ADASYN', adasyn, {}),
            ('SMOTE-ENN', smote_enn, {}),
            ('SMOTE-Tomek', smote_tomek, {})
        ]

    results = []

    for name, func, kwargs in methods:
        if verbose:
            print(f"\n--- {name} ---")

        if func is None:
            # Без обработки
            X_tr, y_tr = X_train, y_train
        else:
            # Применяем метод (с verbose=False, чтобы не засорять вывод)
            X_tr, y_tr, _ = func(X_train, y_train, **kwargs, verbose=False)

        # Обучаем простую модель для оценки
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)

        # Оцениваем на валидации
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        # Собираем метрики
        result = {
            'Method': name,
            'Train_size': len(y_tr),
            'Train_balance': f"{pd.Series(y_tr).value_counts()[1]}/{pd.Series(y_tr).value_counts()[0]}",
            'F1': f1_score(y_val, y_pred),
            'Recall': recall_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred)
        }

        if y_proba is not None:
            try:
                result['ROC_AUC'] = roc_auc_score(y_val, y_proba)
            except:
                result['ROC_AUC'] = None

        results.append(result)

    # Сортируем по F1
    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)

    if verbose:
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ МЕТОДОВ (по убыванию F1)")
        print("=" * 80)
        print(results_df.to_string(index=False))

    return results_df


def auto_select_method(X_train, y_train, X_val, y_val,
                       quick_mode=True, verbose=True):
    """
    Автоматический выбор с опцией быстрого/полного режима.

    quick_mode=True: только 4 метода (быстро)
    quick_mode=False: все методы (медленно, но точно)
    verbose=True: печатать процесс выбора
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    if quick_mode:
        # Быстрый режим (4 метода)
        methods = [
            ('Original', None),
            ('RandomUnderSampler', lambda X, y: random_undersample(X, y, verbose=False)[:2]),
            ('SMOTE', lambda X, y: smote(X, y, verbose=False)[:2]),
            ('SMOTE-Tomek', lambda X, y: smote_tomek(X, y, verbose=False)[:2])
        ]
    else:
        # Полный режим (все методы)
        methods = [
            ('Original', None),
            ('RandomUnderSampler', lambda X, y: random_undersample(X, y, verbose=False)[:2]),
            ('TomekLinks', lambda X, y: tomek_links(X, y, verbose=False)[:2]),
            ('ENN', lambda X, y: enn(X, y, verbose=False)[:2]),
            ('SMOTE', lambda X, y: smote(X, y, verbose=False)[:2]),
            ('Borderline-SMOTE', lambda X, y: borderline_smote(X, y, verbose=False)[:2]),
            ('ADASYN', lambda X, y: adasyn(X, y, verbose=False)[:2]),
            ('SMOTE-ENN', lambda X, y: smote_enn(X, y, verbose=False)[:2]),
            ('SMOTE-Tomek', lambda X, y: smote_tomek(X, y, verbose=False)[:2])
        ]

    best_score = -1
    best_method = 'Original'
    best_func = None

    if verbose:
        print(f"Режим: {'БЫСТРЫЙ' if quick_mode else 'ПОЛНЫЙ'}")
        print("-" * 40)

    for name, func in methods:
        try:
            if func is None:
                X_tr, y_tr = X_train, y_train
            else:
                X_tr, y_tr = func(X_train, y_train)

            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)

            score = f1_score(y_val, model.predict(X_val))

            if verbose:
                print(f"{name}: F1 = {score:.4f}")

            if score > best_score:
                best_score = score
                best_method = name
                best_func = func

        except Exception as e:
            if verbose:
                print(f"{name}: Ошибка - {e}")

    if verbose:
        print("\n" + "=" * 50)
        print(f"✅ Лучший метод: {best_method} (F1 = {best_score:.4f})")

    return best_method, best_func
