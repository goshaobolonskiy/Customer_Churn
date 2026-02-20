"""
Модуль для отбора признаков (Feature Selection)
Использует комбинацию методов для выбора лучших признаков
"""

import pandas as pd
import numpy as np


# ============================================
# 1. БАЗОВЫЕ ФИЛЬТРЫ
# ============================================

def remove_low_variance(X_train, X_test, threshold=0.0, verbose=True):
    """
    Удаляет признаки с дисперсией ниже порога.

    Parameters:
    -----------
    threshold : float, default=0.0
        - 0.0: удаляем только константы (все значения одинаковые)
        - 0.01: удаляем признаки с очень низкой вариативностью
        - выше: более агрессивное удаление

    Работает в два этапа:
    1. Сначала удаляет константы (var = 0)
    2. Потом удаляет признаки с дисперсией ниже порога
    """
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train.fillna(X_train.median()))

    kept_mask = selector.get_support()
    dropped_cols = X_train.columns[~kept_mask].tolist()

    if dropped_cols and verbose:
        if threshold == 0.0:
            print(f"  Удалено констант: {len(dropped_cols)}")
        else:
            print(f"  Удалено с дисперсией <{threshold}: {len(dropped_cols)}")

    X_train_clean = X_train.drop(columns=dropped_cols, errors='ignore')
    X_test_clean = X_test.drop(columns=dropped_cols, errors='ignore')

    return X_train_clean, X_test_clean, dropped_cols


def remove_high_missing(X_train, X_test, threshold=0.9, verbose=True):
    """
    Удаляет признаки с процентом пропусков выше порога.

    Parameters:
    -----------
    threshold : float, default=0.9
        Порог удаления (0.9 = 90% пропусков)
    """
    missing_pct = X_train.isnull().mean()
    dropped_cols = missing_pct[missing_pct > threshold].index.tolist()

    if dropped_cols and verbose:
        print(f"  Удалено с >{threshold * 100}% пропусков: {len(dropped_cols)}")

    X_train_clean = X_train.drop(columns=dropped_cols, errors='ignore')
    X_test_clean = X_test.drop(columns=dropped_cols, errors='ignore')

    return X_train_clean, X_test_clean, dropped_cols


# ============================================
# 2. СТАТИСТИЧЕСКИЕ МЕТОДЫ
# ============================================

def select_by_mutual_info(X_train, y_train, threshold=0.001, verbose=True):
    """
    Отбор по Mutual Information.

    Mutual Information измеряет любую зависимость между признаком и таргетом
    (не только линейную). Значение > 0 означает, что признак несет информацию.

    Parameters:
    -----------
    threshold : float, default=0.001
        Очень низкий порог, чтобы не отсечь потенциально важные признаки
    """
    from sklearn.feature_selection import mutual_info_classif

    # Mutual information не работает с NaN, поэтому заполняем
    X_filled = X_train.fillna(X_train.median())

    mi_scores = mutual_info_classif(X_filled, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_train.columns)

    selected = mi_series[mi_series > threshold].index.tolist()

    if verbose:
        print(f"  MI: отобрано {len(selected)} признаков")
        if selected:
            top5 = mi_series.nlargest(5)
            print(f"    Топ-5 MI: {dict(top5)}")

    return selected, mi_series


# ============================================
# 3. МОДЕЛЬНЫЕ МЕТОДЫ
# ============================================

def select_by_rf_importance(X_train, y_train, threshold='auto', verbose=True):
    """
    Отбор по важности в Random Forest.

    Random Forest Importance считает, как часто признак используется
    для разбиения и насколько он уменьшает неопределенность.

    Parameters:
    -----------
    threshold : str or float
        'auto' - берем признаки выше 30% от среднего
        'mean' - берем признаки выше среднего
        число - конкретный порог
    """
    from sklearn.ensemble import RandomForestClassifier

    # Заполняем пропуски
    X_filled = X_train.fillna(X_train.median())

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_filled, y_train)

    importance = pd.Series(rf.feature_importances_, index=X_train.columns)

    # Определяем порог
    if threshold == 'auto':
        # Берем признаки выше 30% от среднего (мягкий порог)
        threshold = importance.mean() * 0.3
    elif threshold == 'mean':
        threshold = importance.mean()

    selected = importance[importance > threshold].index.tolist()

    if verbose:
        print(f"  RF Importance: отобрано {len(selected)} признаков (порог={threshold:.4f})")
        if selected:
            top5 = importance.nlargest(5)
            print(f"    Топ-5 RF: {dict(top5)}")

    return selected, importance


def select_by_permutation(X_train, y_train, n_repeats=5, threshold=0, verbose=True):
    """
    Отбор по Permutation Importance (самый надежный метод).

    Идея: перемешиваем значения признака и смотрим, как падает качество.
    Если падение сильное - признак важен.
    Если не изменилось - признак бесполезен.

    Parameters:
    -----------
    n_repeats : int, default=5
        Количество перемешиваний для усреднения
    threshold : float, default=0
        Порог важности (обычно 0 - оставляем все с хоть какой-то важностью)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    # Заполняем пропуски
    X_filled = X_train.fillna(X_train.median())

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_filled, y_train)

    # Считаем permutation importance
    perm = permutation_importance(
        rf, X_filled, y_train,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring='roc_auc'
    )

    importance = pd.Series(perm.importances_mean, index=X_train.columns)
    std = pd.Series(perm.importances_std, index=X_train.columns)

    # Берем признаки с важностью > порога
    selected = importance[importance > threshold].index.tolist()

    if verbose:
        print(f"  Permutation: отобрано {len(selected)} признаков")
        if selected:
            top5 = importance.nlargest(5)
            print(f"    Топ-5 Perm: {dict(top5)}")

    return selected, importance, std


# ============================================
# 4. УДАЛЕНИЕ КОРРЕЛИРУЮЩИХ
# ============================================

def remove_high_correlation(X_train, X_test, importance_dict=None,
                            threshold=0.95, verbose=True):
    """
    Удаляет один из пары сильно коррелирующих признаков.

    Почему threshold=0.95:
    - Корреляция 0.95+ означает, что признаки практически дублируют друг друга
    - Они несут почти одинаковую информацию
    - Оставляем тот, у которого выше важность

    Если взять threshold=0.9:
    - Удалим больше признаков
    - Но можем потерять важные, которые просто похожи, но не идентичны
    - Рекомендуется для сильного упрощения модели

    Parameters:
    -----------
    threshold : float, default=0.95
        Порог корреляции для удаления
        - 0.95 - только явные дубли
        - 0.9 - умеренное удаление
        - 0.8 - агрессивное удаление
    importance_dict : dict
        Словарь важности признаков (чтобы решить, какой оставить)
    """
    # Берем только числовые
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        return X_train, X_test, []

    # Считаем корреляцию
    corr_matrix = X_train[numeric_cols].corr().abs()

    # Берем верхний треугольник (чтобы не дублировать пары)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()

    for col in upper.columns:
        # Находим признаки, коррелирующие с текущим
        high_corr = upper[col][upper[col] > threshold].index.tolist()

        for corr_col in high_corr:
            # Если есть словарь важности, удаляем менее важный
            if importance_dict:
                imp_col = importance_dict.get(col, 0)
                imp_corr = importance_dict.get(corr_col, 0)

                if imp_col < imp_corr:
                    to_drop.add(col)
                else:
                    to_drop.add(corr_col)
            else:
                # Иначе удаляем второй
                to_drop.add(corr_col)

    dropped_cols = list(to_drop)

    if dropped_cols and verbose:
        print(f"  Удалено коррелирующих (r>{threshold}): {len(dropped_cols)}")

    X_train_clean = X_train.drop(columns=dropped_cols, errors='ignore')
    X_test_clean = X_test.drop(columns=dropped_cols, errors='ignore')

    return X_train_clean, X_test_clean, dropped_cols


# ============================================
# 5. ВАЛИДАЦИЯ
# ============================================

def validate_feature_set(X_train, y_train, selected_features,
                         baseline_score=None, quality_loss=0.05, verbose=True):
    """
    Проверяет, не слишком ли большая потеря качества после отбора.

    Parameters:
    -----------
    quality_loss : float, default=0.05
        Допустимая потеря качества:
        0.00 - не теряем качество (максимальная точность)
        0.05 - готовы потерять 5% ради упрощения (рекомендуется)
        0.10 - готовы потерять 10% (сильное упрощение)
        None - не проверяем
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    if len(selected_features) == X_train.shape[1]:
        if verbose:
            print("  Все признаки оставлены, валидация не требуется")
        return True, 0

    # Заполняем пропуски
    X_full = X_train.fillna(X_train.median())
    X_sel = X_train[selected_features].fillna(X_train[selected_features].median())

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    # Если baseline не передан, считаем на всех признаках
    if baseline_score is None:
        score_full = cross_val_score(rf, X_full, y_train, cv=3, scoring='roc_auc').mean()
    else:
        score_full = baseline_score

    score_sel = cross_val_score(rf, X_sel, y_train, cv=3, scoring='roc_auc').mean()

    loss = (score_full - score_sel) / score_full if score_full > 0 else 0

    if verbose:
        print(f"\n  Валидация:")
        print(f"    Все признаки: AUC = {score_full:.4f}")
        print(f"    Отобранные:   AUC = {score_sel:.4f}")
        print(f"    Потеря:       {loss * 100:.2f}%")

    if loss > quality_loss:
        if verbose:
            print(f"  ⚠ Потеря {loss * 100:.2f}% > {quality_loss * 100}%")
        return False, loss
    else:
        if verbose:
            print(f"  ✓ Потеря в пределах {quality_loss * 100}%")
        return True, loss


# ============================================
# 6. ОСНОВНАЯ ФУНКЦИЯ (ВСЕ ВМЕСТЕ)
# ============================================

def select_features(X_train, X_test, y_train,
                    # Настройки фильтрации
                    var_threshold=0.0,  # 0.0 - только константы
                    missing_threshold=0.9,  # 90% пропусков

                    # Методы отбора
                    use_mi=True,
                    mi_threshold=0.001,
                    use_rf=True,
                    rf_threshold='auto',
                    use_permutation=True,
                    perm_threshold=0,
                    perm_repeats=5,

                    # Постобработка
                    remove_correlated=True,
                    corr_threshold=0.95,  # 0.95 - только дубли
                    max_features=None,

                    # Валидация
                    validate=True,
                    quality_loss=0.05,  # 5% - разумный компромисс

                    # Прочее
                    verbose=True):
    """
    ПОЛНЫЙ ПАЙПЛАЙН ОТБОРА ПРИЗНАКОВ

    Логика работы:
    1. Сначала удаляем явный мусор (константы, признаки с кучей пропусков)
    2. Несколькими методами собираем кандидатов (MI, RF, Permutation)
    3. Объединяем результаты (берем все, что важно хотя бы по одному методу)
    4. Удаляем дублирующиеся признаки (сильно коррелирующие)
    5. Ограничиваем количество (если нужно)
    6. Проверяем, что качество не упало слишком сильно

    Parameters:
    -----------
    quality_loss : float
        Допустимая потеря качества
        0.00 - не теряем качество (максимальная точность)
        0.05 - готовы потерять 5% ради упрощения (рекомендуется)
        0.10 - готовы потерять 10% (сильное упрощение)
        None - не проверяем качество
    """

    if verbose:
        print("=" * 60)
        print("ОТБОР ПРИЗНАКОВ")
        print("=" * 60)
        print(f"Исходное количество: {X_train.shape[1]}")

    # Копируем
    X_tr = X_train.copy()
    X_te = X_test.copy()

    removed_all = []

    # ========================================
    # 1. БАЗОВЫЕ ФИЛЬТРЫ
    # ========================================

    # Удаляем по дисперсии (константы + низкая вариативность)
    X_tr, X_te, dropped = remove_low_variance(X_tr, X_te, var_threshold, verbose)
    removed_all.extend(dropped)

    # Удаляем с большим процентом пропусков
    X_tr, X_te, dropped = remove_high_missing(X_tr, X_te, missing_threshold, verbose)
    removed_all.extend(dropped)

    if X_tr.shape[1] == 0:
        print("❌ Все признаки удалены!")
        return X_train[[]], X_test[[]], []

    # ========================================
    # 2. СБОР КАНДИДАТОВ
    # ========================================

    candidate_sets = []
    importance_dict = {}

    # Mutual Information
    if use_mi:
        selected, mi_series = select_by_mutual_info(X_tr, y_train, mi_threshold, verbose)
        candidate_sets.append(set(selected))
        importance_dict.update(mi_series.to_dict())

    # RF Importance
    if use_rf:
        selected, rf_series = select_by_rf_importance(X_tr, y_train, rf_threshold, verbose)
        candidate_sets.append(set(selected))
        importance_dict.update(rf_series.to_dict())

    # Permutation Importance (только если не слишком много признаков)
    if use_permutation and X_tr.shape[1] < 200:
        selected, perm_series, _ = select_by_permutation(
            X_tr, y_train, perm_repeats, perm_threshold, verbose
        )
        candidate_sets.append(set(selected))
        importance_dict.update(perm_series.to_dict())

    # ========================================
    # 3. ОБЪЕДИНЕНИЕ
    # ========================================

    if not candidate_sets:
        print("  Методы не сработали, оставляем все")
        selected_features = X_tr.columns.tolist()
    else:
        # Объединяем все множества (берем все, что важно хотя бы по одному методу)
        all_candidates = set.union(*candidate_sets)
        selected_features = [f for f in all_candidates if f in X_tr.columns]

        if verbose:
            print(f"\n  После объединения методов: {len(selected_features)} признаков")

    # ========================================
    # 4. УДАЛЕНИЕ КОРРЕЛИРУЮЩИХ
    # ========================================

    if remove_correlated and len(selected_features) > 1:
        # Временно берем только выбранные
        X_temp = X_tr[selected_features]
        X_te_temp = X_te[selected_features]

        X_temp, X_te_temp, dropped = remove_high_correlation(
            X_temp, X_te_temp, importance_dict, corr_threshold, verbose
        )

        selected_features = X_temp.columns.tolist()
        removed_all.extend(dropped)

    # ========================================
    # 5. ОГРАНИЧЕНИЕ КОЛИЧЕСТВА
    # ========================================

    if max_features and len(selected_features) > max_features:
        if importance_dict:
            # Сортируем по важности
            feat_imp = pd.Series(importance_dict)
            feat_imp = feat_imp[feat_imp.index.isin(selected_features)]
            selected_features = feat_imp.nlargest(max_features).index.tolist()
        else:
            # Если нет важности, берем первые N
            selected_features = selected_features[:max_features]

        if verbose:
            print(f"  Ограничено до {max_features} признаков")

    # ========================================
    # 6. ВАЛИДАЦИЯ
    # ========================================

    if validate and quality_loss is not None:
        if len(selected_features) < X_tr.shape[1]:
            ok, loss = validate_feature_set(
                X_tr, y_train, selected_features,
                quality_loss=quality_loss,
                verbose=verbose
            )

            if not ok:
                # Если потеря слишком большая, возвращаем все
                if verbose:
                    print(f"  Возвращаем все признаки")
                selected_features = X_tr.columns.tolist()

    # ========================================
    # 7. ПРИМЕНЕНИЕ
    # ========================================

    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()

    if verbose:
        print(f"\n  ИТОГ:")
        print(f"    Было: {X_train.shape[1]} признаков")
        print(f"    Стало: {X_train_selected.shape[1]} признаков")
        print(f"    Удалено: {X_train.shape[1] - X_train_selected.shape[1]}")

    return X_train_selected, X_test_selected, selected_features


# ============================================
# 7. УПРОЩЕННЫЕ ВЕРСИИ ДЛЯ РАЗНЫХ СЛУЧАЕВ
# ============================================

def select_features_fast(X_train, X_test, y_train, n_features=50):
    """
    Быстрый отбор (только MI + RF Importance)
    Подходит для экспериментов и прототипов
    """
    return select_features(
        X_train, X_test, y_train,
        use_permutation=False,  # самый медленный метод
        use_mi=True,
        use_rf=True,
        remove_correlated=True,
        corr_threshold=0.95,
        max_features=n_features,
        quality_loss=0.05,
        verbose=True
    )


def select_features_accurate(X_train, X_test, y_train, n_features=None):
    """
    Точный отбор (с permutation importance)
    Подходит для финальной модели, когда качество критично
    """
    return select_features(
        X_train, X_test, y_train,
        use_permutation=True,
        use_mi=True,
        use_rf=True,
        remove_correlated=True,
        corr_threshold=0.95,  # удаляем только явные дубли
        max_features=n_features,
        quality_loss=0.0,  # не теряем качество
        verbose=True
    )


def select_features_aggressive(X_train, X_test, y_train, n_features=20):
    """
    Агрессивный отбор (жертвуем качеством ради простоты)
    Подходит для интерпретируемых моделей и презентаций
    """
    return select_features(
        X_train, X_test, y_train,
        use_permutation=False,
        use_mi=True,
        use_rf=True,
        remove_correlated=True,
        corr_threshold=0.8,  # агрессивно удаляем корреляции
        max_features=n_features,
        quality_loss=0.1,  # готовы потерять 10%
        verbose=True
    )
