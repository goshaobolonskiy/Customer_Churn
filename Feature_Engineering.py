import pandas as pd
import numpy as np


# Комбинации созданные вручную
def manual_combinations(X_train, X_test, verbose=False):
    X_tr = X_train.copy()
    X_te = X_test.copy()

    X_tr['is_single'] = ((X_tr['Partner'] == 'No') & (X_tr['Dependents'] == 'No')).astype(int)
    X_te['is_single'] = ((X_te['Partner'] == 'No') & (X_te['Dependents'] == 'No')).astype(int)

    tenure_median = X_tr['tenure'].median()
    X_tr["predict_LTV"] = tenure_median * X_tr['MonthlyCharges']
    X_te["predict_LTV"] = tenure_median * X_te['MonthlyCharges']

    X_tr['LTV_ratio'] = (1 - X_tr['TotalCharges'] / (X_tr['predict_LTV'] + 1e-6)).clip(-1, 1)
    X_te['LTV_ratio'] = (1 - X_te['TotalCharges'] / (X_te['predict_LTV'] + 1e-6)).clip(-1, 1)

    if verbose:
        print(f"manual_combinations: добавлено 3 признака (is_single, predict_LTV, LTV_ratio)")

    return X_tr, X_te


# ------------------------------------------------------------
# 1. ЧИСЛОВЫЕ ВЗАИМОДЕЙСТВИЯ (PolynomialFeatures)
# ------------------------------------------------------------
def poly_features(X_train, X_test, cols=None, degree=2, max_features=20, verbose=False):
    from sklearn.preprocessing import PolynomialFeatures

    train_idx = X_train.index
    test_idx = X_test.index

    if cols is None:
        cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols) > 10:
            cols = cols[:10]

    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)

    X_tr = poly.fit_transform(X_train[cols])
    X_te = poly.transform(X_test[cols])

    names = poly.get_feature_names_out(cols)

    # Правильная фильтрация
    new_names = []
    for name in names:
        # Оставляем только те, где есть пробел (взаимодействия)
        if ' ' in name:
            new_names.append(name.replace(' ', '_'))  # заменяем пробел на _

    # Ограничиваем количество
    if len(new_names) > max_features:
        new_names = new_names[:max_features]
        X_tr = X_tr[:, :max_features]
        X_te = X_te[:, :max_features]
    else:
        # Берем соответствующие колонки из массива
        X_tr = X_tr[:, -len(new_names):]  # последние N колонок
        X_te = X_te[:, -len(new_names):]

    if verbose:
        print(f"  poly: создано {len(new_names)} взаимодействий")

    return pd.DataFrame(X_tr, columns=new_names, index=train_idx), \
        pd.DataFrame(X_te, columns=new_names, index=test_idx)


# ------------------------------------------------------------
# 2. ГРУППОВЫЕ СТАТИСТИКИ (категория × число)
# ------------------------------------------------------------
def group_stats(X_train, X_test, cat_cols=None, num_cols=None, verbose=False):
    """Средние и медианы по группам"""

    if cat_cols is None:
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns[:3].tolist()
    if num_cols is None:
        num_cols = X_train.select_dtypes(include=[np.number]).columns[:5].tolist()

    X_tr = pd.DataFrame(index=X_train.index)
    X_te = pd.DataFrame(index=X_test.index)

    created_count = 0

    for cat in cat_cols:
        for num in num_cols:
            # Среднее
            means = X_train.groupby(cat)[num].mean()
            X_tr[f'{num}_mean_{cat}'] = X_train[cat].map(means).fillna(X_train[num].mean())
            X_te[f'{num}_mean_{cat}'] = X_test[cat].map(means).fillna(X_train[num].mean())
            created_count += 1

            # Медиана
            medians = X_train.groupby(cat)[num].median()
            X_tr[f'{num}_median_{cat}'] = X_train[cat].map(medians).fillna(X_train[num].median())
            X_te[f'{num}_median_{cat}'] = X_test[cat].map(medians).fillna(X_train[num].median())
            created_count += 1

    if verbose:
        print(f"  groups: создано {created_count} статистик по группам")

    return X_tr, X_te


# ------------------------------------------------------------
# 3. КОМБИНАЦИИ КАТЕГОРИЙ
# ------------------------------------------------------------
def cat_combinations(X_train, X_test, cat_cols=None, max_combos=10, min_freq=0.02, verbose=False):
    """
    Комбинирование категориальных признаков ПЕРЕД кодированием.
    Просто конкатенация строк + группировка редких.
    """

    if cat_cols is None:
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns[:5].tolist()

    from itertools import combinations

    X_tr = pd.DataFrame(index=X_train.index)
    X_te = pd.DataFrame(index=X_test.index)

    pairs = list(combinations(cat_cols, 2))[:max_combos]
    created_count = 0

    for col1, col2 in pairs:
        name = f'{col1}_{col2}'

        # 1. Конкатенация строк
        X_tr[name] = X_train[col1].astype(str) + '_' + X_train[col2].astype(str)
        X_te[name] = X_test[col1].astype(str) + '_' + X_test[col2].astype(str)

        # 2. Группировка редких категорий (частота < min_freq)
        freqs = X_tr[name].value_counts(normalize=True)
        rare = freqs[freqs < min_freq].index
        X_tr[name] = X_tr[name].replace(rare, 'Other')
        X_te[name] = X_te[name].replace(rare, 'Other')

        created_count += 1

    if verbose:
        print(f"  cat_combo: создано {created_count} комбинаций категорий")

    return X_tr, X_te


# ------------------------------------------------------------
# 4. БИНАРНЫЕ ОПЕРАЦИИ (AND/OR/XOR)
# ------------------------------------------------------------
def binary_ops(X_train, X_test, bin_cols=None, verbose=False):
    """Логические операции для бинарных признаков"""

    # СОХРАНЯЕМ ИНДЕКСЫ
    train_idx = X_train.index
    test_idx = X_test.index

    if bin_cols is None:
        bin_cols = []
        for col in X_train.columns:
            # ПРОВЕРЯЕМ ЧТО ЭТО ЧИСЛОВАЯ КОЛОНКА
            if pd.api.types.is_numeric_dtype(X_train[col]):
                if X_train[col].dropna().isin([0, 1]).all():
                    bin_cols.append(col)
        bin_cols = bin_cols[:5]

    from itertools import combinations

    X_tr = pd.DataFrame(index=train_idx)
    X_te = pd.DataFrame(index=test_idx)

    created_count = 0

    for c1, c2 in combinations(bin_cols, 2):
        # ПРИВОДИМ К int
        col1_train = X_train[c1].astype(int)
        col2_train = X_train[c2].astype(int)
        col1_test = X_test[c1].astype(int)
        col2_test = X_test[c2].astype(int)

        X_tr[f'{c1}_and_{c2}'] = (col1_train & col2_train).astype(int)
        X_tr[f'{c1}_or_{c2}'] = (col1_train | col2_train).astype(int)
        X_tr[f'{c1}_xor_{c2}'] = (col1_train ^ col2_train).astype(int)
        created_count += 3

        X_te[f'{c1}_and_{c2}'] = (col1_test & col2_test).astype(int)
        X_te[f'{c1}_or_{c2}'] = (col1_test | col2_test).astype(int)
        X_te[f'{c1}_xor_{c2}'] = (col1_test ^ col2_test).astype(int)

    if verbose:
        print(f"  binary: создано {created_count} бинарных операций")

    return X_tr, X_te


# ------------------------------------------------------------
# 5. УМНАЯ ОБЕРТКА (ПРОСТАЯ)
# ------------------------------------------------------------
def add_interactions(X_train, X_test,
                     use_poly=True,
                     use_groups=True,
                     use_cat_combo=True,
                     use_binary=True,
                     verbose=False):
    """
    Добавляет все взаимодействия одной строкой.
    Возвращает копии с новыми признаками.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()

    total_added = 0

    if use_poly:
        p_tr, p_te = poly_features(X_tr, X_te, verbose=verbose)
        X_tr = pd.concat([X_tr, p_tr], axis=1)
        X_te = pd.concat([X_te, p_te], axis=1)
        total_added += p_tr.shape[1]

    if use_groups:
        g_tr, g_te = group_stats(X_tr, X_te, verbose=verbose)
        X_tr = pd.concat([X_tr, g_tr], axis=1)
        X_te = pd.concat([X_te, g_te], axis=1)
        total_added += g_tr.shape[1]

    if use_cat_combo:
        c_tr, c_te = cat_combinations(X_tr, X_te, verbose=verbose)
        X_tr = pd.concat([X_tr, c_tr], axis=1)
        X_te = pd.concat([X_te, c_te], axis=1)
        total_added += c_tr.shape[1]

    if use_binary:
        b_tr, b_te = binary_ops(X_tr, X_te, verbose=verbose)
        X_tr = pd.concat([X_tr, b_tr], axis=1)
        X_te = pd.concat([X_te, b_te], axis=1)
        total_added += b_tr.shape[1]

    # Выравниваем столбцы
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    if verbose:
        print(f"\n✅ add_interactions: всего добавлено {total_added} новых признаков")
        print(f"   Итоговый размер: {X_tr.shape}")

    return X_tr, X_te
