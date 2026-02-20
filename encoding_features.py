import pandas as pd
import numpy as np


# =============================================
# Кодирование категориальных признаков
# =============================================

def label_encode_split(X_train, X_test, columns=None, return_mappings=False, verbose=True):
    """
    Label Encoding с разделением train/test.

    Parameters:
    -----------
    X_train, X_test : DataFrame
    columns : list или None
        Столбцы для кодирования (None = все категориальные)
    return_mappings : bool
        Возвращать словари сопоставления
    verbose : bool
        Печатать ли информацию

    Returns:
    --------
    X_train_encoded, X_test_encoded : закодированные DataFrames
    mappings : dict (только если return_mappings=True)
        Словарь {столбец: {категория: код}}
    """

    from sklearn.preprocessing import LabelEncoder

    # Определяем столбцы для кодирования
    if columns is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in X_train.columns]

    if not categorical_cols:
        if verbose:
            print("Нет категориальных признаков для кодирования")
        return X_train.copy(), X_test.copy(), {} if return_mappings else (X_train.copy(), X_test.copy())

    # Копируем данные
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    mappings = {}

    # Применяем Label Encoding к каждому столбцу
    for col in categorical_cols:
        # Создаем и обучаем LabelEncoder на train
        le = LabelEncoder()

        # Обучаем на уникальных значениях из train
        X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))

        # Для test: преобразуем только те категории, которые были в train
        try:
            X_test_encoded[col] = le.transform(X_test[col].astype(str))
        except ValueError:
            # Если в test есть новые категории, присваиваем им код -1
            unique_train = set(le.classes_)
            mask_new = ~X_test[col].astype(str).isin(unique_train)
            X_test_encoded.loc[~mask_new, col] = le.transform(X_test.loc[~mask_new, col].astype(str))
            X_test_encoded.loc[mask_new, col] = -1  # Новые категории → -1

        # Сохраняем mapping
        if return_mappings:
            mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    if verbose:
        print(f"Label Encoding применен к {len(categorical_cols)} признакам")

    if return_mappings:
        return X_train_encoded, X_test_encoded, mappings
    return X_train_encoded, X_test_encoded


def one_hot_encode_split(X_train, X_test, columns=None, drop_first=True,
                         sparse=False, handle_unknown='ignore', verbose=True):
    """
    One-Hot Encoding с разделением train/test.

    Parameters:
    -----------
    drop_first : bool
        Удалять ли первый столбец (избегать dummy variable trap)
    sparse : bool
        Возвращать разреженную матрицу
    handle_unknown : str
        'ignore' или 'error' при встрече новых категорий в test
    verbose : bool
        Печатать ли информацию
    """

    from sklearn.preprocessing import OneHotEncoder

    # Определяем столбцы для кодирования
    if columns is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in X_train.columns]

    if not categorical_cols:
        if verbose:
            print("Нет категориальных признаков для One-Hot кодирования")
        return X_train.copy(), X_test.copy(), None

    # Разделяем данные на категориальные и остальные
    X_train_cat = X_train[categorical_cols]
    X_test_cat = X_test[categorical_cols] if X_test is not None else None

    # Числовые и другие признаки
    other_cols = [col for col in X_train.columns if col not in categorical_cols]
    X_train_other = X_train[other_cols] if other_cols else None
    X_test_other = X_test[other_cols] if X_test is not None and other_cols else None

    # Создаем и обучаем OneHotEncoder
    ohe = OneHotEncoder(
        drop='first' if drop_first else None,
        sparse_output=sparse,
        handle_unknown=handle_unknown
    )

    # Обучаем на train
    X_train_ohe = ohe.fit_transform(X_train_cat)

    # Преобразуем в DataFrame (если не sparse)
    if not sparse:
        # Получаем имена признаков
        feature_names = ohe.get_feature_names_out(categorical_cols)
        X_train_ohe_df = pd.DataFrame(
            X_train_ohe,
            columns=feature_names,
            index=X_train.index
        )
    else:
        X_train_ohe_df = X_train_ohe

    # Применяем к test
    if X_test_cat is not None:
        X_test_ohe = ohe.transform(X_test_cat)

        if not sparse:
            X_test_ohe_df = pd.DataFrame(
                X_test_ohe,
                columns=feature_names,
                index=X_test.index
            )
        else:
            X_test_ohe_df = X_test_ohe
    else:
        X_test_ohe_df = None

    # Объединяем с остальными признаками
    if not sparse and X_train_other is not None:
        X_train_final = pd.concat([X_train_other, X_train_ohe_df], axis=1)
        if X_test_other is not None and X_test_ohe_df is not None:
            X_test_final = pd.concat([X_test_other, X_test_ohe_df], axis=1)
        else:
            X_test_final = X_test_ohe_df
    else:
        X_train_final = X_train_ohe_df
        X_test_final = X_test_ohe_df

    if verbose:
        print(f"One-Hot Encoding применен к {len(categorical_cols)} признакам")
        print(f"Создано {len(feature_names)} новых признаков")

    return X_train_final, X_test_final, ohe


def frequency_encode_split(X_train, X_test, columns=None, normalize=True,
                           unseen_value='mean', verbose=True):
    """
    Частотное кодирование: категория → частота её появления.

    Parameters:
    -----------
    normalize : bool
        Использовать доли (True) или абсолютные количества (False)
    unseen_value : str
        Как обрабатывать новые категории в test:
        'mean' - средняя частота из train
        'zero' - 0
        'min' - минимальная частота из train
        'max' - максимальная частота из train
    verbose : bool
        Печатать ли информацию
    """

    # Определяем столбцы для кодирования
    if columns is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in X_train.columns]

    if not categorical_cols:
        if verbose:
            print("Нет категориальных признаков для частотного кодирования")
        return X_train.copy(), X_test.copy(), {}

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    freq_maps = {}

    for col in categorical_cols:
        # Вычисляем частоты на train
        if normalize:
            freq = X_train[col].value_counts(normalize=True)
        else:
            freq = X_train[col].value_counts()

        freq_maps[col] = freq

        # Применяем к train
        X_train_encoded[col] = X_train[col].map(freq)

        # Применяем к test
        X_test_encoded[col] = X_test[col].map(freq)

        # Обрабатываем новые категории в test
        if X_test_encoded[col].isnull().any():
            if unseen_value == 'mean':
                fill_val = freq.mean()
            elif unseen_value == 'zero':
                fill_val = 0
            elif unseen_value == 'min':
                fill_val = freq.min()
            elif unseen_value == 'max':
                fill_val = freq.max()
            else:
                fill_val = freq.mean()

            X_test_encoded[col] = X_test_encoded[col].fillna(fill_val)

        # Также заполняем пропуски в train (если были)
        X_train_encoded[col] = X_train_encoded[col].fillna(freq.mean() if normalize else 0)

    if verbose:
        print(f"Частотное кодирование применено к {len(categorical_cols)} признакам")

    return X_train_encoded, X_test_encoded, freq_maps


def target_encode_split(X_train, y_train, X_test, columns=None,
                        smoothing=1.0, min_samples_leaf=1, noise_level=0, verbose=True):
    """
    Target Encoding (Mean Encoding) со сглаживанием.

    Parameters:
    -----------
    smoothing : float
        Коэффициент сглаживания (выше = больше сглаживание к среднему)
    min_samples_leaf : int
        Минимальное количество образцов в категории для учета
    noise_level : float
        Уровень шума для добавления (для предотвращения переобучения)
    verbose : bool
        Печатать ли информацию
    """

    # Проверяем, что y_train имеет правильный формат
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    # Определяем столбцы для кодирования
    if columns is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in X_train.columns]

    if not categorical_cols:
        if verbose:
            print("Нет категориальных признаков для target encoding")
        return X_train.copy(), X_test.copy(), {}

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoding_maps = {}

    # Глобальное среднее
    global_mean = y_train.mean()

    for col in categorical_cols:
        # Создаем DataFrame для удобства
        temp_df = pd.DataFrame({col: X_train[col], 'target': y_train})

        # Вычисляем среднее по категориям
        category_means = temp_df.groupby(col)['target'].mean()

        # Вычисляем размеры категорий
        category_counts = temp_df.groupby(col)['target'].count()

        # Применяем сглаживание
        smoothed_means = (category_means * category_counts + global_mean * smoothing) / (category_counts + smoothing)

        # Игнорируем категории с малым количеством образцов
        smoothed_means[category_counts < min_samples_leaf] = global_mean

        # Сохраняем mapping
        encoding_maps[col] = smoothed_means.to_dict()

        # Применяем к train
        X_train_encoded[col] = X_train[col].map(smoothed_means)

        # Добавляем шум (опционально)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(X_train_encoded))
            X_train_encoded[col] += noise

        # Применяем к test (новые категории → global_mean)
        X_test_encoded[col] = X_test[col].map(smoothed_means)
        X_test_encoded[col] = X_test_encoded[col].fillna(global_mean)

    if verbose:
        print(f"Target Encoding применен к {len(categorical_cols)} признакам")
        print(f"Сглаживание: {smoothing}, Минимальный размер категории: {min_samples_leaf}")

    return X_train_encoded, X_test_encoded, encoding_maps


def ordinal_encode_split(X_train, X_test, columns_mapping=None,
                         handle_unknown='use_encoded_value', unknown_value=-1, verbose=True):
    """
    Порядковое кодирование для признаков с естественным порядком.

    Parameters:
    -----------
    columns_mapping : dict
        {столбец: [категории в порядке возрастания]}
    handle_unknown : str
        Как обрабатывать новые категории
    unknown_value : int
        Значение для новых категорий
    verbose : bool
        Печатать ли информацию
    """

    from sklearn.preprocessing import OrdinalEncoder

    # Если mapping не предоставлен, используем сортировку по алфавиту
    if columns_mapping is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()

        columns_mapping = {}
        for col in categorical_cols:
            # Сортируем категории по алфавиту (не всегда правильно!)
            categories = sorted(X_train[col].dropna().unique())
            columns_mapping[col] = categories

    # Определяем столбцы для кодирования
    cols_to_encode = list(columns_mapping.keys())
    cols_to_encode = [col for col in cols_to_encode if col in X_train.columns]

    if not cols_to_encode:
        if verbose:
            print("Нет признаков для порядкового кодирования")
        return X_train.copy(), X_test.copy(), None

    # Создаем и обучаем OrdinalEncoder
    ordinal_encoder = OrdinalEncoder(
        categories=[columns_mapping[col] for col in cols_to_encode],
        handle_unknown=handle_unknown,
        unknown_value=unknown_value
    )

    # Обучаем на train
    X_train_encoded_array = ordinal_encoder.fit_transform(X_train[cols_to_encode])

    # Применяем к test
    X_test_encoded_array = ordinal_encoder.transform(X_test[cols_to_encode])

    # Создаем новые DataFrames
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    # Заменяем столбцы
    for i, col in enumerate(cols_to_encode):
        X_train_encoded[col] = X_train_encoded_array[:, i]
        X_test_encoded[col] = X_test_encoded_array[:, i]

    if verbose:
        print(f"Порядковое кодирование применено к {len(cols_to_encode)} признакам")

    return X_train_encoded, X_test_encoded, ordinal_encoder


def binary_encode_split(X_train, X_test, columns=None, verbose=True):
    """
    Binary Encoding: категория → число → двоичный код → разделение битов.
    """
    from sklearn.preprocessing import LabelEncoder

    # Определяем столбцы для кодирования
    if columns is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in X_train.columns]

    if not categorical_cols:
        if verbose:
            print("Нет категориальных признаков для binary encoding")
        return X_train.copy(), X_test.copy(), {}

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoding_info = {}

    for col in categorical_cols:
        # Сначала делаем Label Encoding
        le = LabelEncoder()
        train_encoded = le.fit_transform(X_train[col].astype(str))

        # Преобразуем в двоичный формат
        max_val = train_encoded.max()
        num_bits = int(np.ceil(np.log2(max_val + 1)))

        # Создаем новые столбцы для каждого бита
        for bit in range(num_bits):
            new_col_name = f"{col}_bit{bit}"
            # Для train: извлекаем бит
            X_train_encoded[new_col_name] = (train_encoded >> bit) & 1

            # Для test: сначала label encoding, потом биты
            try:
                test_encoded = le.transform(X_test[col].astype(str))
                X_test_encoded[new_col_name] = (test_encoded >> bit) & 1
            except ValueError:
                # Новые категории в test → все биты 0
                X_test_encoded[new_col_name] = 0

        # Удаляем исходный столбец
        X_train_encoded = X_train_encoded.drop(columns=[col])
        X_test_encoded = X_test_encoded.drop(columns=[col])

        encoding_info[col] = {
            'label_encoder': le,
            'num_bits': num_bits,
            'max_value': max_val
        }

    if verbose:
        print(f"Binary Encoding применен к {len(categorical_cols)} признакам")
        print(f"Создано {sum(info['num_bits'] for info in encoding_info.values())} новых признаков")

    return X_train_encoded, X_test_encoded, encoding_info


def encode_categorical_split(X_train, X_test, y_train=None,
                             method='onehot', verbose=True, **kwargs):
    """
    Универсальная функция для кодирования категориальных признаков.

    Parameters:
    -----------
    method : str
        'label' - Label Encoding
        'onehot' - One-Hot Encoding
        'frequency' - Frequency Encoding
        'target' - Target Encoding (требует y_train)
        'ordinal' - Ordinal Encoding
        'binary' - Binary Encoding
    verbose : bool
        Печатать ли информацию
    **kwargs : дополнительные параметры для конкретного метода
    """

    if verbose:
        print("=" * 60)
        print(f"КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ (метод: {method})")
        print("=" * 60)

    # Сначала находим все категориальные признаки
    categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()

    if not categorical_cols:
        if verbose:
            print("Категориальные признаки не найдены")
        return X_train.copy(), X_test.copy(), None

    if verbose:
        print(f"Найдено категориальных признаков: {len(categorical_cols)}")
        for col in categorical_cols:
            unique_vals = X_train[col].nunique()
            print(f"  {col}: {unique_vals} уникальных значений")

    # Выбираем метод кодирования
    if method == 'label':
        result = label_encode_split(
            X_train, X_test,
            columns=categorical_cols,
            return_mappings=kwargs.get('return_mappings', False),
            verbose=verbose
        )

    elif method == 'onehot':
        low_cardinality = [col for col in categorical_cols
                           if X_train[col].nunique() <= kwargs.get('onehot_threshold', 10)]
        high_cardinality = [col for col in categorical_cols
                            if col not in low_cardinality]

        if low_cardinality and verbose:
            print(
                f"One-Hot Encoding для признаков с ≤{kwargs.get('onehot_threshold', 10)} категориями: {low_cardinality}")
        if high_cardinality and verbose:
            print(
                f"Frequency Encoding для признаков с >{kwargs.get('onehot_threshold', 10)} категориями: {high_cardinality}")

        # Обработка onehot
        if low_cardinality:
            X_train_low, X_test_low, ohe = one_hot_encode_split(
                X_train, X_test,
                columns=low_cardinality,
                drop_first=kwargs.get('drop_first', True),
                sparse=kwargs.get('sparse', False),
                verbose=verbose
            )

        # Обработка frequency
        if high_cardinality:
            X_train_high, X_test_high, freq_maps = frequency_encode_split(
                X_train, X_test,
                columns=high_cardinality,
                normalize=kwargs.get('normalize', True),
                verbose=verbose
            )

        # Объединяем результаты
        if low_cardinality and high_cardinality:
            X_train_encoded = pd.concat([
                X_train.drop(columns=categorical_cols),
                X_train_low.drop(columns=[c for c in low_cardinality if c in X_train_low.columns]),
                X_train_high[high_cardinality]
            ], axis=1)

            X_test_encoded = pd.concat([
                X_test.drop(columns=categorical_cols),
                X_test_low.drop(columns=[c for c in low_cardinality if c in X_test_low.columns]),
                X_test_high[high_cardinality]
            ], axis=1)

            result = (X_train_encoded, X_test_encoded, {'ohe': ohe, 'freq_maps': freq_maps})

        elif low_cardinality:
            result = (X_train_low, X_test_low, ohe)

        else:
            result = (X_train_high, X_test_high, freq_maps)

    elif method == 'frequency':
        result = frequency_encode_split(
            X_train, X_test,
            columns=categorical_cols,
            normalize=kwargs.get('normalize', True),
            unseen_value=kwargs.get('unseen_value', 'mean'),
            verbose=verbose
        )

    elif method == 'target':
        if y_train is None:
            raise ValueError("Для Target Encoding необходим y_train")

        result = target_encode_split(
            X_train, y_train, X_test,
            columns=categorical_cols,
            smoothing=kwargs.get('smoothing', 1.0),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            noise_level=kwargs.get('noise_level', 0),
            verbose=verbose
        )

    elif method == 'ordinal':
        result = ordinal_encode_split(
            X_train, X_test,
            columns_mapping=kwargs.get('columns_mapping', None),
            handle_unknown=kwargs.get('handle_unknown', 'use_encoded_value'),
            unknown_value=kwargs.get('unknown_value', -1),
            verbose=verbose
        )

    elif method == 'binary':
        result = binary_encode_split(
            X_train, X_test,
            columns=categorical_cols,
            verbose=verbose
        )

    else:
        raise ValueError(
            f"Неизвестный метод: {method}. Используйте 'label', 'onehot', 'frequency', 'target', 'ordinal', 'binary'")

    # Проверяем размерности
    X_train_encoded, X_test_encoded, encoder = result

    if verbose:
        print(f"\nРазмерности после кодирования:")
        print(f"  Train: {X_train_encoded.shape}")
        print(f"  Test:  {X_test_encoded.shape}")

        # Проверяем наличие NaN
        nan_train = X_train_encoded.isnull().sum().sum()
        nan_test = X_test_encoded.isnull().sum().sum()

        if nan_train > 0 or nan_test > 0:
            print(f"⚠ ВНИМАНИЕ: Остались NaN значения: train={nan_train}, test={nan_test}")

    return result


def encode_target(y_train, y_test=None, return_encoder=False, verbose=True):
    """
    Кодирование целевой переменной (y) для бинарной классификации.

    Parameters:
    -----------
    y_train : Series или array-like
        Целевая переменная для тренировки
    y_test : Series или array-like, optional
        Целевая переменная для теста
    return_encoder : bool
        Вернуть обученный LabelEncoder для обратного преобразования
    verbose : bool
        Печатать ли информацию

    Returns:
    --------
    y_train_encoded : array
        Закодированный y_train (0/1)
    y_test_encoded : array или None
        Закодированный y_test (если передан)
    encoder : LabelEncoder (если return_encoder=True)
    """

    from sklearn.preprocessing import LabelEncoder

    # Создаем и обучаем encoder на train
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    if verbose:
        print(f"Классы: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Считаем распределение
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        for cls, count in zip(unique, counts):
            original = le.inverse_transform([cls])[0]
            print(f"  {original} ({cls}): {count} ({count / len(y_train) * 100:.1f}%)")

    # Кодируем test если есть
    y_test_encoded = None
    if y_test is not None:
        try:
            y_test_encoded = le.transform(y_test)

            # Проверяем, есть ли в test новые классы
            unique_test = np.unique(y_test)
            unique_train = le.classes_
            new_classes = set(unique_test) - set(unique_train)

            if new_classes and verbose:
                print(f"⚠ ВНИМАНИЕ: В test есть новые классы: {new_classes}")
                print(f"Они будут преобразованы в -1")

                # Обрабатываем новые классы
                y_test_encoded = pd.Series(y_test).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                ).values
        except Exception as e:
            if verbose:
                print(f"Ошибка при кодировании test: {e}")
            y_test_encoded = None

    if return_encoder:
        return y_train_encoded, y_test_encoded, le

    return y_train_encoded, y_test_encoded


def decode_target(y_encoded, encoder):
    """
    Обратное преобразование чисел в исходные метки.

    Parameters:
    -----------
    y_encoded : array
        Закодированные значения (0, 1, ...)
    encoder : LabelEncoder
        Обученный encoder из encode_target

    Returns:
    --------
    array : исходные метки
    """
    return encoder.inverse_transform(y_encoded)
