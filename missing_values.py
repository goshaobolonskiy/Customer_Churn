import pandas as pd
import numpy as np


# Анализ пропусков
def analyse_missing_values(df):
    print(pd.DataFrame({
        'Тип данных': df.dtypes,
        'Всего значений': len(df),
        'Непустых': df.count(),
        'Пустых': df.isnull().sum(),
        '% пропусков': (df.isnull().sum() / len(df) * 100).round(2)
    }))


# Обработка nan значений в численных колонках

# Импутация на среднее/медиану/константу + добавление столбца индикатора
def impute_numeric_math(X_train, X_test=None, strategy='median',
                        constant=0, features=None, add_indicator=False):
    """
    Универсальная функция для простой импутации числовых пропусков.

    Parameters:
    -----------
    X_train, X_test : DataFrame
        Тренировочные и тестовые данные
    strategy : str
        'median' - заполнение медианой (устойчиво к выбросам)
        'mean' - заполнение средним значением
        'constant' - заполнение константой
    constant : numeric
        Значение для стратегии 'constant' (по умолчанию 0)
    features : list или None
        Список признаков для обработки. Если None - обрабатываются все числовые.
    add_indicator : bool
        Добавлять ли индикаторы пропусков (бинарные признаки)

    Returns:
    --------
    X_train_filled, X_test_filled : обработанные DataFrames
    fill_values : dict, значения использованные для заполнения
    indicators_added : list, добавленные индикаторы (если add_indicator=True)
    """

    # Копируем данные
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy() if X_test is not None else None

    # Определяем какие признаки обрабатывать
    if features is None:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Проверяем, что указанные признаки существуют и числовые
        numeric_cols = []
        for col in features:
            if col in X_train.columns:
                # Проверяем тип данных
                if np.issubdtype(X_train[col].dtype, np.number):
                    numeric_cols.append(col)
                else:
                    print(f"Предупреждение: признак '{col}' не числовой, пропускаем")
            else:
                print(f"Предупреждение: признак '{col}' не найден в данных")

    # Находим столбцы с пропусками
    cols_with_missing = [col for col in numeric_cols if X_train[col].isnull().any()]

    if not cols_with_missing:
        print(f"В числовых признаках пропусков не обнаружено")
        return X_train_filled, X_test_filled, {}, []

    print(f"Стратегия заполнения: {strategy}")
    print(f"Найдены пропуски в {len(cols_with_missing)} числовых признаках:")

    # Словарь для хранения значений заполнения
    fill_values = {}

    # 1. Добавляем индикаторы пропусков (если нужно)
    indicators_added = []
    if add_indicator:
        for col in cols_with_missing:
            indicator_name = f'{col}_missing'
            X_train_filled[indicator_name] = X_train[col].isnull().astype(int)
            if X_test_filled is not None and col in X_test_filled.columns:
                X_test_filled[indicator_name] = X_test[col].isnull().astype(int)
            indicators_added.append(indicator_name)
        print(f"Добавлено {len(indicators_added)} индикаторов пропусков")

    # 2. Выполняем импутацию в зависимости от стратегии
    if strategy == 'median':
        # ЗАПОЛНЕНИЕ МЕДИАНОЙ
        for col in cols_with_missing:
            median_val = X_train[col].median()
            fill_values[col] = median_val

            # Заполняем train
            X_train_filled[col] = X_train_filled[col].fillna(median_val)

            # Заполняем test (используем ту же медиану)
            if X_test_filled is not None and col in X_test_filled.columns:
                X_test_filled[col] = X_test_filled[col].fillna(median_val)

        print(f"Заполнено медианой: {len(cols_with_missing)} признаков")

    elif strategy == 'mean':
        # ЗАПОЛНЕНИЕ СРЕДНИМ ЗНАЧЕНИЕМ
        for col in cols_with_missing:
            mean_val = X_train[col].mean()
            fill_values[col] = mean_val

            X_train_filled[col] = X_train_filled[col].fillna(mean_val)
            if X_test_filled is not None and col in X_test_filled.columns:
                X_test_filled[col] = X_test_filled[col].fillna(mean_val)

        print(f"Заполнено средним: {len(cols_with_missing)} признаков")

    elif strategy == 'constant':
        # ЗАПОЛНЕНИЕ КОНСТАНТОЙ
        fill_values['constant_value'] = constant

        for col in cols_with_missing:
            X_train_filled[col] = X_train_filled[col].fillna(constant)
            if X_test_filled is not None and col in X_test_filled.columns:
                X_test_filled[col] = X_test_filled[col].fillna(constant)

        print(f"Заполнено константой {constant}: {len(cols_with_missing)} признаков")

    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}. Используйте 'median', 'mean' или 'constant'")

    # 3. Проверяем результат
    remaining_train = X_train_filled[cols_with_missing].isnull().sum().sum()
    if X_test_filled is not None:
        remaining_test = X_test_filled[cols_with_missing].isnull().sum().sum()
    else:
        remaining_test = 0

    if remaining_train == 0 and remaining_test == 0:
        print(f"✓ Все пропуски успешно заполнены")
    else:
        print(f"⚠ Осталось пропусков: train={remaining_train}, test={remaining_test}")

    # 4. Выводим примеры заполненных значений
    if fill_values and strategy != 'constant':
        print(f"\nПримеры значений для заполнения:")
        for i, (col, val) in enumerate(list(fill_values.items())[:5]):  # Показываем первые 5
            print(f"  {col}: {val:.2f}" if isinstance(val, float) else f"  {col}: {val}")
        if len(fill_values) > 5:
            print(f"  ... и еще {len(fill_values) - 5} признаков")

    return X_train_filled, X_test_filled, fill_values, indicators_added


# ============================================
# KNN ИМПУТАЦИЯ для численных столбцов
# ============================================

def impute_numeric_knn(X_train, X_test=None, features=None, n_neighbors=5):
    """
    KNN импутация - заполнение на основе k ближайших соседей.
    Обучается только на train, применяется к train и test.
    """
    from sklearn.impute import KNNImputer

    if features is None:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in features
                        if col in X_train.columns and
                        np.issubdtype(X_train[col].dtype, np.number)]

    # Создаем датафреймы только с нужными признаками
    X_train_numeric = X_train[numeric_cols].copy()
    X_test_numeric = X_test[numeric_cols].copy() if X_test is not None else None

    # Создаем и обучаем KNNImputer
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train_filled_array = knn_imputer.fit_transform(X_train_numeric)

    # Преобразуем обратно в DataFrame
    X_train_filled = X_train.copy()
    X_train_filled[numeric_cols] = X_train_filled_array

    # Применяем к test
    X_test_filled = None
    if X_test is not None:
        X_test_filled = X_test.copy()
        X_test_filled_array = knn_imputer.transform(X_test_numeric)
        X_test_filled[numeric_cols] = X_test_filled_array

    # print(f"KNN импутация (k={n_neighbors}) применена к {len(numeric_cols)} признакам")
    return X_train_filled, X_test_filled, knn_imputer


# Сравнение способов (knn - самый крутой, выдает реальные значения)

# ============================================
# ИМПУТАЦИЯ ПО ГРУППАМ для численных столбцов
# ============================================

def impute_numeric_by_group(X_train, X_test, group_col, target_cols=None, stat='median'):
    """
    Импутация по группам.

    Parameters:
    -----------
    group_col : str
        Столбец для группировки (например, 'city' или 'department')
    target_cols : list или None
        Столбцы для заполнения (None = все числовые)
    stat : str
        'median' или 'mean' - статистика для заполнения

    Returns:
    --------
    X_train_filled, X_test_filled : DataFrames
    """

    # 1. Определяем целевые столбцы
    if target_cols is None:
        # Все числовые столбцы с пропусками
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        target_cols = [col for col in numeric_cols if X_train[col].isnull().any()]

    # 2. Копируем данные
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()

    # 3. Для каждого целевого столбца
    for col in target_cols:
        if col not in X_train.columns:
            continue

        # Вычисляем статистику по группам НА TRAIN
        if stat == 'median':
            group_stats = X_train.groupby(group_col)[col].median()
        else:  # mean
            group_stats = X_train.groupby(group_col)[col].mean()

        # Общая статистика (если группа не найдена)
        overall_stat = X_train[col].median() if stat == 'median' else X_train[col].mean()

        # Заполняем TRAIN
        mask_train = X_train[col].isnull()
        if mask_train.any():
            # Для каждой строки с пропуском берем значение по группе
            X_train_filled.loc[mask_train, col] = X_train.loc[mask_train, group_col].map(group_stats)
            # Если группа не найдена, заполняем общей статистикой
            X_train_filled[col] = X_train_filled[col].fillna(overall_stat)

        # Заполняем TEST (теми же статистиками!)
        if col in X_test.columns and X_test[col].isnull().any():
            mask_test = X_test[col].isnull()
            X_test_filled.loc[mask_test, col] = X_test.loc[mask_test, group_col].map(group_stats)
            X_test_filled[col] = X_test_filled[col].fillna(overall_stat)

    return X_train_filled, X_test_filled


# Обработка nan значений в категориальных столбцах

# Заполнение модой (самым частым значением)
def impute_categorical_mode_split(X_train, X_test=None, features=None, add_indicator=False):
    """
    Заполнение категориальных пропусков самой частой категорией (модой).
    Мода вычисляется на train, применяется к train и test.
    """

    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy() if X_test is not None else None

    # Определяем категориальные признаки
    if features is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
    else:
        categorical_cols = [col for col in features
                            if col in X_train.columns]

    # Находим столбцы с пропусками
    cols_with_missing = [col for col in categorical_cols
                         if X_train[col].isnull().any()]

    if not cols_with_missing:
        print("В категориальных признаках пропусков не обнаружено")
        return X_train_filled, X_test_filled, {}, []

    print(f"Заполнение модой: {len(cols_with_missing)} признаков")

    # Добавляем индикаторы
    indicators = []
    if add_indicator:
        for col in cols_with_missing:
            indicator_name = f'{col}_missing'
            X_train_filled[indicator_name] = X_train[col].isnull().astype(int)
            if X_test_filled is not None and col in X_test.columns:
                X_test_filled[indicator_name] = X_test[col].isnull().astype(int)
            indicators.append(indicator_name)
        print(f"Добавлено индикаторов: {len(indicators)}")

    # Заполняем пропуски модой
    fill_values = {}

    for col in cols_with_missing:
        # Находим моду (самую частую категорию)
        mode_series = X_train[col].mode()

        if not mode_series.empty:
            mode_val = mode_series[0]
        else:
            # Если все значения пропущены, используем 'Unknown'
            mode_val = 'Unknown'

        fill_values[col] = mode_val

        # Заполняем train
        X_train_filled[col] = X_train_filled[col].fillna(mode_val)

        # Заполняем test (той же модой!)
        if X_test_filled is not None and col in X_test_filled.columns:
            X_test_filled[col] = X_test_filled[col].fillna(mode_val)

    return X_train_filled, X_test_filled, fill_values, indicators


# Заполнение константой
def impute_categorical_constant_split(X_train, X_test=None, features=None,
                                      constant='Unknown', add_indicator=False):
    """
    Заполнение категориальных пропусков константой.
    Полезно, когда пропуски несут информацию (MNAR).
    """

    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy() if X_test is not None else None

    if features is None:
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
    else:
        categorical_cols = [col for col in features
                            if col in X_train.columns]

    cols_with_missing = [col for col in categorical_cols
                         if X_train[col].isnull().any()]

    if not cols_with_missing:
        print("В категориальных признаках пропусков не обнаружено")
        return X_train_filled, X_test_filled, [], {}

    print(f"Заполнение константой '{constant}': {len(cols_with_missing)} признаков")

    # Добавляем индикаторы
    indicators = []
    if add_indicator:
        for col in cols_with_missing:
            indicator_name = f'{col}_missing'
            X_train_filled[indicator_name] = X_train[col].isnull().astype(int)
            if X_test_filled is not None and col in X_test.columns:
                X_test_filled[indicator_name] = X_test[col].isnull().astype(int)
            indicators.append(indicator_name)

    # Заполняем константой
    for col in cols_with_missing:
        X_train_filled[col] = X_train_filled[col].fillna(constant)
        if X_test_filled is not None and col in X_test_filled.columns:
            X_test_filled[col] = X_test_filled[col].fillna(constant)

    return X_train_filled, X_test_filled, indicators, {'constant': constant}


# Заполнение модой по группам
def impute_categorical_by_group_split(X_train, X_test, group_col,
                                      target_cols=None, add_indicator=False):
    """
    Заполнение категориальных пропусков модой по группам.
    Пример: заполняем пропущенный город по стране.
    """

    if group_col not in X_train.columns:
        print(f"Ошибка: колонка {group_col} не найдена")
        return X_train.copy(), X_test.copy(), [], {}

    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()

    # Определяем целевые колонки
    if target_cols is None:
        # Все категориальные колонки, кроме группировочной
        categorical_cols = X_train.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
        target_cols = [col for col in categorical_cols
                       if col != group_col and X_train[col].isnull().any()]
    else:
        # Проверяем существование колонок
        target_cols = [col for col in target_cols
                       if col in X_train.columns and col != group_col]

    if not target_cols:
        print(f"Нет категориальных признаков с пропусками для заполнения по группам")
        return X_train_filled, X_test_filled, [], {}

    print(f"Заполнение по группам '{group_col}': {len(target_cols)} признаков")

    # Добавляем индикаторы
    indicators = []
    if add_indicator:
        for col in target_cols:
            indicator_name = f'{col}_missing'
            X_train_filled[indicator_name] = X_train[col].isnull().astype(int)
            if col in X_test.columns:
                X_test_filled[indicator_name] = X_test[col].isnull().astype(int)
            indicators.append(indicator_name)

    # Для каждого целевого столбца вычисляем моду по группам
    group_modes = {}

    for col in target_cols:
        if X_train[col].isnull().any():
            # Вычисляем моду по группам на train
            modes_by_group = X_train.groupby(group_col)[col].apply(
                lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
            )
            group_modes[col] = modes_by_group

            # Общая мода (на случай новых групп в test)
            overall_mode = X_train[col].mode()
            overall_mode_val = overall_mode[0] if not overall_mode.empty else 'Unknown'

            # Заполняем train
            mask_train = X_train[col].isnull()
            X_train_filled.loc[mask_train, col] = X_train.loc[mask_train, group_col].map(modes_by_group)
            X_train_filled[col] = X_train_filled[col].fillna(overall_mode_val)

            # Заполняем test
            if col in X_test.columns and X_test[col].isnull().any():
                mask_test = X_test[col].isnull()
                X_test_filled.loc[mask_test, col] = X_test.loc[mask_test, group_col].map(modes_by_group)
                X_test_filled[col] = X_test_filled[col].fillna(overall_mode_val)

    return X_train_filled, X_test_filled, indicators, group_modes


def handle_categorical_missing_split(X_train, X_test,
                                     strategy='mode',
                                     constant='Unknown',
                                     group_by=None,
                                     add_indicator=False,
                                     features=None):
    """
    Универсальная функция для обработки категориальных пропусков.

    Parameters:
    -----------
    strategy : str
        'mode' - заполнение модой
        'constant' - заполнение константой
        'group' - заполнение по группам (требует group_by)
    constant : str
        Значение для стратегии 'constant'
    group_by : str
        Колонка для группировки (для стратегии 'group')
    add_indicator : bool
        Добавлять индикаторы пропусков
    features : list
        Список признаков для обработки (None = все категориальные)
    """

    # Применяем выбранную стратегию
    if strategy == 'mode':
        X_train_filled, X_test_filled, fill_values, indicators = impute_categorical_mode_split(
            X_train, X_test, features=features, add_indicator=add_indicator
        )

    elif strategy == 'constant':
        X_train_filled, X_test_filled, indicators, info = impute_categorical_constant_split(
            X_train, X_test, features=features,
            constant=constant, add_indicator=add_indicator
        )
        fill_values = info

    elif strategy == 'group':
        if group_by is None:
            print("Ошибка: для стратегии 'group' необходимо указать group_by")
            return X_train, X_test, {}, []

        X_train_filled, X_test_filled, indicators, fill_values = impute_categorical_by_group_split(
            X_train, X_test, group_by,
            target_cols=features, add_indicator=add_indicator
        )

    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}. Используйте 'mode', 'constant' или 'group'")

    # Проверяем результат
    categorical_cols = X_train_filled.select_dtypes(
        include=['object', 'category', 'string']
    ).columns

    remaining_missing = X_train_filled[categorical_cols].isnull().sum().sum()

    print(f"\n{'=' * 60}")
    print("РЕЗУЛЬТАТ ОБРАБОТКИ:")
    print(f"Осталось пропусков: {remaining_missing}")
    if remaining_missing == 0:
        print("✓ Все категориальные пропуски заполнены")

    return X_train_filled, X_test_filled, fill_values, indicators
