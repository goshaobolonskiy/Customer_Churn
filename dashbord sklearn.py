import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Данные
df = pd.read_excel('ДАШБОРД ЖИЗНИ.xlsx', sheet_name=None)
df_time = df['Время']
df_time.columns = ['Дата', 'Деятельность', 'Часы', 'Примечание']

# ОБРАБОТКА df_time
df_time['Дата'] = pd.to_datetime(df_time['Дата']).dt.date  # Преобразуем в date
df_time['Часы'] = df_time['Часы'].astype(str).str.replace(',', '.')
df_time['Часы'] = pd.to_numeric(df_time['Часы'], errors='coerce')

df_energy = df['Энергия']
df_energy = df_energy.iloc[:, :7]

# ОБРАБОТКА df_energy - ВАЖНО: тоже преобразуем в date
df_energy['Дата'] = pd.to_datetime(df_energy['Дата']).dt.date  # Тут тоже!
df_energy['Оценка дня (1-10)'] = df_energy['Оценка дня (1-10)'].astype(str).str.replace(',', '.')
df_energy["Оценка дня (1-10)"] = pd.to_numeric(df_energy["Оценка дня (1-10)"], errors='coerce')

# формирование Series с индексами даты и оценками дня, плюс удаление Nan начений
rating_df = df_energy[['Дата', 'Оценка дня (1-10)']].dropna()
rating_df["Оценка дня (1-10)"] = pd.to_numeric(rating_df["Оценка дня (1-10)"], errors='coerce')

# Создание pivot (сводной таблицы) по столбцу деятельность со значениями часы, а индексы являются датами
pivot_df = df_time.pivot_table(
    index='Дата',
    columns='Деятельность',
    values='Часы',
    aggfunc='sum',
    fill_value=0
).reset_index()
pivot_df = pd.merge(pivot_df, rating_df, on="Дата", how='inner')
# print(pivot_df.columns)

import numpy as np


x = pivot_df[['Дорога', 'Игры', 'Отдых', 'Прочее', 'Работа', 'Скроллинг','Сон', 'Спорт', 'Учёба', 'Учёба ML']].values
y = pivot_df["Оценка дня (1-10)"].values


# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f'MAE: {mae}\nMSE: {mse}')


print("Предсказания:", predictions)
print("Реальные значения:", y_test)


# Сравнение различных моделей
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score
#
# models = {
#     'Linear Regression': LinearRegression(),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Decision Tree': DecisionTreeRegressor(max_depth=3, random_state=42),
#     'KNN': KNeighborsRegressor(n_neighbors=3),
#     'SVR': SVR(kernel='linear')
# }
#
# results = []
#
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#
#     mae = mean_absolute_error(y_test, predictions)
#     mse = mean_squared_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#
#     results.append({
#         'Модель': name,
#         'MAE': mae,
#         'MSE': mse,
#         'R²': r2
#     })
#
# results_df = pd.DataFrame(results).sort_values('MAE')
# print("\nСРАВНЕНИЕ МОДЕЛЕЙ:")
# print("=" * 50)
# print(results_df.to_string(index=False))


# Предсказание для нового дня
# new_day = pd.DataFrame({
#     'Дорога': [1.5],
#     'Игры': [8.0],
#     'Отдых': [0.0],
#     'Прочее': [0.5],
#     'Работа': [0.0],
#     'Скроллинг': [0.5],
#     'Сон': [8.0],
#     'Спорт': [0.0],
#     'Учёба': [0.0],
#     'Учёба ML': [0.0]
# })
#
# predicted_score = model.predict(new_day.values)
# print(f"\nПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:")
# print("=" * 50)
# print(f"Для дня с параметрами:")
# for col, val in new_day.iloc[0].items():
#     print(f"  {col}: {val} ч")
# print(f"Предсказанная оценка дня: {predicted_score[0]:.1f}")