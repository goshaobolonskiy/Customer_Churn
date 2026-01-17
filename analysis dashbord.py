import pandas as pd
import matplotlib.pyplot as plt

# Импорт xlsx в df
df = pd.read_excel('ДАШБОРД ЖИЗНИ.xlsx', sheet_name=None)
df_time = df['Время']
df_time.columns = ['Дата', 'Деятельность', 'Часы', 'Примечание']

df_energy = df['Энергия']
df_energy = df_energy.iloc[:, :7]

# Преобразование даты в столбце в формат даты в python, так же отрезаем время
df_time['Дата'] = pd.to_datetime(df_time['Дата']).dt.date
# Замена ',' на '.' для правильной работы числовых значений
df_time['Часы'] = df_time['Часы'].astype(str).str.replace(',', '.')
# Перевод всех данных в столбце Часы в численные значения
df_time["Часы"] = pd.to_numeric(df_time["Часы"], errors='coerce')

# те же самые преобразования, как выше для df с энергией
df_energy['Дата'] = pd.to_datetime(df_energy['Дата']).dt.date
df_energy['Оценка дня (1-10)'] = df_energy['Оценка дня (1-10)'].astype(str).str.replace(',', '.')
df_energy["Оценка дня (1-10)"] = pd.to_numeric(df_energy["Оценка дня (1-10)"], errors='coerce')

# формирование Series с индексами даты и оценками дня, плюс удаление Nan начений
rating_df = df_energy[['Дата', 'Оценка дня (1-10)']].dropna()
rating_df["Оценка дня (1-10)"] = pd.to_numeric(rating_df["Оценка дня (1-10)"], errors='coerce')

# print(df_energy['Социальность'].map({'Да': 1, 'Нет': 0}).dropna().corr(df_energy['Качество вечера']))

# Создание pivot (сводной таблицы) по столбцу деятельность со значениями часы, а индексы являются датами
pivot_df = df_time.pivot_table(
    index='Дата',
    columns='Деятельность',
    values='Часы',
    aggfunc='sum',
    fill_value=0
).reset_index()
# Удаляем строки, где есть NaN в ключевых колонках


# Объединение таблицы оценок с pivot таблицей деятельностей по столбцу Дата
pivot_df = pd.merge(pivot_df, rating_df, on="Дата")
# print(pivot_df)


# Проведение корреляции деятельности с оценкой дня
actions = ['Дорога', 'Игры', 'Отдых', 'Работа', 'Скроллинг', 'Сон', 'Спорт', 'Учёба', 'Учёба ML']
cor = []
for i in actions:
    # print(f'Корреляция {i} и оценки дня: {pivot_df[i].corr(pivot_df["Оценка дня (1-10)"])}')
    cor.append(pivot_df[i].corr(pivot_df["Оценка дня (1-10)"]))


# Визуализация
plt.title('Корреляция деятельности с оценкой')
plt.xlabel('Деятельность')
plt.ylabel('Корреляция')

plt.grid(True, alpha=0.3)
bars = plt.bar(actions, cor, color='skyblue', width=0.8, edgecolor='black', alpha=0.7)
plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

plt.tight_layout()
plt.show()



# Другой способ создать совместную таблицу, по которой можно искать корреляции (более объемный и долгий)

# sleep_df = df_time[df_time['Деятельность']=="Сон"].groupby('Дата')['Часы'].sum()

# game_df = df_time[df_time['Деятельность']=='Игры'].groupby('Дата')['Часы'].sum()

# sport_df = df_time[df_time['Деятельность']=='Спорт'].groupby('Дата')['Часы'].sum()

# work_df = df_time[df_time["Деятельность"]=='Работа'].groupby('Дата')['Часы'].sum()

# shared_df = pd.DataFrame({
#     'Сон' : sleep_df,
#     'Игры' : game_df,
#     'Спорт' : sport_df,
#     'Работа' : work_df,
# }).reset_index()

# shared_df = pd.merge(shared_df, rating_df, on="Дата", how='inner')
# print(shared_df)

# print(f'Корреляция сна и оценки дня: {shared_df["Сон"].corr(shared_df["Оценка дня (1-10)"])}')
# print(f"Корреляция игр и оценки дня: {shared_df["Игры"].corr(shared_df["Оценка дня (1-10)"])}")
# print(f"Корреляция спорта и оценки дня: {shared_df["Спорт"].corr(shared_df["Оценка дня (1-10)"])}")
# print(f"Корреляция работы и оценки дня: {shared_df["Работа"].corr(shared_df["Оценка дня (1-10)"])}")
