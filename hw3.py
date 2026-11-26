import re
import pandas as pd

# 1. Считывание датасета из файла train.csv
df = pd.read_csv('train.csv')

# 2. Основная информация о датасете: типы данных, пропуски, средние и др.
print("=== 2. Основная информация о датасете ===")
df.info()
print("\nКоличество пропусков по столбцам:")
print(df.isnull().sum())
print("\nОписательные статистики:")
print(df.describe())

# 3. Процент выживаемости у каждого класса пассажиров (Pclass)
print("\n=== 3. Процент выживаемости по классам (Pclass) ===")
survival_by_class = (df.groupby('Pclass')['Survived'].mean() * 100).round(2)
print(survival_by_class)


# Функция для корректного извлечения личного имени из столбца Name
def extract_first_name(row):
    name = row['Name']
    sex = row['Sex']
    # У замужних женщин (Mrs., Countess., Lady., Mme.) собственное имя указано в скобках
    if sex == 'female' and re.search(r'\b(Mrs|Countess|Lady|Mme)\.', name):
        match = re.search(r'\(([^)]+)\)', name)
        if match:
            inner = match.group(1).replace('"', '').strip()
            return inner.split()[0]
    # В остальных случаях берём первое слово после титула (Mr., Miss., Master. и т.д.)
    after_title = name.split('. ', 1)[1].strip().lstrip('("').strip()
    return after_title.split()[0].strip(')"')


df['FirstName'] = df.apply(extract_first_name, axis=1)

# 4. Самое популярное мужское и самое популярное женское имя на корабле
print("\n=== 4. Самые популярные имена на корабле ===")
for sex, label in [('male', 'Мужское'), ('female', 'Женское')]:
    counts = df[df['Sex'] == sex]['FirstName'].value_counts()
    max_count = counts.max()
    top_names = counts[counts == max_count].index.tolist()
    print(f"{label}: {', '.join(top_names)} ({max_count})")

# 5. Самое популярное мужское и женское имя в каждом классе
print("\n=== 5. Самые популярные имена в каждом классе ===")
for pclass in sorted(df['Pclass'].unique()):
    print(f"Класс {pclass}:")
    for sex, label in [('male', 'Мужское'), ('female', 'Женское')]:
        counts = df[(df['Pclass'] == pclass) & (df['Sex'] == sex)]['FirstName'].value_counts()
        max_count = counts.max()
        top_names = counts[counts == max_count].index.tolist()
        print(f"  {label}: {', '.join(top_names)} ({max_count})")

# 6. Часть таблицы с пассажирами, возраст которых больше 44 лет
print("\n=== 6. Пассажиры старше 44 лет ===")
age_gt_44 = df[df['Age'] > 44].drop(columns=['FirstName'])
print(f"Всего пассажиров: {len(age_gt_44)}")
print(age_gt_44.head())

# 7. Часть таблицы с пассажирами мужского пола младше 44 лет
print("\n=== 7. Мужчины младше 44 лет ===")
male_lt_44 = df[(df['Age'] < 44) & (df['Sex'] == 'male')].drop(columns=['FirstName'])
print(f"Всего пассажиров: {len(male_lt_44)}")
print(male_lt_44.head())

# 8. Количества n-местных кабин (в которых было 2, 3, 4, ... человека)
print("\n=== 8. Количества n-местных кабин ===")
cabin_counts = df['Cabin'].value_counts()
n_person_cabins = cabin_counts[cabin_counts >= 2].value_counts().sort_index()
for n_people, count in n_person_cabins.items():
    print(f"{n_people}-местных кабин: {count}")