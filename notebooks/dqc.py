# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
df_categories = pd.read_csv("D:\ML_innovise\item_categories.csv")
df_items = pd.read_csv("D:\ML_innovise\items.csv")
df_sales_train = pd.read_csv("D:\ML_innovise\sales_train.csv")
df_submission = pd.read_csv("D:\ML_innovise\sample_submission.csv")
df_shops = pd.read_csv("D:\ML_innovise\shops.csv")
df_testt = pd.read_csv(r"D:\ML_innovise\test.csv")


# %%
df = df_items.merge(
    df_categories, on="item_category_id", how="left", validate="many_to_one"
)


# %%
df_1 = df_sales_train.merge(df, on="item_id", how="left", validate="many_to_one")


# %%
# df_1 - датасет, который содержит вообще всю информацию про продаваемость товаров за весь период

df_1 = df_1.merge(df_shops, on="shop_id", how="left", validate="many_to_one")
df_1.head()

# %%
df_1.isnull().sum()  # убедимся, что нет пропусков

# %%
df_1.describe()

# %%
df_1.info()

# %%
df_2 = df_testt.merge(df, on="item_id", how="left", validate="many_to_one")


# %%
# df_2 - датасет, который содержит всю информацию про товары, продаваемость которых мы будем предсказывать
df_2 = df_2.merge(df_shops, on="shop_id", how="left", validate="many_to_one")
df_2.head()

# %%
df_2.info()

# %%
df_2.describe()

# %%
# распределение цены (в лог шкале )
plt.figure(figsize=(10, 5))
sns.histplot(
    df_1["item_price"],
    bins=50,
    kde=True,
    color="skyblue",
    edgecolor="black",
    log_scale=(True, False),
)
plt.title("Распределение цен товаров (лог шкала)")
plt.xlabel("item_price (log)")
plt.ylabel("Частота")
plt.show()


# %%
sns.boxplot(df_1["item_price"])


# %%
# сначала разберемся. что же за выбросы в цене товара
rich_items = df_1[df_1["item_price"] > 100_000]
(len(rich_items))
rich_items.head(5)

# %%
# поскольку ошибка в этом товаре произошла один раз, товар вообще продавался один раз всю историю и предсказывать его продаваемость не будем - удаляем
df_1 = df_1[df_1["item_id"] != 6066]


# %%
sns.boxplot(df_1["item_price"])  # убедились, что стало лучше

# %%
selected = df_1[df_1["item_price"] < 0]
selected

# %%
df_1 = df_1[
    df_1["item_price"] > 0
]  # убрала товары с отрицательной ценой, с выбросами с item_price разобрались

# %%
sns.boxplot(df_1["item_cnt_day"])

# %%
selected = df_1[df_1["item_cnt_day"] > 900]
selected


# %%
df_1 = df_1[df_1["item_cnt_day"] <= 900]  # удаляю выбросы в item_cnt_day

# %%
sns.boxplot(df_1["item_cnt_day"])


# %%
# поскольку предсказание идет о продажах за месяц, а не за день, могу сгруппировать и избавиться от лишних признаков
df_month = df_1.groupby(
    ["shop_id", "item_id", "date_block_num", "item_category_id"], as_index=False
).agg({"item_price": "mean", "item_cnt_day": "sum"})

df_month = df_month.rename(columns={"item_cnt_day": "item_cnt_month"})

df_month = df_month[
    [
        "item_id",
        "shop_id",
        "item_price",
        "date_block_num",
        "item_cnt_month",
        "item_category_id",
    ]
]

df_month.head()


# %%
sns.boxplot(df_month["item_cnt_month"])

# %% [markdown]
# Тут я хочу проверить и убедиться, какие есть выбросы в item_cnt_month:
# - два значения экстремальные
# - отрицательные продажи (их много)

# %%
selected = df_month[df_month["item_cnt_month"] > 1500]
selected

# %%
selected = df_month[df_month["item_cnt_month"] < 0]
selected

# %%
# убедимся, что если уберем эти выбросы - это не единственная запись о продаже данного товара в данном магазине
df_11 = df_1[
    (df_1["item_id"] == 9248) & (df_1["shop_id"] == 12) & (df_1["date_block_num"] == 32)
]
df_11.head()

# %%
df_11 = df_1[
    (df_1["item_id"] == 9249) & (df_1["shop_id"] == 55) & (df_1["date_block_num"] == 32)
]
df_11.head()

# %% [markdown]
# Видим, что записей о продаже данных товара много, так что если удалим эти две строки с выбросами, нам все равно будет на чем строить предсказания

# %%
df_month = df_month[df_month["item_cnt_month"] < 1500]


# %%
sns.boxplot(df_month["item_cnt_month"])

# %%
q99 = df_month["item_cnt_month"].quantile(0.99)
df_cut = df_month[df_month["item_cnt_month"] <= q99]

plt.figure(figsize=(10, 5))
sns.histplot(data=df_cut, x="item_cnt_month", bins=30, kde=False, color="steelblue")
plt.title(f"item_cnt_month (≤ 99%-квантиль)")
plt.xlabel("item_cnt_month")
plt.ylabel("Частота")
plt.tight_layout()
plt.show()


# %% [markdown]
# Далее, чтобы оставить в правильном диапозоне предсказания нашей модели - нужно учесть строку на кагле:
# "True target values are clipped into [0,20] range"
# Поскольку нам нужно, чтобы модель давала предсказания в диапозоне от 0 до 20, нужно и чтобы обучалась она на данных в соответственно в таком же диапозоне,
# поэтому придется для таргета сделать clipping[0,20]

# %%
df_month["item_cnt_month"] = df_month["item_cnt_month"].clip(lower=0, upper=20)

# %% [markdown]
# Далее начнем чекать магазины

# %%
# from scripts.src1 import clean_name

import re


def clean_name(s):
    s = s.lower().strip()
    s = re.sub(r"[^а-яa-z0-9\s]", "", s)
    return s


# %%
df_shops["shop_clean"] = df_shops["shop_name"].apply(clean_name)

# %%
sorted_shops = df_shops["shop_clean"].sort_values()
print(sorted_shops.to_list())

# %% [markdown]
# Вызывающие сомнения магазины:
#
# 'ростовнадону трк мегацентр горизонт', 'ростовнадону трк мегацентр горизонт островной', 'ростовнадону тц мега'
#
# 'якутск орджоникидзе 56', 'якутск орджоникидзе 56 фран'
#
# 'жуковский ул. чкалова 39м', 'жуковский ул. чкалова 39м'
#
# С моей точки зрения, это могут быть одни и те же магазины, но нужно проверить статистически

# %%
df_11 = df_1[df_1["shop_id"] == 39]
df_22 = df_1[df_1["shop_id"] == 40]
df_33 = df_1[df_1["shop_id"] == 41]

# %%
series1 = df_11.groupby("date_block_num")["item_cnt_day"].sum()
series2 = df_22.groupby("date_block_num")["item_cnt_day"].sum()
series3 = df_33.groupby("date_block_num")["item_cnt_day"].sum()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(series1.index, series1.values, marker="o", label="Магазин 1")
ax.plot(series2.index, series2.values, marker="s", label="Магазин 2")
ax.plot(series3.index, series3.values, marker="o", label="Магазин 3")
ax.set_xlabel("Месяц (date_block_num)")
ax.set_ylabel("Продажи за месяц")
ax.set_title("Сравнение месячных продаж магазина 1 и магазина 2 и 3")
ax.legend(title="shop_id")
ax.grid(True)

plt.xticks(
    sorted(set(series1.index) | set(series2.index) | set(series3.index)), rotation=45
)
plt.tight_layout()
plt.show()


# %% [markdown]
# Не одинаковые магазины

# %%
df_3 = df_1[df_1["shop_id"] == 10]
df_4 = df_1[df_1["shop_id"] == 11]

# %%
series1 = df_3.groupby("date_block_num")["item_cnt_day"].sum()
series2 = df_4.groupby("date_block_num")["item_cnt_day"].sum()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(series1.index, series1.values, marker="o", label="Магазин 3")
ax.plot(series2.index, series2.values, marker="s", label="Магазин 4")

ax.set_xlabel("Месяц (date_block_num)")
ax.set_ylabel("Продажи за месяц")
ax.set_title("Сравнение месячных продаж магазина 1 и магазина 2")
ax.legend(title="shop_id")
ax.grid(True)

plt.xticks(sorted(set(series1.index) | set(series2.index)), rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# Получаем вывод -  получается просто кто-то ошибся и ровно один месяц записал данные одного магазина создав другое название, без зазарения совести добавлю его туда, обьединив эти магазины

# %%
# заменю и в df_1 и df_month
df_1.loc[df_1["shop_id"] == 11, ["shop_id", "shop_name"]] = [
    10,
    "Жуковский ул. Чкалова 39м",
]

df_month.loc[df_month["shop_id"] == 11, ["shop_id"]] = [10]


# %%
df_5 = df_1[df_1["shop_id"] == 0]
df_6 = df_1[df_1["shop_id"] == 57]


# %%
series1 = df_5.groupby("date_block_num")["item_cnt_day"].sum()
series2 = df_6.groupby("date_block_num")["item_cnt_day"].sum()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(series1.index, series1.values, marker="o", label="Магазин 5")
ax.plot(series2.index, series2.values, marker="s", label="Магазин 6")

ax.set_xlabel("Месяц (date_block_num)")
ax.set_ylabel("Продажи за месяц")
ax.set_title("Сравнение месячных продаж магазина 1 и магазина 2")
ax.legend(title="shop_id")
ax.grid(True)

plt.xticks(sorted(set(series1.index) | set(series2.index)), rotation=45)
plt.tight_layout()
plt.show()

# %%
df_1.loc[df_1["shop_id"] == 0, ["shop_id", "shop_name"]] = [
    57,
    "Якутск Орджоникидзе, 56",
]

df_month.loc[df_month["shop_id"] == 0, ["shop_id"]] = [57]

# %% [markdown]
# Мы нашли магазины, которые совпадают, изменили информацию о них в тренировочном датасете (в тестовом вообще нет магазинов, которые дублированы, я проверила)

# %%
selected = df_2[df_2["shop_id"] == 0]  # собственно сама проверка
selected

# %%
selected = df_2[df_2["shop_id"] == 11]  # проверка
selected

# %%
df_items.shape

# %% [markdown]
# Мы видим, что всего товаров у нас 22170, до этого я пробовала чистить товары и нашла, что 77 пар, в которых не понятно, разные ли это товары и стоит проверить статистически, но я считаю, что раз товаров повторяющихся так мало (всего 0.35%) я не считаю, что будет такой большой выхлоп от ее очистки

# %%
df_month_test = df_2[["shop_id", "item_id", "item_category_id"]]

# %%
df_month_test["date_block_num"] = 34
df_month_test.head()

# %%
df_month.to_csv("df_train_dqc.csv", index=False)

# %%
df_month_test.to_csv("df_test_dqc.csv", index=False)

# %% [markdown]
# Далее к получившимся датасетам добавлю новые фичи, возможно модернизирую имеющиеся или выкину старые
