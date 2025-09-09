# %% [markdown]
# Что изменила:
# - добавила зависимости в requirements.txt
# - изменила названия датасетов на более информативные (df_1 -> df_sales, df_2 -> df_predict)
# - testt - это не опечатка, а чтобы отличать df_test, который дается в условии и тот, который под конец сохраним для тестирования модели
# - заменила неагрегированный датасет на агрегированный
# - использовала seaborn lineplot для графиков
# - на графиках видны имена магазинов
#

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
df_sales = df_sales_train.merge(df, on="item_id", how="left", validate="many_to_one")


# %%
# df_sales - датасет, который содержит вообще всю информацию про продаваемость товаров за весь период

df_sales = df_sales.merge(df_shops, on="shop_id", how="left", validate="many_to_one")
df_sales.head()

# %%
df_sales.isnull().sum()  # убедимся, что нет пропусков

# %%
df_sales.describe()

# %%
df_sales.info()

# %%
df_predict = df_testt.merge(df, on="item_id", how="left", validate="many_to_one")


# %%
# df_predict - датасет, который содержит всю информацию про товары, продаваемость которых мы будем предсказывать
df_predict = df_predict.merge(
    df_shops, on="shop_id", how="left", validate="many_to_one"
)
df_predict.head()

# %%
df_predict.info()

# %%
df_predict.describe()

# %%
# распределение цены (в лог шкале )
plt.figure(figsize=(10, 5))
sns.histplot(
    df_sales["item_price"],
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
sns.boxplot(df_sales["item_price"])


# %%
# сначала разберемся. что же за выбросы в цене товара
rich_items = df_sales[df_sales["item_price"] > 100_000]
(len(rich_items))
rich_items.head(5)

# %%
# поскольку ошибка в этом товаре произошла один раз, товар вообще продавался один раз всю историю и предсказывать его продаваемость не будем - удаляем
df_sales = df_sales[df_sales["item_id"] != 6066]


# %%
sns.boxplot(df_sales["item_price"])  # убедились, что стало лучше

# %%
selected = df_sales[df_sales["item_price"] < 0]
selected

# %%
df_sales = df_sales[
    df_sales["item_price"] > 0
]  # убрала товары с отрицательной ценой, с выбросами с item_price разобрались

# %%
sns.boxplot(df_sales["item_cnt_day"])

# %%
selected = df_sales[df_sales["item_cnt_day"] > 900]
selected


# %%
df_sales = df_sales[df_sales["item_cnt_day"] <= 900]  # удаляю выбросы в item_cnt_day

# %%
sns.boxplot(df_sales["item_cnt_day"])


# %%
# поскольку предсказание идет о продажах за месяц, а не за день, могу сгруппировать и избавиться от лишних признаков
df_month = df_sales.groupby(
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
df_11 = df_month[
    (df_month["item_id"] == 9248)
    & (df_month["shop_id"] == 12)
    & (df_month["date_block_num"] == 32)
]
df_11.head()

# %%
df_11 = df_month[
    (df_month["item_id"] == 9249)
    & (df_month["shop_id"] == 55)
    & (df_month["date_block_num"] == 32)
]
df_11.head()

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
# import sys
# sys.path.append("../")

# %%
# from scripts.scr1 import clean_name

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
shop_names = {
    39: "ростовнадону трк мегацентр горизонт",
    40: "ростовнадону трк мегацентр горизонт островной",
    41: "ростовнадону тц мега",
}


df_plot = df_month.reset_index().assign(
    shop_name=lambda d: d["shop_id"].map(shop_names)
)

shops_of_interest = [39, 40, 41]
df_sub = df_plot[df_plot["shop_id"].isin(shops_of_interest)]

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_sub,
    x="date_block_num",
    y="item_cnt_month",
    hue="shop_name",
    style="shop_name",
    markers=True,
    dashes=False,
)


plt.xlabel("Месяц (date_block_num)")
plt.ylabel("Продажи за месяц")
plt.title("Сравнение месячных продаж магазинов")
plt.xticks(sorted(df_sub["date_block_num"].unique()), rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# Не одинаковые магазины

# %%

shop_names = {
    10: "жуковский ул чкалова 39м - id_10",
    11: "жуковский ул чкалова 39м - id_11",
}


df_plot = df_month.reset_index().assign(
    shop_name=lambda d: d["shop_id"].map(shop_names)
)

shops_of_interest = [10, 11]
df_sub = df_plot[df_plot["shop_id"].isin(shops_of_interest)]

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_sub,
    x="date_block_num",
    y="item_cnt_month",
    hue="shop_name",
    style="shop_name",
    markers=True,
    dashes=False,
)


plt.xlabel("Месяц (date_block_num)")
plt.ylabel("Продажи за месяц")
plt.title("Сравнение месячных продаж магазинов")
plt.xticks(sorted(df_sub["date_block_num"].unique()), rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# Получаем вывод -  получается просто кто-то ошибся и ровно один месяц записал данные одного магазина создав другое название, без зазарения совести добавлю его туда, обьединив эти магазины

# %%
# заменю и в df_1 и df_month
df_sales.loc[df_sales["shop_id"] == 11, ["shop_id", "shop_name"]] = [
    10,
    "Жуковский ул. Чкалова 39м",
]

df_month.loc[df_month["shop_id"] == 11, ["shop_id"]] = [10]


# %%

shop_names = {
    0: "якутск орджоникидзе 56 фран",
    57: "якутск орджоникидзе 56",
}


df_plot = df_month.reset_index().assign(
    shop_name=lambda d: d["shop_id"].map(shop_names)
)

shops_of_interest = [0, 57]
df_sub = df_plot[df_plot["shop_id"].isin(shops_of_interest)]

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_sub,
    x="date_block_num",
    y="item_cnt_month",
    hue="shop_name",
    style="shop_name",
    markers=True,
    dashes=False,
)


plt.xlabel("Месяц (date_block_num)")
plt.ylabel("Продажи за месяц")
plt.title("Сравнение месячных продаж магазинов")
plt.xticks(sorted(df_sub["date_block_num"].unique()), rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
df_sales.loc[df_sales["shop_id"] == 0, ["shop_id", "shop_name"]] = [
    57,
    "Якутск Орджоникидзе, 56",
]

df_month.loc[df_month["shop_id"] == 0, ["shop_id"]] = [57]

# %% [markdown]
# Мы нашли магазины, которые совпадают, изменили информацию о них в тренировочном датасете (в тестовом вообще нет магазинов, которые дублированы, я проверила)

# %%
selected = df_predict[df_predict["shop_id"] == 0]  # собственно сама проверка
selected

# %%
selected = df_predict[df_predict["shop_id"] == 11]  # проверка
selected

# %%
df_items.shape

# %% [markdown]
# Мы видим, что всего товаров у нас 22170, до этого я пробовала чистить товары и нашла, что 77 пар, в которых не понятно, разные ли это товары и стоит проверить статистически, но я считаю, что раз товаров повторяющихся так мало (всего 0.35%) я не считаю, что будет такой большой выхлоп от ее очистки

# %%
df_month_test = df_predict[["shop_id", "item_id", "item_category_id"]]

# %%
df_month_test["date_block_num"] = 34
df_month_test.head()

# %%
df_month.to_csv("df_train_dqc.csv", index=False)

# %%
df_month_test.to_csv("df_test_dqc.csv", index=False)

# %% [markdown]
# Далее к получившимся датасетам добавлю новые фичи, возможно модернизирую имеющиеся или выкину старые
