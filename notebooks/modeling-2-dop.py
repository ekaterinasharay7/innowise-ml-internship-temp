# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df_train = pd.read_csv("D:\\repos\\ML_innovise\\data\\processed\\df_train_eda.csv")
df_test = pd.read_csv("D:\\repos\\ML_innovise\\data\\processed\\df_test_eda.csv")

# %% [markdown]
# Тут у меня появилась, как мне казалось, гениальная идея - добавить полиномаильные признаки цены в лин рег.
# Обьясню откуда идея возникла:
#
# В будущем я буду оценивать важность признаков с помощью RFR и окажется, что цена имеет самую высокую важность, но корреляция между ценой и таргетом  - вообще 0.35, а значит между ними есть какая-то сильная зависимоть, но она не линейная.
#
# Результат этой затеи - в конце.
# * также я решила опробовать CatBoost в обучении.

# %%
from Splitter import Splitter

splitter = Splitter(time_col="date_block_num", train_end=33)

# %%
df_testt = pd.read_csv(r"D:\ML_innovise\test.csv")
df_test = df_test.merge(df_testt, on=["shop_id", "item_id"], how="left")

# %%
features = [
    "cnt_lag_1",
    "cnt_lag_3",
    "cnt_lag_2",
    "cnt_lag_12",
    "cnt_rm_3",
    "cnt_rm_6",
    "item_avg_price_prev",
    "item_price",
    "date_block_num",
    "avg_price_prev_missing",
]


# %%
X_tr, y_tr, X_val, y_val = splitter.split(
    df_train, features, target_col="item_cnt_month"
)
X_full = pd.concat([X_tr, X_val], axis=0)
y_full = pd.concat([y_tr, y_val], axis=0)


# %%
X = X_full.copy()
X["log_item_price"] = np.log1p(X["item_price"])
X["log_item_price_2"] = X["log_item_price"] ** 2
X["log_item_price_3"] = X["log_item_price"] ** 3
X["log_item_price_4"] = X["log_item_price"] ** 4


# %%
df_full = pd.concat([X, y_full], axis=1)


# %%
df_full.head()

# %%
df_full.isna().sum()

# %%
cols = [
    "item_cnt_month",
    "log_item_price",
    "cnt_lag_1",
    "cnt_lag_2",
    "cnt_lag_12",
    "cnt_rm_6",
    "item_avg_price_prev",
]

corr = df_full[cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.show()


# %%
from Validator import Validator
from math import sqrt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# %%
cols = [
    "item_cnt_month",
    "log_item_price_4",
    "cnt_lag_1",
    "cnt_lag_2",
    "cnt_lag_12",
    "cnt_rm_6",
    "item_avg_price_prev",
    "date_block_num",
    "avg_price_prev_missing",
]
features = [
    "log_item_price_4",
    "cnt_lag_1",
    "cnt_lag_2",
    "cnt_lag_12",
    "cnt_rm_6",
    "item_avg_price_prev",
    "date_block_num",
    "avg_price_prev_missing",
]


df_train = df_full[cols]
validator = Validator(
    df_train,
    splitter,
    model=Ridge(alpha=1.0),
    metric=lambda y, yhat: sqrt(mean_squared_error(y, yhat)),
)
result = validator.run(features, target_col="item_cnt_month")

print("Val RMSE:", result)

# %%
model = Ridge(alpha=1.0)
model.fit(df_full[features], y_full)


# %%
X_test = df_test.copy()
X_test["log_item_price"] = np.log1p(X_test["item_price"])
X_test["log_item_price_2"] = X_test["log_item_price"] ** 2
X_test["log_item_price_3"] = X_test["log_item_price"] ** 3
X_test["log_item_price_4"] = X["log_item_price"] ** 4


# %%
X_test = X_test[features]

# %%
preds = model.predict(X_test)

# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": preds})

submission.to_csv("submissionRidgeLogPrice4.csv", index=False)

# %%
from catboost import CatBoostRegressor

# %%
model = CatBoostRegressor(random_seed=42, verbose=False)


# %%
model.fit(X, y_full)

# %%
preds = model.predict(X_test)

# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": preds})

submission.to_csv("submissionCatBoost.csv", index=False)

# %% [markdown]
# Результат:
# - вместо обычной фичи цены ее лог трансформация - 1.07484
# - вместо обычной фичи цены ее лог трансформация в квадрате - 1.07036
# - вместо обычной фичи цены ее лог трансформация в кубе - 1.06243
# - вместо обычной фичи цены ее лог трансформация в 4 степени - 1.10683
# Видим, дальше степень нет смысла повышать
# Все это хорошо, но вот результат обычной лин рег никто из них не побил - 1.03054
# * catboost - 1.53542
# Если посмотреть, как эта идея отработала на валидационной выборке - то лучше, чем когда обычная цена, но из-за огромного количества пропусков в тестовом наборе цены видимо идея и не удалась.

# %% [markdown]
#
