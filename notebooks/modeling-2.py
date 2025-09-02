# %%
import numpy as np
import pandas as pd
from Validator import Validator
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from Splitter import Splitter


# %%
df_train = pd.read_csv("D:\\repos\\ML_innovise\\data\\processed\\df_train_eda.csv")
df_test = pd.read_csv("D:\\repos\\ML_innovise\\data\\processed\\df_test_eda.csv")

# %%
df_testt = pd.read_csv(r"D:\ML_innovise\test.csv")
df_test = df_test.merge(df_testt, on=["shop_id", "item_id"], how="left")

# %% [markdown]
# ### 1. Линейная регрессия:

# %% [markdown]
# Перед тем, как обучать, есть вещи, на которые линейная регрессия смотрит ну очень пристально:
# - мультиколлинеарность
# Поэтому построим hit_map

# %%
cols = [
    "item_cnt_month",
    "item_price",
    "cnt_lag_1",
    "cnt_lag_2",
    "cnt_lag_3",
    "cnt_lag_12",
    "cnt_rm_3",
    "cnt_rm_6",
    "item_avg_price_prev",
]
corr = df_train[cols].corr()
plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Корреляционная матрица")
plt.show()


# %% [markdown]
# Видим мультиколлинераность между: cnt_lag_1 и cnt_rm_3 - оставлю cnt_lag_1

# %%
cols = [
    "item_cnt_month",
    "item_price",
    "cnt_lag_1",
    "cnt_lag_2",
    "cnt_lag_3",
    "cnt_lag_12",
    "cnt_rm_6",
    "item_avg_price_prev",
]
corr = df_train[cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Корреляционная матрица")
plt.show()

# %% [markdown]
# Поскольку здесь признаков мало и хочелось побольше сохранить - я считаю, что порог такой:
#
# все признаки,у кот корреляция выше 0.8 - скоррелированы. Таких больше нет, значит можем обучать.

# %%
modelLR = LinearRegression(fit_intercept=True, n_jobs=-1)

# %%
splitter = Splitter("date_block_num", 33)

# %%
validator = Validator(
    df_train,
    splitter,
    modelLR,
    metric=lambda y, yhat: sqrt(mean_squared_error(y, yhat)),
)
features = [
    "cnt_lag_1",
    "cnt_lag_3",
    "cnt_lag_2",
    "cnt_lag_12",
    "cnt_rm_6",
    "item_avg_price_prev",
    "item_price",
    "date_block_num",
    "avg_price_prev_missing",
]
result = validator.run(features, "item_cnt_month")
print("Val_score_RMSE:", result)


# %%
X_tr, y_tr, X_val, y_val = splitter.split(df_train, features, "item_cnt_month")
X_tr_full = pd.concat([X_tr, X_val], axis=0)
y_tr_full = pd.concat([y_tr, y_val])

# %%
modelLR.fit(X_tr_full, y_tr_full)
X_test = df_test[features]
pred = modelLR.predict(X_test)

# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": pred})

# %%
submission.to_csv("submissionLR.csv", index=False)

# %% [markdown]
# результат на кагле - 1.03054
#
# Теперь хочу попробовать, а если перед тем, как подавать данные модели - я их масштабирую -  что будет?

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_full)

X_test_scaled = scaler.transform(X_test)

modelLR2 = LinearRegression(fit_intercept=True, n_jobs=-1)
modelLR2.fit(X_tr_scaled, y_tr_full)
pred = modelLR2.predict(X_test_scaled)


# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": pred})


# %%
submission.to_csv("submissionLR_scaled.csv", index=False)

# %% [markdown]
# результат оказалось абсолютно таким же.
# Почитав внимательнее - я узнала, что линейная регрессия без регуляризации (как я использовала здесь) - абослютно инвариантна к линейным преобразованиям,
# поэтому то, что я тут пыталась сделать - и не возымело эффекта.

# %% [markdown]
# ### 2. SVR

# %% [markdown]
# Дальше у нас пойдет SVR - это вид SVM для регресси я так поняла. Основаная подготовка данных:
# - масштабировать обязательно
# Здесь уже придется подбирать гиперпараметры, я буду использовать GridSearchCV:
# Какие гиперпараматеры нужно подобрать:
# - C - штраф за ошибки (то есть параметр регуляризации)
# - epsilon - ширина разделительной зоны
# - kernel — ядро ('linear', 'rbf', 'poly', …)
# - gamma - масштаб ядра
# - degre - степень полинома (когда выбираем ядро poly)
# - coef - свободный какой-то коэф

# %% [markdown]
# Я попыталась обучить несколько раз SVR, пыталась подобрать самые облегченные гиперпараметры - но сколько бы не обучала модель, она так и не обучислась и я решила перейти к другим моделям.

# %% [markdown]
# ### *Target Encoding

# %% [markdown]
# я захотела закодировать id моих переменных , например: shop_id, item_id, simple_category_id - потому что мне кажется, что эти фичи могут нести в себе полезную информацию - тк например один и тот же товар в разных магазинах могут иметь похожие продажи - но мы не можем знать, что это один и тот товар, не зная его id. Но при этом, id могут сбить нашу модель, ведь она будет искать какую-то зависимость в упорядоченности,а это всего лишь id.
#
# Мне кажется, будет хорошей идеей использовать target encoding.


# %%
# cols - это мой список колонок для кодирования
def fit_target_encoding(X_tr, y_tr, cols, smoothing=10):
    global_mean = (
        y_tr.mean()
    )  # это понадобится в формуле сглаживания и для новых категорий
    mapping_dicts = {}  # короче словарь со словарями
    for col in cols:
        stats = (
            pd.DataFrame({col: X_tr[col], "y": y_tr})
            .groupby(col)["y"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "col_mean", "count": "col_count"})
        )
        # теперь нужно применить формулу te со сглаживанием
        stats["te"] = (
            stats["col_count"] * stats["col_mean"] + global_mean * smoothing
        ) / (stats["col_count"] + smoothing)
        mapping_dicts[col] = stats["te"].to_dict()
    return mapping_dicts, global_mean


def apply_target_encoding(X, mapping_dicts, global_mean, cols):
    X_enc = X.copy()
    for col in cols:
        te_col = f"{col}_te"
        X_enc[te_col] = X_enc[col].map(mapping_dicts[col])
        X_enc[te_col].fillna(global_mean, inplace=True)
    return X_enc


cols_to_encode = ["item_id", "shop_id", "simple_category_id"]
smoothing_k = 10

mapping_dicts, global_mean = fit_target_encoding(
    X_tr_full, y_tr_full, cols_to_encode, smoothing_k
)
X_tr_enc = apply_target_encoding(X_tr_full, mapping_dicts, global_mean, cols_to_encode)
X_test_enc = apply_target_encoding(X_test, mapping_dicts, global_mean, cols_to_encode)


# %%
from sklearn.ensemble import RandomForestRegressor

# %%
# без target encoding и без id
modelRF1 = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    oob_score=True,
)

#  с target encoding вместо id
modelRF2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    oob_score=True,
)


# %%
cols_to_drop = ["shop_id", "item_id", "simple_category_id"]
X_tr_enc = X_tr_enc.drop(columns=cols_to_drop)
X_test_enc = X_test_enc.drop(columns=cols_to_drop)


# %%
X_tr_full = X_tr_full.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

# %%
modelRF1.fit(X_tr_full, y_tr_full)
preds1 = modelRF1.predict(X_test)

# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": preds1})

# %%
submission.to_csv("submissionRF_without_TE_scaled1.csv", index=False)

# %%
modelRF2.fit(X_tr_enc, y_tr_full)
preds2 = modelRF2.predict(X_test_enc)

# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": preds2})

# %%
submission.to_csv("submissionRF_with_TE_scaled1.csv", index=False)

# %% [markdown]
# Я решила попробовать обучить Random Forest со классически гиперпаарметрами (тк подбор гиперпараметров - это уже следующий этап), но обучала я две разные модели:
#
# - одну без target encoding
# - одну с target encoding
#
# Результат получился следующий:
#
# - RF without TE -> 1.21640
# - RF with TE -> 1.19098
#
# Делаем вывод - добавление target encoding дает выигрыш в обучении для случайного леса.

# %%
modelLR_with_TE = LinearRegression(fit_intercept=True, n_jobs=-1)

# %%
modelLR_with_TE.fit(X_tr_enc, y_tr_full)
preds3 = modelLR_with_TE.predict(X_test_enc)

# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": preds3})

# %%
submission.to_csv("submission_LR_with_TE.csv", index=False)

# %% [markdown]
# Тут у меня появилась мысль: попробовать добавить target encoding в лин регрессию, станет ли ее показания лучше:
#
# Результат: 1.03230 -> результат не стал лучше, ведь до этого был 1.03054
#
#

# %%
import xgboost as xgb

# %%
from xgboost import XGBRegressor

model = XGBRegressor(
    objective="reg:squarederror", n_estimators=200, learning_rate=0.1, random_state=42
)

model.fit(X_tr_full, y_tr_full)
preds4 = model.predict(X_test)

# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": preds4})

# %%
submission.to_csv("submission_XGB.csv", index=False)

# %% [markdown]
# Здесь я постаралась обучить модель XGBoost с базовыми гиперпарамтерами:
#
# Результат -> 1.18062

# %% [markdown]
# Результат: в данном ноутбуке я обучила такие модели, как:
# - линейная регрессия
# - линейная регрессия with Standard Scaler
# - линейная регрессия with Target encoding
# - Random Forest
# - Random Forest with Target encoding
# - XGBoost
#
# Самые лучшие результаты показала линейная регрессия. ЧТо это может сказать о наших данных:
#
# Наша целевая переменная практически линейно связана с нашими признаками, и в данных нет выраженвх нелинейностей или сложных взаимодействий,
#
# которые могли бы принести выгоду более сложным алгоритмам.
#
#
