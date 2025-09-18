# %%
import pandas as pd

# %%
df_train = pd.read_csv("D:\\repos\\ML_innovise\\notebooks\\df_train_eda.csv")
df_test = pd.read_csv("D:\\repos\\ML_innovise\\notebooks\\df_test_eda.csv")

# %%
from Splitter import Splitter

splitter = Splitter(time_col="date_block_num", train_end=33)

# %%
from Validator import Validator
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt

validator = Validator(
    df_train,
    splitter,
    model=Ridge(alpha=1.0),
    metric=lambda y, yhat: sqrt(mean_squared_error(y, yhat)),
)
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
result = validator.run(features, target_col="item_cnt_month")

print("Val RMSE:", result)

# %%
df_testt = pd.read_csv(r"D:\ML_innovise\test.csv")
df_test = df_test.merge(df_testt, on=["shop_id", "item_id"], how="left")

# %%
# это я для проверки классов Splitter и Validator брала базовую модель Ridge - результат на кагле 1.03054
X_tr, y_tr, X_val, y_val = splitter.split(
    df_train, features, target_col="item_cnt_month"
)
X_full = pd.concat([X_tr, X_val], axis=0)
y_full = pd.concat([y_tr, y_val], axis=0)

final_model = Ridge(alpha=1.0, random_state=42)
final_model.fit(X_full, y_full)


X_test = df_test[features]


test_pred = final_model.predict(X_test)


# %%
submission = pd.DataFrame({"ID": df_test["ID"], "item_cnt_month": test_pred})

submission.to_csv("submission2.csv", index=False)

# %%
from Model import Model

# а здесь я уже тестирую собственную модель - результат на кагле 1.17012
my_model = Model(
    "date_block_num", "simple_category_id", "item_cnt_month", ["shop_id", "item_id"]
)
my_model.fit(df_train)
my_pred = my_model.predict(df_test)
my_pred = my_pred.drop_duplicates()
my_pred.head()


# %%
my_pred["item_cnt_month"] = my_pred["target_med"]


# %%
preds = my_pred[["ID", "item_cnt_month"]]

# %%
preds.to_csv("simple_submission.csv", index=False)
