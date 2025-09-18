import pandas as pd
from sklearn.base import clone


class ExpandingValidator:
    def __init__(self, model, splitter, metric):
        self.model = model
        self.splitter = splitter
        self.metric = metric

    def run(self, df, feature_cols, target_col):
        fold_metrics = []
        all_preds = []
        for val_month, df_tr, df_val in self.splitter.split(df):
            X_tr, y_tr = df_tr[feature_cols], df_tr[target_col]
            X_val, y_val = df_val[feature_cols], df_val[target_col]
            mdl = clone(self.model)
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_val)
            score = self.metric(y_val, y_pred)

            fold_metrics.append({"val_month": val_month, "val_score": score})
            preds_df = pd.DataFrame(
                {
                    self.splitter.time_col: df_val[self.splitter.time_col],
                    "y_true": y_val,
                    "y_pred": y_pred,
                },
                index=df_val.index,
            )
            all_preds.append(preds_df)

        all_preds_df = pd.concat(all_preds).sort_index()

        return fold_metrics, all_preds_df
