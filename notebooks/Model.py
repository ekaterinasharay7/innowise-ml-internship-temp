class Model:
    """
    Итак, это моя кривая модель. Смысль ее предсказаний прост:
    когда я смотрела декомпозицию временного ряда - я заметила тренд и сезонность.
    - сезонность показывала, что каждый год в октябре продаваемость товаров начинает расти, впоть до декабря.
    А это значит, что можно ожидать, что продаваемость товаров в ноябре 2015 будет выше, чем в октябре 2015.
    А это уже в свою очередь значит, что мы имеем нижнюю границу для предсказаний продаж в ноябре.
    - тренд показывает, что с каждым годом продаваемость становится только хуже.
    Значит мы можем ожидать, что в ноябре 2014 продажи будут выше, чем в ноябре 2015.
    А значит мы имеем верхнюю границу.
    Если есть верхняя и нижняя границы - то ничто нам не мешает взять медиану между этими значениями - это и будет наше предсказание.
    *если товар не продавался в октябре 2015 или ноябре 2014 - заменяю на продаваемость категории данного товара в этом месяце

    """

    def __init__(self, date_col, category_col, target_col, key_cols):
        self.date_col = date_col
        self.category_col = category_col
        self.target_col = target_col
        self.key_cols = key_cols
        self._item_sales = None
        self._category_sales = None

    def fit(self, df):
        # сформировала мою табличку с продажами чисто для 22 и 33 нужных мне месяца
        df = df.copy()
        columns = self.key_cols + [self.date_col, self.target_col, self.category_col]
        self._item_sales = df.loc[
            df[self.date_col].isin([22, 33]), columns
        ].reset_index(drop=True)

        self._category_sales = self._item_sales.groupby(self.category_col)[
            self.target_col
        ].mean()
        mask_zero = self._item_sales[self.target_col] == 0
        self._item_sales.loc[mask_zero, self.target_col] = self._item_sales.loc[
            mask_zero, self.category_col
        ].map(self._category_sales)
        med_df = (
            self._item_sales.groupby(self.key_cols)[self.target_col]
            .median()
            .reset_index()
            .rename(columns={self.target_col: "target_med"})
        )
        self._item_sales = self._item_sales.merge(med_df, on=self.key_cols, how="left")
        return self

    def predict(self, df_true):
        df = df_true.copy()
        merge_cols = self.key_cols + ["target_med"]
        df = df.merge(self._item_sales[merge_cols], on=self.key_cols, how="left")
        df["target_med"] = df["target_med"].fillna(
            df[self.category_col].map(self._category_sales)
        )
        df = df.drop_duplicates()
        df["target_med"] = df["target_med"].clip(lower=0, upper=20)

        return df[self.key_cols + ["target_med", "ID"]]
