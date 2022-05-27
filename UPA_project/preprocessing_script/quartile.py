from sklearn.base import TransformerMixin


class Quartile(TransformerMixin):

    def __init__(self, final_dataset, column):
        self.quartil_down = final_dataset[column].quantile(0.05)
        self.quartil_up = final_dataset[column].quantile(0.95)

    def fit(self):
        return self.quartil_down, self.quartil_up

    def transform(self, final_dataset, column):
        for i, value in final_dataset.iterrows():
            if value[column] < self.quartil_down:
                final_dataset.loc[i, column] = self.quartil_down
            elif value[column] > self.quartil_up:
                final_dataset.loc[i, column] = self.quartil_up
        return final_dataset