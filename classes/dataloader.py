import pandas as pd

class DataLoader:
    """Laduje dane train, test, val z plikow .csv"""
    def __init__(self, splits=['train', 'test']):
        self.splits = splits
        self.datasets = {}

    def load(self):
        for split in self.splits:
            path = f"datasets/{split}_preprocessed.csv"
            df = pd.read_csv(path)
            X = df.iloc[:, 1:].values
            y = df.iloc[:, 0].values
            self.datasets[split] = (X, y)
        return self.datasets