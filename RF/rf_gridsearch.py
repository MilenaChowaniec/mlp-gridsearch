import pandas as pd
from pandas.plotting import table
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class DataLoader:
    """Laduje dane train, test, val z plikow .csv"""
    def __init__(self, splits=['train', 'test', 'val']):
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
    
class GridSearchTrainer:
    """Przeprowadza GridSearchCV na modelu KerasClassifier i raportuje wyniki."""
    def __init__(self, model, param_grid, cv=3, scoring='accuracy', n_jobs=1):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.grid_result = None

    def train(self, X, y):
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=2
        )
        self.grid_result = grid.fit(X, y)
        return self.grid_result

    def get_results_table(self):
        results_df = pd.DataFrame(self.grid_result.cv_results_)
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        score_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']

        table = results_df[param_cols + score_cols].copy()
        new_columns = [col.replace('param_', '').replace('model__', '') for col in param_cols]
        new_columns.extend(['Mean_Score', 'Std_Score', 'Rank'])
        table.columns = new_columns
        table = table.sort_values('Mean_Score', ascending=False)
        table['Mean_Score'] = table['Mean_Score'].round(4)
        table['Std_Score'] = table['Std_Score'].round(4)
        return table

    def evaluate_on_test(self, X_test, y_test):
        y_pred = self.grid_result.best_estimator_.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        return f1, precision, recall, acc

data_loader = DataLoader()
datasets = data_loader.load()

X_train, y_train = datasets['train']
X_test, y_test = datasets['test']

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))

rf = RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

trainer = GridSearchTrainer(rf, param_grid)
grid_result = trainer.train(X_train, y_train)

best_params = grid_result.best_params_
print("Najlepsze parametry modelu:")
for k, v in best_params.items():
    print(f"{k}: {v}")

f1, precision, recall, acc = trainer.evaluate_on_test(X_test, y_test)
print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

best_model = grid_result.best_estimator_.model_
best_model.save("best_model.keras")  # zapis w formacie Keras

results_table = trainer.get_results_table() # tworzenie tabeli z wszystkimi modelami i parametrami
fig, ax = plt.subplots(figsize=(12, len(results_table)*0.5))
ax.axis('off')
tbl = table(ax, results_table, loc='center', cellLoc='center', colWidths=[0.1]*len(results_table.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.savefig("gridsearch_results.png", bbox_inches='tight', dpi=150)
plt.close()