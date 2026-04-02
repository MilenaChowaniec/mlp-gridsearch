# import pandas as pd
# from collections import Counter
# import numpy as np
# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
# from sklearn import datasets

# # slownik z danymi dla train, val i test
# datasets = {}

# for split in ['train', 'test', 'val']:
#   path = f"{split}_preprocessed.csv"
#   df = pd.read_csv(path)
#   X = df.iloc[:, 1:].values # cechy
#   y = df.iloc[:, 0].values # etykiety
#   datasets[split] = (X, y)

# X_train, y_train = datasets['train'][0], datasets['train'][1]
# X_test, y_test = datasets['test'][0], datasets['test'][1]
# input_dim = X_train.shape[1]
# num_classes = len(np.unique(y_train))

# def build_model(num_layers=1, units=32, activation='relu', lr=0.001):
#     model = Sequential()

#     # input layer
#     model.add(Input(shape=(input_dim,)))

#     # hidden layers
#     for i in range(num_layers):
#       model.add(Dense(units, activation=activation))

#     # output layers
#     model.add(Dense(num_classes, activation='softmax'))

#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # parametry do testowania
# param_grid = {
#     'model__num_layers': [1, 2, 3],
#     'model__units': [16, 32, 64],
#     'model__activation': ['relu', 'tanh'],
#     'model__lr': [0.01, 0.001, 0.0001],
#     'batch_size': [64],
#     'epochs': [50]
# }

# model = KerasClassifier(
#     model=build_model,
#     verbose=2
# )

# grid = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=3,
#     scoring='accuracy',
#     n_jobs=1,
#     verbose=2
# )

# print(grid.param_grid)

# # trenowanie wszystkich kombinacji
# grid_result = grid.fit(X_train, y_train)

# results_df = pd.DataFrame(grid_result.cv_results_)

# param_cols = [col for col in results_df.columns if col.startswith('param_')]
# score_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']

# table = results_df[param_cols + score_cols].copy()

# new_columns = []
# for col in param_cols:
#     new_columns.append(col.replace('param_', '').replace('model__', ''))
# new_columns.extend(['Mean_Score', 'Std_Score', 'Rank'])
# table.columns = new_columns

# table = table.sort_values('Mean_Score', ascending=False)

# table['Mean_Score'] = table['Mean_Score'].round(4)
# table['Std_Score'] = table['Std_Score'].round(4)

# print(table.to_string(index=False))

# # sprawdzenie najlepszego modelu na danych testowych
# y_pred = grid_result.best_estimator_.predict(X_test)

# f1 = f1_score(y_test, y_pred, average='macro')
# precision = precision_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')

# print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

import pandas as pd
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class DataLoader:
    """Laduje dane train, test, val z plikow .csv"""
    def __init__(self, splits=['train', 'test', 'val']):
        self.splits = splits
        self.datasets = {}

    def load(self):
        for split in self.splits:
            path = f"{split}_preprocessed.csv"
            df = pd.read_csv(path)
            X = df.iloc[:, 1:].values
            y = df.iloc[:, 0].values
            self.datasets[split] = (X, y)
        return self.datasets


class MLPModel:
    """Tworzy i kompiluje model MLP zgodnie z parametrami."""
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes

    def build_model(self, num_layers=1, units=32, activation='relu', lr=0.001):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,))) # input layer
        for _ in range(num_layers): # hidden layers
            model.add(Dense(units, activation=activation))
        model.add(Dense(self.num_classes, activation='softmax')) # output layer
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


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
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        return f1, precision, recall



# Wczytanie danych
data_loader = DataLoader()
datasets = data_loader.load()

X_train, y_train = datasets['train']
X_test, y_test = datasets['test']

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))

# Utworzenie modelu
mlp = MLPModel(input_dim, num_classes)
keras_model = KerasClassifier(model=mlp.build_model, verbose=2)

# Parametry do GridSearch
param_grid = {
    'model__num_layers': [1, 2, 3],
    'model__units': [16, 32, 64],
    'model__activation': ['relu', 'tanh'],
    'model__lr': [0.01, 0.001, 0.0001],
    'batch_size': [64],
    'epochs': [50]
}

# Trenowanie
trainer = GridSearchTrainer(keras_model, param_grid)
grid_result = trainer.train(X_train, y_train)

# Wyświetlenie tabeli wyników
table = trainer.get_results_table()
print(table.to_string(index=False))

# Ocena na danych testowych
f1, precision, recall = trainer.evaluate_on_test(X_test, y_test)
print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")