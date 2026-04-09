import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import table
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from classes.dataloader import DataLoader
from classes.grid_search_trainer import GridSearchTrainer

SEED = 42

class SVMPipeline:
    def __init__(self):
        data_loader = DataLoader()
        datasets = data_loader.load()

        self.X_train, self.y_train = datasets['train']
        self.X_test, self.y_test = datasets['test']

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=SEED)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test, random_state=SEED)

        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(np.unique(self.y_train))

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC())
        ])
    
    def train(self):
        param_grid = [
            {
                'svm__kernel': ['linear'],
                'svm__C': [0.001, 0.1, 1, 10]
            },
            {
                'svm__kernel': ['rbf'],
                'svm__C': [0.001, 0.1, 1, 10],
                'svm__gamma': ['scale', 'auto', 0.1]
            },
            {
                'svm__kernel': ['poly'],
                'svm__C': [0.001, 0.1, 1, 10],
                'svm__degree': [2, 3, 4],
                'svm__gamma': ['scale']
            }
        ]

        trainer = GridSearchTrainer(self.pipeline, param_grid)
        grid_result = trainer.train(self.X_train, self.y_train) # trenowanie

        os.makedirs("SVM", exist_ok=True)
        joblib.dump(grid_result.best_estimator_, "SVM/best_model.pkl")

        self.save_table(trainer)

    def save_table(self, trainer):
        results_table = trainer.get_results_table() # tworzenie tabeli z wszystkimi modelami i parametrami
        fig, ax = plt.subplots(figsize=(12, len(results_table)*0.5))
        ax.axis('off')
        tbl = table(ax, results_table, loc='center', cellLoc='center', colWidths=[0.1]*len(results_table.columns))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        os.makedirs("SVM", exist_ok=True)
        plt.savefig("SVM/gridsearch_results.png", bbox_inches='tight', dpi=150)
        plt.close()
