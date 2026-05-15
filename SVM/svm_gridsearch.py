import os
import numpy as np
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

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=SEED)

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
                'svm__C': [0.1, 1, 10],
                'svm__class_weight': [None, 'balanced']
            },
            {
                'svm__kernel': ['rbf'],
                'svm__C': [0.1, 1, 10],
                'svm__gamma': ['scale', 'auto'],
                'svm__class_weight': [None, 'balanced']
            },
            {
                'svm__kernel': ['poly'],
                'svm__C': [0.1, 1, 10],
                'svm__degree': [2, 3, 4],
                'svm__gamma': ['scale', 'auto'],
                'svm__class_weight': [None, 'balanced']
            },
            {
                'svm__kernel': ['sigmoid'],
                'svm__C': [0.1, 1, 10],
                'svm__gamma': ['scale', 'auto'],
                'svm__class_weight': [None, 'balanced']
            },
        ]

        trainer = GridSearchTrainer(self.pipeline, param_grid)
        trainer.train(self.X_train, self.y_train) # trenowanie

        self.save_table(trainer)


    def save_table(self, trainer):
        results_table = trainer.get_results_table()

        os.makedirs("SVM", exist_ok=True)

        with open("SVM/gridsearch_results.txt", "w", encoding="utf-8") as f:
            f.write(results_table.to_string(index=False))
