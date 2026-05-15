import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from classes.dataloader import DataLoader
from classes.grid_search_trainer import GridSearchTrainer

SEED = 42

class RFPipeline:
    def __init__(self):
        data_loader = DataLoader()
        datasets = data_loader.load()

        self.X_train, self.y_train = datasets['train']

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=SEED)

        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(np.unique(self.y_train))

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=SEED))
        ])
    
    def train(self):
        param_grid = { 
            'rf__n_estimators': [100, 300],
            'rf__criterion' :['gini', 'entropy', 'log_loss'],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__max_depth' : [10, 20, None],
            'rf__bootstrap' : [True, False],
            'rf__class_weight' : [None, 'balanced']
        }

        trainer = GridSearchTrainer(self.pipeline, param_grid)
        trainer.train(self.X_train, self.y_train) # trenowanie

        self.save_table(trainer)

    def save_table(self, trainer):
        results_table = trainer.get_results_table()

        os.makedirs("RF", exist_ok=True)

        with open("RF/gridsearch_results.txt", "w", encoding="utf-8") as f:
            f.write(results_table.to_string(index=False))
