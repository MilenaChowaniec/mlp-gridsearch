import os
import random
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import table
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from classes.dataloader import DataLoader
from classes.grid_search_trainer import GridSearchTrainer

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

class MLPPipeline:
    def __init__(self):
        data_loader = DataLoader()
        datasets = data_loader.load()

        self.X_train, self.y_train = datasets['train']
        self.X_test, self.y_test = datasets['test']

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=SEED)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test, random_state=SEED)

        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(np.unique(self.y_train))

        self.model = KerasClassifier(model = self.build_model_fn, verbose=2)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', self.model)
        ])
    
    def build_model_fn(self, num_layers=1, units=32, activation='relu', lr=0.001):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        for _ in range(num_layers):
            model.add(Dense(units, activation=activation))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model
    
    def train(self):
        param_grid = {
            'mlp__model__num_layers': [1, 2, 3],
            'mlp__model__units': [16, 32, 64],
            'mlp__model__activation': ['relu', 'tanh'],
            'mlp__model__lr': [0.01, 0.001, 0.0001],
            'mlp__batch_size': [64],
            'mlp__epochs': [50]
        }

        trainer = GridSearchTrainer(self.pipeline, param_grid)
        grid_result = trainer.train(self.X_train, self.y_train) # trenowanie

        os.makedirs("MLP", exist_ok=True)
        joblib.dump(grid_result.best_estimator_, "MLP/best_model.pkl")

        self.save_table(trainer)

    def save_table(self, trainer):
        results_table = trainer.get_results_table() # tworzenie tabeli z wszystkimi modelami i parametrami
        fig, ax = plt.subplots(figsize=(12, len(results_table)*0.5))
        ax.axis('off')
        tbl = table(ax, results_table, loc='center', cellLoc='center', colWidths=[0.1]*len(results_table.columns))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        os.makedirs("MLP", exist_ok=True)
        plt.savefig("MLP/gridsearch_results.png", bbox_inches='tight', dpi=150)
        plt.close()