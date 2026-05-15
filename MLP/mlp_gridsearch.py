import os
import random
import numpy as np
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

data_loader = DataLoader()
datasets = data_loader.load()

X_train, y_train = datasets['train']

X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))

def build_model_fn(num_layers=1, units=32, activation='relu', lr=0.001):

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(Dense(units, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

class MLPPipeline:
    def __init__(self):
        self.model = KerasClassifier(model=build_model_fn, verbose=2)

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', self.model)
        ])

    
    def train(self):
        param_grid = {
            'mlp__model__num_layers': [1, 2, 3, 4],
            'mlp__model__units': [16, 32, 64, 128],
            'mlp__model__activation': ['tanh', 'relu', 'sigmoid'],
            'mlp__model__lr': [0.001, 0.01, 0.1],
            'mlp__batch_size': [128],
            'mlp__epochs': [50]
        }

        trainer = GridSearchTrainer(self.pipeline, param_grid)
        trainer.train(X_train, y_train) # trenowanie

        self.save_table(trainer)

    def save_table(self, trainer):
        results_table = trainer.get_results_table()

        os.makedirs("MLP", exist_ok=True)

        with open("MLP/gridsearch_results.txt", "w", encoding="utf-8") as f:
            f.write(results_table.to_string(index=False))