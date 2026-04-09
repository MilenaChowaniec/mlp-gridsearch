from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
        cm = confusion_matrix(y_test, y_pred)
        return f1, precision, recall, acc, cm