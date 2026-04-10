import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from classes.dataloader import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    def __init__(self, model_path, output_dir="results"):
        self.model = joblib.load(model_path)
        self.output_dir = output_dir

        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.plots_dir = os.path.join(output_dir, "plots")

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate(self, X_test, y_test):
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)

        # podstawowe metryki
        results = {
            "accuracy": accuracy_score(y_test, self.y_pred),
            "precision": precision_score(y_test, self.y_pred, average="weighted"),
            "recall": recall_score(y_test, self.y_pred, average="weighted"),
            "f1": f1_score(y_test, self.y_pred, average="weighted"),
        }

        # zapis do pliku
        with open(os.path.join(self.metrics_dir, "metrics.txt"), "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v:.4f}\n")

        # classification report
        report = classification_report(y_test, self.y_pred)
        with open(os.path.join(self.metrics_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        return results

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", ax=ax, values_format='d')

        # większe liczby w komórkach
        for text in ax.texts:
            text.set_fontsize(10)

        # większe opisy osi
        ax.set_xlabel("Predicted label", fontsize=14)
        ax.set_ylabel("True label", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)

        plt.title("Confusion Matrix", fontsize=16)

        path = os.path.join(self.plots_dir, "confusion_matrix.png")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()

    def plot_roc_curve(self, X_test, y_test):
        # działa dla binarnej i multi-class (one-vs-rest)
        y_score = self.model.predict_proba(X_test)

        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)

        plt.figure()

        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        path = os.path.join(self.plots_dir, "roc_curve.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def run_all(self, X_test, y_test):
        results = self.evaluate(X_test, y_test)
        self.plot_confusion_matrix()

        return results

data_loader = DataLoader()
dataset = data_loader.load()
X_test, y_test = dataset['test']

model_path = 'SVM/best_model.pkl'
output_dir = 'SVM'

mlp = ModelEvaluator(model_path=model_path, output_dir=output_dir)
mlp.run_all(X_test, y_test)