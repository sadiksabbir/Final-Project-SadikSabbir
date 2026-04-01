import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Data loading
def load_data(file_path, label_col="label"):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found.")

    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    return X.values.astype(np.float32), y.values



# Dataset
class RamanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Model
class Raman1DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)

    return acc, macro_f1, cm, all_targets, all_preds


def main(args):
    X, y_raw = load_data(args.data_path, label_col=args.label_col)

    label_encoder = joblib.load(os.path.join(args.output_dir, "label_encoder.joblib"))
    scaler = joblib.load(os.path.join(args.output_dir, "scaler.joblib"))
    splits = np.load(os.path.join(args.output_dir, "splits.npz"))

    y = label_encoder.transform(y_raw)
    test_idx = splits["test_idx"]

    X_test = X[test_idx]
    y_test = y[test_idx]

    X_test = scaler.transform(X_test)

    test_ds = RamanDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model = Raman1DCNN(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    acc, macro_f1, cm, y_true, y_pred = evaluate(model, test_loader, device)

    class_names = list(label_encoder.classes_)

    print("Test Accuracy:", round(acc, 4))
    print("Test Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(args.output_dir, "test_confusion_matrix.csv"))

    pred_df = pd.DataFrame({
        "true_label": label_encoder.inverse_transform(y_true),
        "predicted_label": label_encoder.inverse_transform(y_pred)
    })
    pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(args.output_dir, "test_summary.json"), "w") as f:
        json.dump({
            "test_accuracy": float(acc),
            "test_macro_f1": float(macro_f1)
        }, f, indent=4)

    with open(os.path.join(args.output_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\nSaved test results to: {args.output_dir}")


if __name__ == "__main__":
    class Args:
        data_path = r"D:\onedrive okstate\Courses\Spring 26\DL\DL Final Project\Code\raman_spectra_api_compounds.csv"
        label_col = "label"
        output_dir = r"D:\onedrive okstate\Courses\Spring 26\DL\DL Final Project\Code\raman_output"
        batch_size = 32

    args = Args()
    main(args)