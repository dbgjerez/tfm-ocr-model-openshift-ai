from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=[
        "pandas==2.2.2",
        "pillow==10.4.0",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "numpy==2.0.2",
    ],
)
def validate_component(
    repo: Input[Dataset],
    datos_artifacts: Input[Dataset],
    model_artifact: Input[Model],
    validation_metrics: Output[Metrics],
    validation_artifacts: Output[Dataset],
    # Defaults
    dataset_rel_dir: str = "data/english/fnt",
    csv_name: str = "chars74k_labels.csv",
    repetitions: int = 50,
    warmup: int = 10,
    min_accuracy: float = 0.90,
) -> str:
    """
    Returns:
      "true" if accuracy >= min_accuracy else "false"
    """
    import json
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import Dataset

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        recall_score,
        precision_score,
        confusion_matrix,
    )

    # ----------------------------
    # Paths
    # ----------------------------
    repo_root = Path(repo.path)
    ds_root = repo_root / dataset_rel_dir
    if not ds_root.exists():
        raise FileNotFoundError(
            f"Dataset dir not found: {ds_root.resolve()}\n"
            f"Repo root contents: {[p.name for p in repo_root.iterdir()]}"
        )

    datos_dir = Path(datos_artifacts.path)
    csv_path = datos_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path.resolve()}\n"
            f"Datos dir contents: {[p.name for p in datos_dir.iterdir()]}"
        )

    mapping_path = datos_dir / "mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.json not found: {mapping_path.resolve()}")

    model_dir = Path(model_artifact.path)
    model_pt = model_dir / "model.pt"
    model_spec_path = model_dir / "model_spec.json"
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found: {model_pt.resolve()}")
    if not model_spec_path.exists():
        raise FileNotFoundError(f"model_spec.json not found: {model_spec_path.resolve()}")

    out_dir = Path(validation_artifacts.path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load mappings/spec
    # ----------------------------
    mapping = json.loads(mapping_path.read_text())
    char_to_label = mapping["char_to_label"]

    spec = json.loads(model_spec_path.read_text())
    num_classes = int(spec["num_classes"])

    # ----------------------------
    # Rebuild model (same as train)
    # ----------------------------
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Recomendado por warning de PyTorch
    state = torch.load(model_pt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ----------------------------
    # Transform
    # ----------------------------
    transform_ocr = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # ----------------------------
    # Dataset
    # ----------------------------
    class CharsDataset(Dataset):
        def __init__(self, df: pd.DataFrame, root_dir: Path, transform=None):
            self.df = df.reset_index(drop=True)
            self.root = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = self.root / row["path"]
            label = row["label"]  # char

            if isinstance(label, str):
                label = int(char_to_label[label])
            else:
                label = int(label)

            img = Image.open(img_path).convert("L")
            if self.transform:
                img = self.transform(img)
            return img, label

    df = pd.read_csv(csv_path)
    dataset = CharsDataset(df, root_dir=ds_root, transform=transform_ocr)

    # ----------------------------
    # Inference
    # ----------------------------
    imgs, labels = [], []
    for img, lab in dataset:
        imgs.append(img.numpy())
        labels.append(int(lab))

    x = torch.tensor(np.array(imgs), dtype=torch.float32).to(device)
    y_true = np.array(labels, dtype=np.int64)

    with torch.no_grad():
        logits = model(x)
        y_pred = logits.argmax(1).cpu().numpy()

    # ----------------------------
    # Performance metrics
    # ----------------------------
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    rec = float(recall_score(y_true, y_pred, average="macro"))
    prec = float(precision_score(y_true, y_pred, average="macro"))

    print(f"[VALIDATE] acc={acc:.4f} f1={f1:.4f} recall={rec:.4f} precision={prec:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # ----------------------------
    # Operational: latency
    # ----------------------------
    dummy = torch.randn(1, 1, 32, 32, device=device)

    with torch.no_grad():
        for _ in range(int(warmup)):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(int(repetitions)):
            t0 = time.time()
            _ = model(dummy)
            times.append(time.time() - t0)

    latency_mean_ms = float(np.mean(times) * 1000.0)
    latency_p95_ms = float(np.percentile(times, 95) * 1000.0)
    latency_p99_ms = float(np.percentile(times, 99) * 1000.0)

    model_size_mb = float(model_pt.stat().st_size / (1024 * 1024))

    report = {
        "performance": {
            "accuracy": acc,
            "f1_macro": f1,
            "recall_macro": rec,
            "precision_macro": prec,
        },
        "operational": {
            "latency_mean_ms": latency_mean_ms,
            "latency_p95_ms": latency_p95_ms,
            "latency_p99_ms": latency_p99_ms,
            "model_size_mb": model_size_mb,
            "device": device,
        },
        "inputs": {
            "dataset_rel_dir": dataset_rel_dir,
            "csv_name": csv_name,
            "num_samples": int(len(dataset)),
        },
        "model": {"num_classes": num_classes},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # KFP metrics (UI)
    validation_metrics.log_metric("val_accuracy", acc)
    validation_metrics.log_metric("val_f1_macro", f1)
    validation_metrics.log_metric("val_recall_macro", rec)
    validation_metrics.log_metric("val_precision_macro", prec)
    validation_metrics.log_metric("latency_mean_ms", latency_mean_ms)
    validation_metrics.log_metric("latency_p95_ms", latency_p95_ms)
    validation_metrics.log_metric("latency_p99_ms", latency_p99_ms)
    validation_metrics.log_metric("model_size_mb", model_size_mb)

    is_passed = "true" if acc >= float(min_accuracy) else "false"
    print("[VALIDATE] passed:", is_passed)
    return is_passed
