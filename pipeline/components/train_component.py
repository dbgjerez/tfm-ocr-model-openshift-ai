from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    # Si luego vas a usar GPU, cambia a una imagen con CUDA. De momento CPU.
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=[
        "pandas==2.2.2",
        "pillow==10.4.0",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "onnx==1.16.2",
        "onnxruntime==1.19.2",  
    ],
)
def train_component(
    repo: Input[Dataset],
    datos_artifacts: Input[Dataset],
    dataset_rel_dir: str = "data/english/fnt",
    csv_name: str = "chars74k_labels.csv",
    epochs: int = 3,
    batch_size: int = 64,
    val_split: float = 0.2,
    lr: float = 1e-3,
    model_artifact: Output[Model] = None,
):
    """
    Step TRAIN:
      - Lee CSV generado en datos_artifacts
      - Carga imágenes desde repo + dataset_rel_dir
      - Entrena CNN simple
      - Guarda modelo y artefactos en model_artifact.path
    """
    import json
    import os
    from dataclasses import asdict, dataclass
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader, random_split

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # ----------------------------
    # Paths
    # ----------------------------
    repo_root = Path(repo.path)
    ds_root = repo_root / dataset_rel_dir
    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset dir not found: {ds_root.resolve()}")

    datos_dir = Path(datos_artifacts.path)
    csv_path = datos_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found in datos_artifacts: {csv_path.resolve()}")

    mapping_path = datos_dir / "mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.json not found in datos_artifacts: {mapping_path.resolve()}")

    out_dir = Path(model_artifact.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eda").mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load mapping
    # ----------------------------
    mapping = json.loads(mapping_path.read_text())
    char_to_label = mapping["char_to_label"]
    label_to_char = {int(k): v for k, v in mapping["label_to_char"].items()}  # keys can come as strings

    num_classes = len(char_to_label)

    # ----------------------------
    # Transform (igual que tu notebook)
    # ----------------------------
    transform_ocr = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # ----------------------------
    # Dataset (igual que tu CharsDataset)
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
            img_path = self.root / row["path"]  # path relativo tipo SampleXXX/img.png
            label_char = row["label"]
            label_idx = int(char_to_label[label_char])

            img = Image.open(img_path).convert("L")
            if self.transform:
                img = self.transform(img)

            return img, label_idx

    # ----------------------------
    # Model
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

    # ----------------------------
    # Helpers
    # ----------------------------
    def train_one_epoch(model, loader, device, opt, crit):
        model.train()
        total_loss, correct = 0.0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            correct += int((out.argmax(1) == labels).sum().item())

        return total_loss / max(1, len(loader)), correct / max(1, len(loader.dataset))

    def eval_one_epoch(model, loader, device, crit):
        model.eval()
        total_loss, correct = 0.0, 0
        preds, labels_all = [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = crit(out, labels)

                total_loss += float(loss.item())
                correct += int((out.argmax(1) == labels).sum().item())

                preds.extend(out.argmax(1).cpu().numpy().tolist())
                labels_all.extend(labels.cpu().numpy().tolist())

        return total_loss / max(1, len(loader)), correct / max(1, len(loader.dataset)), labels_all, preds

    def save_confusion_matrix(labels, preds, path: Path):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(csv_path)
    dataset = CharsDataset(df, root_dir=ds_root, transform=transform_ocr)

    val_len = int(len(dataset) * float(val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ----------------------------
    # Train
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(num_classes=num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(lr))

    history = []
    last_val_acc = 0.0

    for ep in range(int(epochs)):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, opt, crit)
        val_loss, val_acc, y_true, y_pred = eval_one_epoch(model, val_loader, device, crit)

        last_val_acc = float(val_acc)
        history.append(
            {
                "epoch": ep,
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

        print(f"[Epoch {ep}] Train acc={tr_acc:.3f} | Val acc={val_acc:.3f}")

    # Confusion matrix (último epoch)
    cm_path = save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    # ----------------------------
    # Save model + metadata (source-of-truth + serving artifact)
    # ----------------------------
    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Spec para reconstruir el modelo (ya lo usas en validate)
    spec = {
        "arch": "SimpleCNN",
        "num_classes": int(num_classes),
        "input_shape": [1, 32, 32],
        "normalization": {"mean": [0.5], "std": [0.5]},
        "label_to_char": label_to_char,
        "char_to_label": char_to_label,
        # Opcional: info útil para serving
        "serving": {
            "format": "onnx",
            "input_name": "input",
            "output_name": "logits",
            "opset": 17,
            "dynamic_batch": True,
        },
    }

    spec_path = out_dir / "model_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    metrics = {
        "val_acc": float(last_val_acc),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "val_split": float(val_split),
        "lr": float(lr),
    }
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    history_path.write_text(json.dumps(history, indent=2))

    # ----------------------------
    # Export ONNX (NO retraining)
    #   - ModelMesh/OVMS no puede cargar un .pt (state_dict)
    #   - Exportamos ONNX desde los pesos entrenados
    # ----------------------------
    onnx_path = out_dir / "model.onnx"

    # Export en CPU para evitar dependencias/GPU en serving
    model_cpu = SimpleCNN(num_classes=num_classes).to("cpu")
    state_cpu = torch.load(model_path, map_location="cpu", weights_only=True)
    model_cpu.load_state_dict(state_cpu)
    model_cpu.eval()

    dummy = torch.randn(1, 1, 32, 32, device="cpu")  # input_shape
    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    if not onnx_path.exists():
        raise RuntimeError("ONNX export failed: model.onnx was not created")

    # ----------------------------
    # KFP artifact metadata
    # ----------------------------
    model_artifact.metadata["framework"] = "pytorch"
    model_artifact.metadata["arch"] = "SimpleCNN"
    model_artifact.metadata["val_acc"] = float(last_val_acc)
    model_artifact.metadata["num_classes"] = int(num_classes)
    model_artifact.metadata["exported_formats"] = ["pt_state_dict", "onnx"]
    model_artifact.metadata["onnx_file"] = onnx_path.name
    model_artifact.metadata["onnx_opset"] = 17
    model_artifact.metadata["input_shape"] = [1, 32, 32]

    # ----------------------------
    # Logs
    # ----------------------------
    print("[TRAIN] wrote:", model_path)
    print("[TRAIN] wrote:", onnx_path)
    print("[TRAIN] wrote:", cm_path)
    print("[TRAIN] wrote:", spec_path)
    print("[TRAIN] wrote:", metrics_path)
    print("[TRAIN] wrote:", history_path)
    print("[TRAIN] Done.")
