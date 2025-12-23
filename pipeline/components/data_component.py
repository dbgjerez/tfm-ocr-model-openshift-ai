from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=[
        "pandas==2.2.2",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "pillow==10.4.0",
    ],
)
def datos_component(
    repo: Input[Dataset],
    dataset_rel_dir: str = "data/english/fnt",
    csv_name: str = "chars74k_labels.csv",
    dataset_version: str = "v1",
    datos_artifacts: Output[Dataset] = None,
):
    """
    Step DATOS:
      - Lee dataset desde <repo.path>/<dataset_rel_dir>
      - Genera CSV + mapping/version + plot EDA
      - Escribe todo en datos_artifacts.path (artifact KFP)
    """
    import os
    import json
    import string
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    out_dir = Path(datos_artifacts.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Config & mappings (igual que tu MLflow)
    # ----------------------------
    digits = "0123456789"
    uppercase = string.ascii_uppercase

    char_to_label = {d: i for i, d in enumerate(digits)}
    for i, ch in enumerate(uppercase, start=10):
        char_to_label[ch] = i

    label_to_char = {v: k for k, v in char_to_label.items()}
    num_classes = len(char_to_label)

    # ----------------------------
    # 1) Recolección
    # ----------------------------
    rows = []
    samples = sorted(os.listdir(ds_root))

    for sample in samples:
        if not sample.startswith("Sample"):
            continue

        try:
            num = int(sample.replace("Sample", ""))
        except ValueError:
            continue

        if 1 <= num <= 10:
            label = str(num - 1)
        elif 11 <= num <= 36:
            label = chr(ord("A") + (num - 11))
        else:
            continue

        folder = ds_root / sample
        if not folder.is_dir():
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith(".png"):
                rows.append(
                    {
                        "path": f"{sample}/{fname}",  # path relativo a dataset_rel_dir
                        "label": label,
                    }
                )

    df_raw = pd.DataFrame(rows)

    # ----------------------------
    # 2) Limpieza (placeholder)
    # ----------------------------
    df_clean = df_raw.copy()

    # ----------------------------
    # 3) Transformación / Enriquecimiento
    # ----------------------------
    mapping_path = out_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps({"char_to_label": char_to_label, "label_to_char": label_to_char}, indent=4)
    )

    # ----------------------------
    # 4) EDA
    # ----------------------------
    if "label" not in df_clean.columns:
        raise ValueError("Expected column 'label' not found in dataframe")

    plt.figure(figsize=(12, 4))
    sns.countplot(data=df_clean, x="label", order=sorted(df_clean["label"].unique()))
    plt.title("Distribución de clases en Chars74K (digits + uppercase)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plot_path = eda_dir / "class_distribution.png"
    plt.savefig(plot_path)
    plt.close()

    # ----------------------------
    # 5) Versionado
    # ----------------------------
    version_file = out_dir / "version.json"
    version_file.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "dataset_rel_dir": dataset_rel_dir,
                "num_samples": int(len(df_clean)),
                "num_classes": int(num_classes),
            },
            indent=4,
        )
    )

    # CSV final
    csv_path = out_dir / csv_name
    df_clean.to_csv(csv_path, index=False)

    # Logs
    print("[DATOS] repo.path:", repo_root)
    print("[DATOS] dataset:", ds_root)
    print("[DATOS] num_samples:", len(df_clean))
    print("[DATOS] wrote:", csv_path)
    print("[DATOS] wrote:", mapping_path)
    print("[DATOS] wrote:", version_file)
    print("[DATOS] wrote:", plot_path)
