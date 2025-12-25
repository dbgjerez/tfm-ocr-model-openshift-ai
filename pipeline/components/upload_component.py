from kfp import dsl
from kfp.dsl import Input, Model, OutputPath


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=[
        "boto3==1.35.30",
    ],
)
def upload_model_to_s3_component(
    model_artifact: Input[Model],
    s3_prefix: str,
    model_s3_uri: OutputPath(str),
):
    """
    Upload for ModelMesh/OVMS serving.

    It uploads ONLY the serving artifacts (by default: model.onnx, and optionally
    model_spec.json), and writes the exact ONNX URI into model_s3_uri.

    Requires env vars in the step container:
      - AWS_S3_ENDPOINT (recommended: internal svc URL, e.g. http://minio-service.<ns>.svc:9000)
      - AWS_S3_BUCKET
      - AWS_DEFAULT_REGION (recommended, e.g. us-east-1)
      - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

    Output:
      - model_s3_uri: s3://<bucket>/<s3_prefix>/model.onnx
    """
    import os
    from pathlib import Path

    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("AWS_S3_ENDPOINT")
    bucket = os.environ.get("AWS_S3_BUCKET")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if not endpoint:
        raise RuntimeError("Missing env var AWS_S3_ENDPOINT")
    if not bucket:
        raise RuntimeError("Missing env var AWS_S3_BUCKET")

    src_dir = Path(model_artifact.path)
    if not src_dir.exists():
        raise FileNotFoundError(f"model_artifact.path not found: {src_dir}")

    prefix = s3_prefix.strip("/")

    # --- Serving files we expect from TRAIN step ---
    onnx_path = src_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"model.onnx not found: {onnx_path.resolve()}\n"
            f"Contents in model_artifact.path: {[p.name for p in src_dir.iterdir()]}"
        )

    # Optional, useful for inference/decoding/traceability
    spec_path = src_dir / "model_spec.json"

    files_to_upload = [onnx_path]
    if spec_path.exists():
        files_to_upload.append(spec_path)

    # S3 client (MinIO/ODF/etc.)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )

    # Upload
    files_uploaded = 0
    uploaded_keys = []

    for p in files_to_upload:
        key = f"{prefix}/{p.name}"
        s3.upload_file(str(p), bucket, key)
        files_uploaded += 1
        uploaded_keys.append(key)

    # For ModelMesh deploy, you will use storage.path = "<s3_prefix>/model.onnx"
    uri = f"s3://{bucket}/{prefix}/{onnx_path.name}"
    Path(model_s3_uri).write_text(uri)

    print("[UPLOAD] endpoint:", endpoint)
    print("[UPLOAD] bucket:", bucket)
    print("[UPLOAD] region:", region)
    print("[UPLOAD] prefix:", prefix)
    print("[UPLOAD] files_uploaded:", files_uploaded)
    print("[UPLOAD] uploaded_keys:", uploaded_keys)
    print("[UPLOAD] model_s3_uri:", uri)
