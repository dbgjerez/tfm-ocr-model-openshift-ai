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
    Sube el contenido de model_artifact.path a:
      s3://<AWS_S3_BUCKET>/<s3_prefix>/

    Requiere env vars (en el contenedor del step):
      - AWS_S3_ENDPOINT  (ej: https://minio-api-... o https://s3.openshift-storage.svc)
      - AWS_S3_BUCKET
      - AWS_DEFAULT_REGION (opcional)
    Y credenciales (habitualmente ya están en el pod vía Secret/SA):
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
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

    # Cliente S3 compatible (MinIO/ODF/etc.)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )

    # Subida recursiva
    files_uploaded = 0
    for p in src_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src_dir).as_posix()
        key = f"{prefix}/{rel}"
        s3.upload_file(str(p), bucket, key)
        files_uploaded += 1

    uri = f"s3://{bucket}/{prefix}"
    Path(model_s3_uri).write_text(uri)

    print("[UPLOAD] endpoint:", endpoint)
    print("[UPLOAD] bucket:", bucket)
    print("[UPLOAD] prefix:", prefix)
    print("[UPLOAD] files_uploaded:", files_uploaded)
    print("[UPLOAD] model_s3_uri:", uri)
