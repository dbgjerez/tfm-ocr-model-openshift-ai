from kfp import dsl

from pipeline.components.git_clone_component import git_clone_component
from pipeline.components.data_component import datos_component
from pipeline.components.train_component import train_component
from pipeline.components.validate_component import validate_component
from pipeline.components.upload_component import upload_model_to_s3_component
from pipeline.components.deploy_component import deploy_component


@dsl.pipeline(name="tfm-ocr-chars")
def ocr_pipeline(
    repo_url: str = "https://github.com/dbgjerez/tfm-ocr-model-openshift-ai.git",
    branch: str = "main",
    dataset_rel_dir: str = "data/english/fnt",
    min_accuracy: float = 0.90,
    # --- S3 destino para Serving (KServe) ---
    s3_endpoint: str = "",     # ej: https://minio-api-<ns>.apps....
    s3_bucket: str = "",       # ej: ocr-model
    s3_region: str = "us-east-1",
    s3_prefix: str = "models/ocr-rest/latest",
    # --- KServe / Serving ---
    runtime_name: str = "ocr-rest-runtime",
    isvc_name: str = "ocr-rest",
    namespace: str = "",       # vacío => autodetect en runtime
    storage_secret_name: str = "ocr-s3-creds",
    service_account: str = "default",
):
    # 1) CLONE
    repo_task = git_clone_component(repo_url=repo_url, branch=branch)

    # 2) DATOS
    datos_task = datos_component(
        repo=repo_task.outputs["repo"],
        dataset_rel_dir=dataset_rel_dir,
    )

    # 3) TRAIN
    train_task = train_component(
        repo=repo_task.outputs["repo"],
        datos_artifacts=datos_task.outputs["datos_artifacts"],
        dataset_rel_dir=dataset_rel_dir,
    )

    # 4) VALIDATE (devuelve str: "true"/"false")
    validate_task = validate_component(
        repo=repo_task.outputs["repo"],
        datos_artifacts=datos_task.outputs["datos_artifacts"],
        model_artifact=train_task.outputs["model_artifact"],
        dataset_rel_dir=dataset_rel_dir,
        min_accuracy=min_accuracy,
    )

    # Condicional: sólo si valida OK
    with dsl.If(validate_task.outputs["Output"] == "true"):
        # 5) UPLOAD a S3 (de cara a Serving)
        upload_task = upload_model_to_s3_component(
            model_artifact=train_task.outputs["model_artifact"],
            s3_prefix=s3_prefix,
        )

        # 6) DEPLOY InferenceService apuntando a ese storageUri (S3)
        deploy_component(
            model_s3_uri=upload_task.outputs["model_s3_uri"],
            namespace=namespace,
            storage_secret_name=storage_secret_name,
            runtime_name=runtime_name,
            isvc_name=isvc_name,
            service_account=service_account,
        )
