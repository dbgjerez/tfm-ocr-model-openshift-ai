from kfp import dsl


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=["kubernetes==30.1.0"],
)
def deploy_component(
    model_s3_uri: str,
    runtime_name: str = "ocr-rest-runtime",
    isvc_name: str = "ocr-rest",
    namespace: str = "",
    storage_secret_name: str = "ocr-s3-creds",
    service_account: str = "default",
):
    """
    Crea/actualiza un InferenceService (KServe) para servir el modelo desde S3.

    - model_s3_uri: "s3://bucket/prefix"
    - runtime_name: ServingRuntime existente (custom runtime / modelmesh runtime, etc.)
    - storage_secret_name: Secret con credenciales/endpoint para S3 (según tu instalación)
    """
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    # Dentro del cluster
    config.load_incluster_config()

    core = client.CoreV1Api()
    co_api = client.CustomObjectsApi()

    if not namespace:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            namespace = f.read().strip()

    print("[DEPLOY] namespace:", namespace)
    print("[DEPLOY] storageUri:", model_s3_uri)
    print("[DEPLOY] runtime:", runtime_name)
    print("[DEPLOY] isvc:", isvc_name)

    # Verifica Secret
    try:
        core.read_namespaced_secret(storage_secret_name, namespace)
        print(f"[DEPLOY] Secret OK: {storage_secret_name}")
    except ApiException as e:
        raise RuntimeError(
            f"[DEPLOY] Secret not found: {storage_secret_name} in ns {namespace}. Error: {e}"
        )

    # InferenceService
    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": namespace,
            "annotations": {
                # OJO: esta annotation es la más común para S3 creds en KServe
                "serving.kserve.io/storageSecretName": storage_secret_name,
            },
        },
        "spec": {
            "predictor": {
                "serviceAccountName": service_account,
                "model": {
                    "modelFormat": {"name": "ocr-rest"},
                    "runtime": runtime_name,
                    "storageUri": model_s3_uri,
                }
            }
        },
    }

    group = "serving.kserve.io"
    version = "v1beta1"
    plural = "inferenceservices"

    # Upsert create/patch
    try:
        co_api.get_namespaced_custom_object(group, version, namespace, plural, isvc_name)
        print("[DEPLOY] InferenceService exists, patching...")
        co_api.patch_namespaced_custom_object(group, version, namespace, plural, isvc_name, isvc)
        print("[DEPLOY] Patched InferenceService:", isvc_name)
    except ApiException as e:
        if e.status == 404:
            print("[DEPLOY] InferenceService not found, creating...")
            co_api.create_namespaced_custom_object(group, version, namespace, plural, isvc)
            print("[DEPLOY] Created InferenceService:", isvc_name)
        else:
            raise

    print("[DEPLOY] Done. Wait until READY (oc get isvc -n <ns>).")
