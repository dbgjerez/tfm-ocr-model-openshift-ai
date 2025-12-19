from kfp import dsl

# ---------- ETAPA 1: DATOS ----------
@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311")
def datos():
    import os
    print("[DATOS] Hola desde Kubeflow / OpenShift AI")
    print("[DATOS] NB_PREFIX =", os.environ.get("NB_PREFIX"))
    print("[DATOS] KUBERNETES_SERVICE_HOST =", os.environ.get("KUBERNETES_SERVICE_HOST"))
    print("[DATOS] Done")


# ---------- ETAPA 2: ENTRENAMIENTO ----------
@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311")
def train():
    import os
    print("[TRAIN] Entrenamiento placeholder")
    print("[TRAIN] PYTHON_VERSION =", os.environ.get("PYTHON_VERSION"))
    print("[TRAIN] Done")


# ---------- ETAPA 3: VALIDACIÓN ----------
@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311")
def validate():
    print("[VALIDATE] Validación placeholder")
    print("[VALIDATE] Done")


# ---------- ETAPA 4: DESPLIEGUE ----------
@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311")
def deploy():
    print("[DEPLOY] Deploy placeholder")
    print("[DEPLOY] Done")


@dsl.pipeline(
    name="tfm-ocr-chars-pipeline",
    description="Pipeline OCR (datos -> train -> validate -> deploy) - placeholders"
)
def ocr_pipeline():
    s1 = datos()
    s2 = train().after(s1)
    s3 = validate().after(s2)
    deploy().after(s3)
