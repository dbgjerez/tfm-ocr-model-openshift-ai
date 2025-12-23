from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=["gitpython==3.1.43"],
)
def git_clone_component(
    repo_url: str,
    branch: str = "main",
    repo: Output[Dataset] = None,
):
    """
    Clona un repo Git en repo.path (directorio gestionado por KFP).
    KFP subirá este artifact a S3/ODF y lo hará accesible en steps posteriores.
    """
    from pathlib import Path
    from git import Repo

    out = Path(repo.path)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[GIT] Cloning {repo_url}@{branch} -> {out}")
    Repo.clone_from(repo_url, out, branch=branch, depth=1)

    print("[GIT] Done. Top-level:", [p.name for p in out.iterdir()])
