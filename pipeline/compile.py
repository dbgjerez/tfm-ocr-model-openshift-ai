from kfp import compiler
from .pipeline import ocr_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ocr_pipeline,
        package_path="ocr_pipeline.yaml",
    )
    print("Generated: ocr_pipeline.yaml")
