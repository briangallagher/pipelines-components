"""KFP Pipeline: RAG Ingest with Document Acquisition.

Three-step linear pipeline for M3+:
1. Acquire documents from source systems via Document Registry
2. Parse & chunk PDFs (Docling + HybridChunker via RayJob -> S3)
3. Ingest into Milvus (read chunks from S3, embed locally or via endpoint, insert)

No model deployment steps — embedding uses either a local sentence-transformers
model or a pre-existing endpoint. LLM deployment is handled separately (see the
full rag_multistep_pipeline for M4+).
"""

from kfp import dsl, kubernetes
from kfp_components.components.data_processing.acquire_documents import acquire_documents
from kfp_components.components.data_processing.ingest_to_milvus import ingest_to_milvus
from kfp_components.components.data_processing.parse_and_chunk import parse_and_chunk


@dsl.pipeline(
    name="RAG Ingest Pipeline",
    description=(
        "Data-chain-only RAG pipeline: parse & chunk PDFs with Docling "
        "(output to S3), then ingest into Milvus with local or remote embeddings. "
        "No model deployment steps."
    ),
)
def rag_ingest_pipeline(
    # Shared
    pvc_name: str = "data-pvc",
    pvc_mount_path: str = "/mnt/data",
    namespace: str = "ray-docling",
    # S3 (MinIO)
    s3_endpoint: str = "http://minio-service.default.svc.cluster.local:9000",
    s3_bucket: str = "rag-chunks",
    s3_prefix: str = "chunks",
    s3_staging_prefix: str = "staging",
    s3_secret_name: str = "minio-secret",
    # Acquisition (M3+)
    registry_url: str = "http://doc-registry:8080",
    connector_type: str = "s3",
    # PDF parsing
    input_path: str = "input/pdfs",
    ray_image: str = "quay.io/rhoai-szaher/docling-ray:latest",
    num_workers: int = 2,
    worker_cpus: int = 8,
    worker_memory_gb: int = 16,
    head_cpus: int = 2,
    head_memory_gb: int = 8,
    cpus_per_actor: int = 4,
    min_actors: int = 2,
    max_actors: int = 4,
    batch_size: int = 4,
    chunk_max_tokens: int = 256,
    num_files: int = 1000,
    timeout_seconds: int = 600,
    enable_profiling: bool = False,
    verbose: bool = True,
    bypass_kueue: bool = False,
    # Embedding (local model or pre-existing endpoint — no deployment)
    embedding_endpoint: str = "",
    embedding_model: str = "ibm-granite/granite-embedding-125m-english",
    embedding_dim: int = 768,
    # Milvus
    milvus_host: str = "milvus-milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    milvus_db: str = "default",
    milvus_token: str = "",
    collection_name: str = "rag_documents",
    drop_existing: bool = True,
    embed_batch_size: int = 64,
    milvus_batch_size: int = 256,
    # Traceability and metadata
    pipeline_run_id: str = "",
    doc_category: str = "",
    doc_subcategory: str = "",
    doc_date: str = "",
    index_type: str = "HNSW",
):
    # Step 1: Acquire documents from source systems via registry
    acquire_task = acquire_documents(
        registry_url=registry_url,
        collection_name=collection_name,
        connector_type=connector_type,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_staging_prefix=s3_staging_prefix,
        namespace=namespace,
        s3_secret_name=s3_secret_name,
        pipeline_run_id=pipeline_run_id,
    )
    acquire_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        acquire_task,
        secret_name=s3_secret_name,
        secret_key_to_env={
            "access_key": "S3_ACCESS_KEY",
            "secret_key": "S3_SECRET_KEY",
        },
    )
    kubernetes.use_config_map_as_env(
        acquire_task,
        config_map_name="data-strat-lineage-config",
        config_map_key_to_env={
            "OPENLINEAGE_URL": "OPENLINEAGE_URL",
            "MLFLOW_BRIDGE_ENABLED": "MLFLOW_BRIDGE_ENABLED",
        },
    )

    # Step 2: Parse & chunk PDFs → S3
    chunk_task = parse_and_chunk(
        pvc_name=pvc_name,
        pvc_mount_path=pvc_mount_path,
        input_path=input_path,
        ray_image=ray_image,
        namespace=namespace,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_secret_name=s3_secret_name,
        tokenizer=embedding_model,
        chunk_max_tokens=chunk_max_tokens,
        num_workers=num_workers,
        worker_cpus=worker_cpus,
        worker_memory_gb=worker_memory_gb,
        head_cpus=head_cpus,
        head_memory_gb=head_memory_gb,
        cpus_per_actor=cpus_per_actor,
        min_actors=min_actors,
        max_actors=max_actors,
        batch_size=batch_size,
        num_files=num_files,
        timeout_seconds=timeout_seconds,
        enable_profiling=enable_profiling,
        verbose=verbose,
        bypass_kueue=bypass_kueue,
        doc_category=doc_category,
        doc_subcategory=doc_subcategory,
        doc_date=doc_date,
        pipeline_run_id=pipeline_run_id,
        manifest_s3_key=acquire_task.output,
    )
    chunk_task.after(acquire_task)
    chunk_task.set_caching_options(False)
    kubernetes.use_config_map_as_env(
        chunk_task,
        config_map_name="data-strat-lineage-config",
        config_map_key_to_env={
            "OPENLINEAGE_URL": "OPENLINEAGE_URL",
            "MLFLOW_BRIDGE_ENABLED": "MLFLOW_BRIDGE_ENABLED",
        },
    )

    # Step 2: Ingest into Milvus (embed locally or via pre-existing endpoint)
    ingest_task = ingest_to_milvus(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        milvus_db=milvus_db,
        milvus_token=milvus_token,
        collection_name=collection_name,
        drop_existing=drop_existing,
        embedding_endpoint=embedding_endpoint,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embed_batch_size=embed_batch_size,
        milvus_batch_size=milvus_batch_size,
        pipeline_run_id=pipeline_run_id,
        index_type=index_type,
    )
    kubernetes.use_secret_as_env(
        ingest_task,
        secret_name=s3_secret_name,
        secret_key_to_env={
            "access_key": "S3_ACCESS_KEY",
            "secret_key": "S3_SECRET_KEY",
        },
    )
    ingest_task.after(chunk_task)
    ingest_task.set_caching_options(False)
    kubernetes.use_config_map_as_env(
        ingest_task,
        config_map_name="data-strat-lineage-config",
        config_map_key_to_env={
            "OPENLINEAGE_URL": "OPENLINEAGE_URL",
            "MLFLOW_BRIDGE_ENABLED": "MLFLOW_BRIDGE_ENABLED",
        },
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        rag_ingest_pipeline,
        package_path="rag_ingest_pipeline.yaml",
    )
    print("Pipeline compiled to rag_ingest_pipeline.yaml")
