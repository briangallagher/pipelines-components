"""KFP component: Read chunks from S3, embed, and ingest into Milvus.

Reads JSONL chunk files from an S3-compatible bucket, generates embeddings
using either a deployed embedding service endpoint or a local
sentence-transformers model, and inserts the vectors into Milvus.
"""

from kfp import dsl
from kfp_components.utils.consts import RAY_RAG_BASE_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=RAY_RAG_BASE_IMAGE,
    packages_to_install=[
        "pymilvus>=2.4.0",
        "sentence-transformers>=2.2.0",
        "requests>=2.28.0",
        "boto3>=1.28.0",
        "rhoai-lineage @ git+https://github.com/briangallagher/rhoai-lineage.git",
    ],
)
def ingest_to_milvus(
    s3_endpoint: str,
    s3_bucket: str,
    milvus_host: str,
    s3_prefix: str = "chunks",
    embedding_endpoint: str = "",
    embedding_model: str = "ibm-granite/granite-embedding-125m-english",
    embedding_dim: int = 768,
    milvus_port: int = 19530,
    milvus_db: str = "default",
    milvus_token: str = "",
    collection_name: str = "rag_documents",
    drop_existing: bool = True,
    embed_batch_size: int = 64,
    milvus_batch_size: int = 256,
    pipeline_run_id: str = "",
    index_type: str = "HNSW",
) -> str:
    """Read chunks from S3, embed, and insert into Milvus.

    Args:
        s3_endpoint: S3-compatible endpoint URL (e.g. MinIO).
        s3_bucket: S3 bucket containing chunk files.
        milvus_host: Milvus service hostname.
        s3_prefix: Key prefix for chunk files in S3.
        embedding_endpoint: Optional embedding service URL. If empty,
            uses a local sentence-transformers model.
        embedding_model: Embedding model name (for API or local).
        embedding_dim: Dimension of the embedding vectors.
        milvus_port: Milvus gRPC port.
        milvus_db: Milvus database name.
        milvus_token: Milvus authentication token. Empty string for unauthenticated connections.
        collection_name: Milvus collection name.
        drop_existing: If True, drop and recreate the collection. If False, append to it.
        embed_batch_size: Batch size for embedding requests.
        milvus_batch_size: Batch size for Milvus inserts.

    Returns:
        The Milvus collection name and total vectors inserted.
    """
    import json
    import os
    import time

    import boto3
    import requests as req_lib
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

    # --- S3 client ---
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1",
    )

    def stream_chunks_from_s3():
        """Yield chunks one at a time from S3 JSONL files to avoid loading all into memory."""
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue
                resp = s3.get_object(Bucket=s3_bucket, Key=key)
                body = resp["Body"].read().decode("utf-8")
                for line in body.strip().split("\n"):
                    if line:
                        yield json.loads(line)

    # --- Setup Milvus collection ---
    uri = f"http://{milvus_host}:{milvus_port}"
    milvus_kwargs = {"uri": uri, "db_name": milvus_db}
    if milvus_token:
        milvus_kwargs["token"] = milvus_token
    client = MilvusClient(**milvus_kwargs)

    collection_exists = client.has_collection(collection_name)

    if collection_exists and not drop_existing:
        desc = client.describe_collection(collection_name)
        for field in desc.get("fields", []):
            if field.get("name") == "embedding":
                existing_dim = field.get("params", {}).get("dim")
                if existing_dim is not None and int(existing_dim) != embedding_dim:
                    raise ValueError(
                        f"Existing collection '{collection_name}' has dim={existing_dim}, "
                        f"but embedding_dim={embedding_dim}. Drop the collection or fix the dimension."
                    )
                break
        print(f"Appending to existing collection '{collection_name}'.")
    else:
        if collection_exists:
            print(f"Dropping existing collection '{collection_name}'")
            client.drop_collection(collection_name)

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="source_document_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="pipeline_run_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="subcategory", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="document_date", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=embedding_dim,
                ),
            ],
            description="RAG document chunks with traceability and metadata",
        )
        client.create_collection(collection_name=collection_name, schema=schema)

        index_params = client.prepare_index_params()
        if index_type == "HNSW":
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 256},
            )
        elif index_type == "IVF_FLAT":
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )
        else:
            raise ValueError(f"Unsupported index_type: {index_type}. Use 'HNSW' or 'IVF_FLAT'.")
        client.create_index(collection_name=collection_name, index_params=index_params)
        print(f"Collection '{collection_name}' created (dim={embedding_dim}, {index_type} index).")

    # --- Setup embedding ---
    use_endpoint = bool(embedding_endpoint)
    local_model = None

    if use_endpoint:
        print(f"Using embedding endpoint: {embedding_endpoint}")
    else:
        print(f"Using local embedding model: {embedding_model}")
        from sentence_transformers import SentenceTransformer

        local_model = SentenceTransformer(embedding_model)

    def _embed_and_insert(batch):
        texts = [c["text"] for c in batch]
        if use_endpoint:
            all_embeddings = []
            for j in range(0, len(texts), embed_batch_size):
                embed_batch = texts[j : j + embed_batch_size]
                resp = req_lib.post(
                    f"{embedding_endpoint}/v1/embeddings",
                    json={"model": embedding_model, "input": embed_batch},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                all_embeddings.extend([d["embedding"] for d in data])
            embeddings = all_embeddings
        else:
            embeddings = local_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=embed_batch_size,
            ).tolist()

        data = [
            {
                "source_file": c["source_file"],
                "source_document_id": c.get("source_document_id", c["source_file"]),
                "pipeline_run_id": pipeline_run_id or "unknown",
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "category": c.get("category", ""),
                "subcategory": c.get("subcategory", ""),
                "document_date": c.get("document_date", ""),
                "embedding": emb,
            }
            for c, emb in zip(batch, embeddings)
        ]
        client.insert(collection_name=collection_name, data=data)
        return len(data)

    # --- Embed and insert in streaming batches ---
    start_time = time.time()
    total_inserted = 0
    file_count = 0
    batch = []
    seen_files = set()

    for chunk in stream_chunks_from_s3():
        batch.append(chunk)
        src = chunk.get("source_file", "")
        if src not in seen_files:
            seen_files.add(src)
            file_count += 1

        if len(batch) < milvus_batch_size:
            continue

        total_inserted += _embed_and_insert(batch)
        batch = []

        if (total_inserted // milvus_batch_size) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Inserted {total_inserted} vectors ({file_count} files, {elapsed:.1f}s)")

    if batch:
        total_inserted += _embed_and_insert(batch)

    if total_inserted == 0:
        raise FileNotFoundError(f"No chunks found in s3://{s3_bucket}/{s3_prefix}/")

    # Load collection for searching
    client.load_collection(collection_name)
    stats = client.get_collection_stats(collection_name)

    wall_clock = time.time() - start_time
    print(f"\nIngestion complete: {total_inserted} vectors in {wall_clock:.1f}s")
    print(f"Collection stats: {stats}")

    # --- OpenLineage emission (best-effort) ---
    try:
        from rhoai_lineage.kfp.lineage import kfp_lineage
        from rhoai_lineage.naming import s3_dataset, milvus_dataset
        from urllib.parse import urlparse as _urlparse

        _s3_parsed = _urlparse(s3_endpoint)
        _s3_host = _s3_parsed.hostname or "minio"
        _s3_port = _s3_parsed.port or 9000

        input_ns, input_name = s3_dataset(
            bucket=s3_bucket, path=s3_prefix, host=_s3_host, port=_s3_port,
        )
        input_ds = {"namespace": input_ns, "name": input_name}

        output_ns, output_name = milvus_dataset(
            collection=collection_name, host=milvus_host, port=milvus_port,
        )
        output_ds = {
            "namespace": output_ns,
            "name": output_name,
            "facets": {
                "custom_metrics": {
                    "_producer": "https://github.com/rhoai-lineage",
                    "_schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/CustomFacet",
                    "vectors_inserted": total_inserted,
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                    "collection_name": collection_name,
                    "index_type": index_type,
                    "duration_seconds": round(wall_clock, 2),
                },
            },
        }

        ol_run_facets = {}
        if pipeline_run_id:
            ol_run_facets["pipelineRunId"] = {
                "_producer": "rhoai-lineage",
                "_schemaURL": "https://openlineage.io/spec/1-0-0/OpenLineage.json",
                "id": pipeline_run_id,
            }

        with kfp_lineage(
            "ingest_to_milvus",
            inputs=[input_ds],
            outputs=[output_ds],
            run_facets=ol_run_facets,
        ):
            pass
        print("OpenLineage COMPLETE event emitted for ingest_to_milvus")
    except Exception as e:
        print(f"WARNING: OpenLineage emission failed (non-fatal): {e}")

    # --- MLflow tracking via REST API (best-effort) ---
    # Direct REST calls with SA token auth — the RHOAI MLflow Operator
    # requires Authorization + X-Mlflow-Workspace headers that the vanilla
    # mlflow client doesn't inject from KFP pods.
    try:
        class _MLflowRESTTracker:
            def __init__(self):
                self._url = "https://mlflow.redhat-ods-applications.svc:8443"
                self._headers = {"Content-Type": "application/json"}
                self._run_id = None
                self._experiment_id = None
                _sa_token = "/var/run/secrets/kubernetes.io/serviceaccount/token"
                _sa_ns = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
                if os.path.exists(_sa_token):
                    with open(_sa_token) as f:
                        self._headers["Authorization"] = f"Bearer {f.read().strip()}"
                if os.path.exists(_sa_ns):
                    with open(_sa_ns) as f:
                        self._headers["X-Mlflow-Workspace"] = f.read().strip()

            def _post(self, endpoint, data):
                resp = req_lib.post(f"{self._url}{endpoint}", json=data,
                                    headers=self._headers, verify=False, timeout=10)
                resp.raise_for_status()
                return resp.json()

            def create_experiment(self, name):
                try:
                    result = self._post("/api/2.0/mlflow/experiments/create", {"name": name})
                    self._experiment_id = result.get("experiment_id")
                except Exception:
                    resp = req_lib.get(f"{self._url}/api/2.0/mlflow/experiments/get-by-name",
                                       params={"experiment_name": name},
                                       headers=self._headers, verify=False, timeout=10)
                    if resp.ok:
                        self._experiment_id = resp.json().get("experiment", {}).get("experiment_id")
                return self._experiment_id

            def start_run(self, run_name=""):
                if not self._experiment_id:
                    return None
                result = self._post("/api/2.0/mlflow/runs/create", {
                    "experiment_id": self._experiment_id, "run_name": run_name,
                })
                self._run_id = result.get("run", {}).get("info", {}).get("run_id")
                return self._run_id

            def create_parent_run(self, run_name, tags=None):
                """Create a parent run (pipeline-level)."""
                body = {
                    "experiment_id": self._experiment_id,
                    "run_name": run_name,
                }
                if tags:
                    body["tags"] = [{"key": k, "value": str(v)} for k, v in tags.items()]
                result = self._post("/api/2.0/mlflow/runs/create", body)
                return result.get("run", {}).get("info", {}).get("run_id")

            def create_nested_run(self, run_name, parent_run_id):
                """Create a nested run under a parent."""
                result = self._post("/api/2.0/mlflow/runs/create", {
                    "experiment_id": self._experiment_id,
                    "run_name": run_name,
                    "tags": [
                        {"key": "mlflow.parentRunId", "value": parent_run_id},
                    ],
                })
                self._run_id = result.get("run", {}).get("info", {}).get("run_id")
                return self._run_id

            def find_run_by_name(self, run_name):
                """Find a run by name in the current experiment."""
                resp = req_lib.post(
                    f"{self._url}/api/2.0/mlflow/runs/search",
                    json={
                        "experiment_ids": [self._experiment_id],
                        "filter": f"tags.`mlflow.runName` = '{run_name}'",
                        "max_results": 1,
                    },
                    headers=self._headers, verify=False, timeout=10,
                )
                if resp.ok:
                    runs = resp.json().get("runs", [])
                    if runs:
                        return runs[0].get("info", {}).get("run_id")
                return None

            def log_param(self, key, value):
                if not self._run_id:
                    return
                self._post("/api/2.0/mlflow/runs/log-parameter", {
                    "run_id": self._run_id, "key": key, "value": str(value),
                })

            def log_metric(self, key, value):
                if not self._run_id:
                    return
                self._post("/api/2.0/mlflow/runs/log-metric", {
                    "run_id": self._run_id, "key": key, "value": float(value),
                    "timestamp": int(time.time() * 1000),
                })

            def end_run(self, status="FINISHED", run_id=None):
                """End a specific run (or the current run)."""
                target = run_id or self._run_id
                if not target:
                    return
                self._post("/api/2.0/mlflow/runs/update", {
                    "run_id": target, "status": status,
                    "end_time": int(time.time() * 1000),
                })

        tracker = _MLflowRESTTracker()
        tracker.create_experiment("data-strat-ingest")

        parent_id = tracker.find_run_by_name(pipeline_run_id or "unknown")

        if parent_id:
            tracker.create_nested_run("ingest_to_milvus", parent_id)
        else:
            tracker.start_run(run_name=f"ingest-{collection_name}")
            print("MLflow: parent run not found, creating standalone run")

        tracker.log_param("pipeline_run_id", pipeline_run_id)
        tracker.log_param("collection_name", collection_name)
        tracker.log_param("embedding_model", embedding_model)
        tracker.log_param("embedding_dim", str(embedding_dim))
        tracker.log_param("index_type", index_type)
        tracker.log_param("milvus_host", milvus_host)
        tracker.log_param("s3_bucket", s3_bucket)
        tracker.log_param("s3_prefix", s3_prefix)
        tracker.log_param("embed_batch_size", str(embed_batch_size))
        tracker.log_param("milvus_batch_size", str(milvus_batch_size))
        tracker.log_param("drop_existing", str(drop_existing))
        tracker.log_metric("vectors_inserted", float(total_inserted))
        tracker.log_metric("documents_processed", float(file_count))
        tracker.log_metric("duration_seconds", wall_clock)
        tracker.log_metric("vectors_per_second", float(total_inserted / max(wall_clock, 0.1)))
        tracker.end_run()

        if parent_id:
            tracker.end_run(status="FINISHED", run_id=parent_id)
            print(f"MLflow: logged ingest_to_milvus + closed parent run {pipeline_run_id}")
        else:
            print("MLflow: logged ingest_to_milvus as standalone run")
    except Exception as e:
        print(f"MLflow tracking failed (non-blocking): {e}")

    return f"{collection_name}:{total_inserted}"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        ingest_to_milvus,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
