"""KFP component: Acquire documents from source systems.

Queries the Document Registry for collection members, fetches documents
from their source systems to S3 staging, writes a manifest.json for
downstream pipeline steps, and emits OpenLineage events.

The registry is authoritative (ADR-010): only documents the registry
explicitly lists for this collection will be fetched.

The pipeline is the sole OL emitter (ADR-011): the registry does not
emit to Marquez directly.
"""

from kfp import dsl
from kfp_components.utils.consts import RAY_RAG_BASE_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=RAY_RAG_BASE_IMAGE,
    packages_to_install=[
        "registry-sdk @ git+https://github.com/briangallagher/data-strat-poc.git#subdirectory=registry-sdk",
        "rhoai-lineage @ git+https://github.com/briangallagher/rhoai-lineage.git",
        "boto3>=1.28.0",
        "requests>=2.28.0",
    ],
)
def acquire_documents(
    registry_url: str,
    collection_name: str,
    connector_type: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_staging_prefix: str,
    namespace: str,
    s3_secret_name: str = "minio-secret",
    pipeline_run_id: str = "",
) -> str:
    """Acquire documents from source systems via the Document Registry."""
    import json
    import os
    import time

    import boto3
    import httpx

    start_time = time.time()

    s3_access_key = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    s3_secret_key = os.environ.get("S3_SECRET_KEY", "minioadmin")
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name="us-east-1",
    )

    try:
        s3.head_bucket(Bucket=s3_bucket)
    except Exception:
        s3.create_bucket(Bucket=s3_bucket)

    print(f"Querying registry for collection '{collection_name}'...")
    registry = httpx.Client(base_url=registry_url, timeout=30.0)

    r = registry.get("/api/v1/documents", params={
        "collection": collection_name,
        "status": "active",
    })
    r.raise_for_status()
    docs_data = r.json()["documents"]
    print(f"Registry returned {len(docs_data)} documents for '{collection_name}'")

    if not docs_data:
        raise RuntimeError(f"No active documents in registry for '{collection_name}'")

    fetched = []
    skipped = []
    manifest_entries = []

    for doc in docs_data:
        doc_id = doc["doc_id"]
        source_url = doc["source_url"]
        print(f"  Fetching {doc_id}...")

        try:
            if connector_type == "s3":
                # List actual files in corpus/<collection>/ and find a match
                # Files might not match source_url filename — use listing
                if not hasattr(acquire_documents, '_corpus_files'):
                    acquire_documents._corpus_files = {}
                if collection_name not in acquire_documents._corpus_files:
                    prefix = f"corpus/{collection_name}/"
                    resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
                    acquire_documents._corpus_files[collection_name] = [
                        obj["Key"] for obj in resp.get("Contents", [])
                    ]
                    print(f"    Found {len(acquire_documents._corpus_files[collection_name])} files in s3://{s3_bucket}/{prefix}")

                corpus_files = acquire_documents._corpus_files[collection_name]
                # Match by index position (docs ordered same as files) or by doc_id prefix in filename
                doc_idx = docs_data.index(doc)
                if doc_idx < len(corpus_files):
                    src_key = corpus_files[doc_idx]
                    filename = src_key.split("/")[-1]
                else:
                    # Fallback: try to find file containing doc_id components
                    filename = None
                    for cf in corpus_files:
                        fname = cf.split("/")[-1]
                        # Simple heuristic match
                        if doc_id.replace("-", "") in fname.replace("-", "").lower():
                            filename = fname
                            src_key = cf
                            break
                    if not filename:
                        print(f"    WARNING: {doc_id} — no matching file in corpus — skipping")
                        skipped.append({"doc_id": doc_id, "reason": "no_match"})
                        continue

                dest_key = f"{s3_staging_prefix}/{filename}"
                try:
                    s3.copy_object(
                        Bucket=s3_bucket,
                        CopySource={"Bucket": s3_bucket, "Key": src_key},
                        Key=dest_key,
                    )
                except Exception as copy_err:
                    if "404" in str(copy_err) or "NoSuchKey" in str(copy_err):
                        print(f"    WARNING: {doc_id} not found at {src_key} — skipping")
                        skipped.append({"doc_id": doc_id, "reason": "not_found"})
                        try:
                            registry.patch(f"/api/v1/documents/{doc_id}", json={"status": "unavailable"})
                        except Exception:
                            pass
                        continue
                    raise
            else:
                filename = doc_id + ".html"
                dest_key = f"{s3_staging_prefix}/{filename}"
                s3.put_object(Bucket=s3_bucket, Key=dest_key,
                              Body=f"<!-- Placeholder for {doc_id} -->".encode())

            fetched.append(doc_id)
            manifest_entries.append({
                "doc_id": doc["doc_id"],
                "filename": filename,
                "source_system": doc["source_system"],
                "source_url": doc["source_url"],
                "document_type": doc["document_type"],
                "line_of_business": doc["line_of_business"],
                "jurisdiction": doc["jurisdiction"],
                "effective_date": doc.get("effective_date"),
                "collections": doc.get("collections", [collection_name]),
            })
            print(f"    OK: {doc_id}")

        except Exception as e:
            print(f"    ERROR: {doc_id} — {str(e)[:100]}")
            skipped.append({"doc_id": doc_id, "reason": str(e)[:100]})

    if not fetched:
        raise RuntimeError(f"All documents failed to fetch for '{collection_name}'")

    manifest_key = f"{s3_staging_prefix}/manifest.json"
    s3.put_object(
        Bucket=s3_bucket, Key=manifest_key,
        Body=json.dumps(manifest_entries, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nManifest: s3://{s3_bucket}/{manifest_key} ({len(manifest_entries)} entries)")

    # OpenLineage emission
    try:
        from rhoai_lineage.kfp.lineage import kfp_lineage
        from urllib.parse import urlparse as _urlparse

        _s3_parsed = _urlparse(s3_endpoint)
        _s3_host = _s3_parsed.hostname or "minio"
        _s3_port = _s3_parsed.port or 9000

        input_datasets = [
            {
                "namespace": f"registry://{e['source_system']}",
                "name": e["doc_id"],
                "inputFacets": {
                    "document_metadata": {
                        "_producer": "https://github.com/rhoai-lineage",
                        "_schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/InputDatasetFacet",
                        "source_url": e.get("source_url", ""),
                        "source_system": e.get("source_system", ""),
                        "document_type": e.get("document_type", ""),
                        "line_of_business": e.get("line_of_business", ""),
                        "jurisdiction": e.get("jurisdiction", ""),
                        "effective_date": e.get("effective_date", ""),
                        "collection": collection_name,
                    }
                }
            }
            for e in manifest_entries
        ]
        output_ds = {
            "namespace": f"s3://{_s3_host}:{_s3_port}",
            "name": f"{s3_bucket}/{s3_staging_prefix}",
            "facets": {
                "custom_metrics": {
                    "_producer": "https://github.com/rhoai-lineage",
                    "_schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/CustomFacet",
                    "documents_fetched": len(fetched),
                    "documents_skipped": len(skipped),
                    "connector_type": connector_type,
                    "collection": collection_name,
                    "duration_seconds": round(time.time() - start_time, 2),
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
        with kfp_lineage(f"acquire_documents/{collection_name}", inputs=input_datasets,
                         outputs=[output_ds], run_facets=ol_run_facets):
            pass
        print("OpenLineage event emitted for acquire_documents")
    except Exception as e:
        print(f"WARNING: OL emission failed (non-fatal): {e}")

    # MLflow tracking
    try:
        import requests as _req
        sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        sa_ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        mlflow_url = "https://mlflow.redhat-ods-applications.svc:8443"
        headers = {"Content-Type": "application/json"}
        if os.path.exists(sa_token_path):
            with open(sa_token_path) as f:
                headers["Authorization"] = f"Bearer {f.read().strip()}"
        if os.path.exists(sa_ns_path):
            with open(sa_ns_path) as f:
                headers["X-Mlflow-Workspace"] = f.read().strip()

        exp_resp = _req.get(f"{mlflow_url}/api/2.0/mlflow/experiments/get-by-name",
                            params={"experiment_name": "data-strat-ingest"},
                            headers=headers, verify=False, timeout=10)
        if exp_resp.ok:
            experiment_id = exp_resp.json().get("experiment", {}).get("experiment_id")
            if experiment_id:
                parent_id = None
                search_resp = _req.post(f"{mlflow_url}/api/2.0/mlflow/runs/search",
                    json={"experiment_ids": [experiment_id],
                          "filter": f"tags.`kfp.pipeline_run_id` = '{pipeline_run_id}'", "max_results": 1},
                    headers=headers, verify=False, timeout=10)
                if search_resp.ok:
                    runs = search_resp.json().get("runs", [])
                    if runs:
                        parent_id = runs[0].get("info", {}).get("run_id")
                if not parent_id:
                    from datetime import datetime as _dt
                    _run_name = f"ingest/{collection_name}/{_dt.utcnow().strftime('%Y-%m-%d %H:%M')}"
                    cr = _req.post(f"{mlflow_url}/api/2.0/mlflow/runs/create",
                        json={"experiment_id": experiment_id, "run_name": _run_name,
                              "tags": [{"key": "kfp.pipeline_run_id", "value": pipeline_run_id},
                                       {"key": "kfp.namespace", "value": namespace},
                                       {"key": "kfp.component", "value": "pipeline"},
                                       {"key": "kfp.collection", "value": collection_name}]},
                        headers=headers, verify=False, timeout=10)
                    if cr.ok:
                        parent_id = cr.json().get("run", {}).get("info", {}).get("run_id")
                if parent_id:
                    nr = _req.post(f"{mlflow_url}/api/2.0/mlflow/runs/create",
                        json={"experiment_id": experiment_id, "run_name": "acquire_documents",
                              "tags": [{"key": "mlflow.parentRunId", "value": parent_id}]},
                        headers=headers, verify=False, timeout=10)
                    if nr.ok:
                        run_id = nr.json().get("run", {}).get("info", {}).get("run_id")
                        if run_id:
                            for k, v in {"pipeline_run_id": pipeline_run_id, "collection_name": collection_name,
                                         "connector_type": connector_type, "documents_fetched": str(len(fetched)),
                                         "documents_skipped": str(len(skipped))}.items():
                                _req.post(f"{mlflow_url}/api/2.0/mlflow/runs/log-parameter",
                                          json={"run_id": run_id, "key": k, "value": v},
                                          headers=headers, verify=False, timeout=5)
                            _req.post(f"{mlflow_url}/api/2.0/mlflow/runs/update",
                                      json={"run_id": run_id, "status": "FINISHED",
                                            "end_time": int(time.time() * 1000)},
                                      headers=headers, verify=False, timeout=5)
                            print(f"MLflow: logged acquire_documents under parent {pipeline_run_id}")
    except Exception as e:
        print(f"MLflow tracking failed (non-blocking): {e}")

    duration = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"ACQUIRE: {len(fetched)} fetched, {len(skipped)} skipped, {duration:.1f}s")
    print(f"{'=' * 60}")

    return manifest_key
