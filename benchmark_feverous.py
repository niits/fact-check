import asyncio
import csv
import zipfile
from pathlib import Path

import click
import mlflow
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from tqdm import tqdm

from src.impls.events.base import FactCheckStartEvent
from src.impls.workflows.simple import (
    EvidenceSimpleBaseFactCheck,
    SimpleBaseFactCheck
)
from src.modules.datasets.feverous import FeverousEvidenceFormat
from src.modules.evaluator import evaluate_file

load_dotenv()

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "dev.jsonl"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "feverous_wikiv1.db"


def _download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar, resuming from partial download if present."""
    import httpx

    partial = dest.with_suffix(dest.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    with httpx.stream(
        "GET", url, headers=headers, follow_redirects=True, timeout=None
    ) as resp:
        # If server doesn't support range or file is already complete
        if resp.status_code == 200:
            # Full response — start from scratch
            downloaded = 0
            mode = "wb"
        elif resp.status_code == 206:
            # Partial content — resume
            mode = "ab"
        else:
            resp.raise_for_status()
            return

        content_length = int(resp.headers.get("content-length", 0)) or None
        total = (downloaded + content_length) if content_length else None

        with (
            open(partial, mode) as f,
            tqdm(
                total=total,
                initial=downloaded,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as pbar,
        ):
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Download complete — rename .part to final name
    partial.rename(dest)


def download_feverous_data(data_dir: Path) -> None:
    """Download FEVEROUS dataset files if they are missing."""
    data_dir.mkdir(parents=True, exist_ok=True)

    files = {
        data_dir
        / "train.jsonl": "https://fever.ai/download/feverous/feverous_train_challenges.jsonl",
        data_dir
        / "dev.jsonl": "https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
        data_dir
        / "test_unlabeled.jsonl": "https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl",
        data_dir
        / "feverous-wiki-pages-db.zip": "https://fever.ai/download/feverous/feverous-wiki-pages-db.zip",
    }

    for dest, url in files.items():
        if not dest.exists():
            _download_file(url, dest)
        else:
            click.echo(f"Already exists: {dest}")

    db_path = data_dir / "feverous_wikiv1.db"
    zip_path = data_dir / "feverous-wiki-pages-db.zip"
    if not db_path.exists() and zip_path.exists():
        click.echo("Extracting feverous_wikiv1.db ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                filename = Path(member).name
                if filename:
                    dest = data_dir / filename
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

async def process_sample(
    i: int,
    sample: dict,
    wf,
    with_evidence: bool,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        try:
            context = sample["context"]
            claim = sample["claim"]
            evidence = sample["evidence"]
            label = sample["label"]

            start_ev = FactCheckStartEvent(
                context=evidence if with_evidence else "",
                claim=claim,
            )
            output = await wf.run(start_event=start_ev)

            result = output.result if hasattr(output, "result") else output
            if isinstance(result, dict):
                prediction = str(result.get("label"))
            else:
                prediction = str(result)

            return {
                "idx": i,
                "context": context,
                "claim": claim,
                "evidence": evidence,
                "label": label,
                "pred": prediction,
            }

        except (KeyError, ValueError, RuntimeError) as e:
            print(f"[benchmark] error at idx={i}: {e}")
            return None


async def benchmark(
    dataset,
    wf,
    output_file: str,
    with_evidence: bool,
    max_concurrency: int = 10,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["context", "claim", "evidence", "label", "pred"]

    semaphore = asyncio.Semaphore(max_concurrency)
    samples = list(dataset)
    print(f"[benchmark] Starting benchmark with {len(samples)} samples (concurrency={max_concurrency})")

    tasks = [
        process_sample(
            i, sample, wf, with_evidence, semaphore,
        )
        for i, sample in enumerate(samples)
    ]

    raw_results: list[dict | None] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await coro
        raw_results.append(result)

    # Sort by original index and write
    results: list[dict] = [r for r in raw_results if r is not None]
    results.sort(key=lambda r: r["idx"])
    print(f"[benchmark] Completed {len(results)}/{len(samples)} samples, writing to {output_file}")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "context": r["context"],
                    "claim": r["claim"],
                    "evidence": r["evidence"],
                    "label": r["label"],
                    "pred": r["pred"],
                }
            )


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=False),
    default=str(DEFAULT_DATA_PATH),
    show_default=True,
    help="Path to the FEVEROUS JSONL data file.",
)
@click.option(
    "--db-path",
    type=click.Path(exists=False),
    default=str(DEFAULT_DB_PATH),
    show_default=True,
    help="Path to the FEVEROUS wiki database file.",
)
@click.option(
    "--model",
    type=str,
    default="gpt-4.1-mini",
    show_default=True,
    help="Name of the OpenAI model to use.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/feverous-simple-evidence-format.csv",
    show_default=True,
    help="Path for the output CSV file.",
)
@click.option(
    "--with-evidence/--no-evidence",
    default=True,
    show_default=True,
    help="Whether to include evidence (instead of raw context) in the fact-check input.",
)
@click.option(
    "--concurrency",
    "-c",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of concurrent workflow tasks.",
)
@click.option(
    "--download",
    is_flag=True,
    default=False,
    help="Download FEVEROUS data files if they do not exist.",
)
@click.option(
    "--experiment-name",
    type=str,
    default="feverous-fact-check",
    show_default=True,
    help="MLflow experiment name.",
)
@click.option(
    "--tracking-uri",
    type=str,
    default=None,
    help="MLflow tracking URI (e.g. 'databricks'). Uses MLFLOW_TRACKING_URI env var if not set.",
)
def main(
    data_path: str,
    db_path: str,
    model: str,
    output: str,
    with_evidence: bool,
    concurrency: int,
    download: bool,
    experiment_name: str,
    tracking_uri: str | None,
) -> None:
    if download:
        data_dir = Path(data_path).parent
        download_feverous_data(data_dir)

    data_file = Path(data_path)
    db_file = Path(db_path)
    if not data_file.exists():
        raise click.BadParameter(
            f"Data file not found: {data_file}. Use --download to fetch it.",
            param_hint="'--data-path'",
        )
    if not db_file.exists():
        raise click.BadParameter(
            f"Database file not found: {db_file}. Use --download to fetch it.",
            param_hint="'--db-path'",
        )

    dataset = FeverousEvidenceFormat.from_path(data_path, db_path)
    llm = OpenAI(model=model)

    if with_evidence:
        wf = EvidenceSimpleBaseFactCheck(llm=llm)
    else:
        wf = SimpleBaseFactCheck(llm=llm)

    workflow_name = type(wf).__name__
    print(f"[benchmark] Model: {model}, output: {output}, with_evidence: {with_evidence}")

    # --- MLflow setup ---
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.llama_index.autolog()  # type: ignore[attr-defined]

    with mlflow.start_run(run_name=f"{workflow_name}-{model}") as parent_run:
        # Log params
        mlflow.log_params({
            "model": model,
            "workflow": workflow_name,
            "with_evidence": with_evidence,
            "concurrency": concurrency,
            "data_path": data_path,
            "db_path": db_path,
            "output_file": output,
        })

        # Run benchmark
        asyncio.run(
            benchmark(
                dataset, wf, output, with_evidence,
                max_concurrency=concurrency,
            )
        )

        # Evaluate
        print("[benchmark] Evaluating results...")
        eval_result = evaluate_file(output)

        # Log output CSV as artifact
        mlflow.log_artifact(output)

        # Log output dataframe as table
        df = eval_result["df"]
        mlflow.log_table(df, artifact_file="predictions.json")

        # Log classification report text
        mlflow.log_text(eval_result["cls_report_text"], "classification_report.txt")

        # Log classification report metrics (per-class + overall)
        cls_dict = eval_result["cls_report_dict"]
        for key, value in cls_dict.items():
            if isinstance(value, dict):
                for metric_name, metric_val in value.items():
                    safe_key = key.replace(" ", "_")
                    mlflow.log_metric(f"{safe_key}_{metric_name}", metric_val)
            else:
                # accuracy, macro avg, etc. scalar
                mlflow.log_metric(key.replace(" ", "_"), value)

        # Log workflow source code as artifact
        import inspect
        wf_source = inspect.getsource(type(wf))
        mlflow.log_text(wf_source, "workflow_source.py")

    print("[benchmark] Done. MLflow run logged.")
    print(f"[benchmark] Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[benchmark] Experiment: {experiment_name}")
    print(f"[benchmark] Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter