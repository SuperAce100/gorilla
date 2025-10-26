import csv
from datetime import datetime
import os
from types import SimpleNamespace
from typing import List, Optional

import typer
from importlib.metadata import version as _version
from bfcl_eval._llm_response_generation import main as generation_main
from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    PROJECT_ROOT,
    RESULT_PATH,
    SCORE_PATH,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
from dotenv import load_dotenv
from tabulate import tabulate

from bfcl_eval.augmentation.split import SplitConfig, run_split
from bfcl_eval.augmentation.train import TrainConfig, run_train
from bfcl_eval.augmentation.test import TestConfig, run_test
from bfcl_eval.augmentation.report import run_report, run_compare


class ExecutionOrderGroup(typer.core.TyperGroup):
    def list_commands(self, ctx):
        return [
            "models",
            "test-categories",
            "generate",
            "results",
            "evaluate",
            "scores",
            "augment",
            "version",
        ]


cli = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    no_args_is_help=True,
    cls=ExecutionOrderGroup,
)


def handle_multiple_input(input_str):
    """
    Input is like 'a,b,c,d', we need to transform it to ['a', 'b', 'c', 'd'] because that's the expected format in the actual main funciton
    """
    if input_str is None:
        """
        Cannot return None here, as typer will check the length of the return value and len(None) will raise an error
        But when default is None, an empty list will be internally converted to None, and so the pipeline still works as expected
        ```
        if default_value is None and len(value) == 0:
            return None
        ```
        """
        return []

    return [item.strip() for item in ",".join(input_str).split(",") if item.strip()]


augment_app = typer.Typer(help="Agentic ICL augmentation workflows")
cli.add_typer(augment_app, name="augment")


@augment_app.command("split")
def augment_split(
    tag: str = typer.Option(..., help="Run tag name"),
    seed: int = typer.Option(42, help="Random seed for deterministic split"),
    ratio: str = typer.Option("2:1", help="Train:Test ratio, e.g., 2:1"),
    test_category: List[str] = typer.Option(
        ["agentic"], callback=handle_multiple_input
    ),
):
    """Create a deterministic split over agentic subcategories and write split.json."""
    cfg = SplitConfig(tag=tag, seed=seed, ratio=ratio, test_categories=test_category)
    path = run_split(cfg)
    print(f"✅ Wrote split file: {path}")


@augment_app.command("train")
def augment_train(
    tag: str = typer.Option(..., help="Run tag name"),
    model_train: str = typer.Option(..., help="OpenAI FC model for training pool"),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name"
    ),
    num_threads: Optional[int] = typer.Option(
        None, help="Number of concurrent examples to run in parallel"
    ),
):
    """Run generation on train split, evaluate, create success pool and FAISS index."""
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    cfg = TrainConfig(
        tag=tag,
        model_train=model_train,
        embedding_model=embedding_model,
        num_threads=num_threads,
    )
    run_train(cfg)
    print("✅ Training run complete (results, scores, success pool, index)")


@augment_app.command("test")
def augment_test(
    tag: str = typer.Option(..., help="Run tag name"),
    model_eval: str = typer.Option(..., help="OpenAI FC eval model X"),
    source_model: str = typer.Option(..., help="Model Y that produced the pool"),
    retrieval_scope: str = typer.Option("subcategory", help="subcategory|agentic"),
    k: int = typer.Option(5, help="Top-k examples to retrieve"),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name"
    ),
    num_threads: Optional[int] = typer.Option(
        None, help="Number of concurrent examples to run in parallel"
    ),
):
    """Run augmented generation on the test split and evaluate."""
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    cfg = TestConfig(
        tag=tag,
        model_eval=model_eval,
        source_model=source_model,
        retrieval_scope=retrieval_scope,
        k=k,
        embedding_model=embedding_model,
        num_threads=num_threads,
    )
    run_test(cfg)
    print("✅ Augmented test run complete (results, scores)")


@augment_app.command("report")
def augment_report(
    tag: str = typer.Option(..., help="Run tag name"),
):
    """Generate train/test CSV summaries under augment/runs/TAG/report."""
    out = run_report(tag)
    print(f"📈 Wrote reports under: {out}")


@augment_app.command("compare")
def augment_compare(
    tag: str = typer.Option(..., help="Run tag name"),
    model_eval: str = typer.Option(..., help="Eval model X"),
    source_model: str = typer.Option(..., help="Pool model Y"),
    retrieval_scope: str = typer.Option("subcategory", help="subcategory|agentic"),
):
    """Print where augmented reports live; compare manually vs baseline scores output."""
    res = run_compare(tag, model_eval, source_model, retrieval_scope)
    print(json.dumps(res, indent=2))


@cli.command()
def version():
    """
    Show the bfcl version. PyPI versions are in development, please rely on the commit hash for reproducibility.
    """
    print(
        f"bfcl version: {_version('bfcl')} \nNote: pypi versions are in development, please rely on the commit hash for reproducibility."
    )


@cli.command()
def test_categories():
    """
    List available test categories.
    """
    table = tabulate(
        [
            (category, "\n".join(test for test in tests))
            for category, tests in TEST_COLLECTION_MAPPING.items()
        ],
        headers=["Test category", "Test names"],
        tablefmt="grid",
    )
    print(table)


@cli.command()
def models():
    """
    List available models.
    """
    table = tabulate(
        [[model] for model in MODEL_CONFIG_MAPPING.keys()],
        tablefmt="plain",
        colalign=("left",),
    )
    print(table)


@cli.command()
def generate(
    model: List[str] = typer.Option(
        ["gorilla-openfunctions-v2"],
        help="A list of model names to generate the llm response. Use commas to separate multiple models.",
        callback=handle_multiple_input,
    ),
    test_category: List[str] = typer.Option(
        ["all"],
        help="A list of test categories to run the evaluation on. Use commas to separate multiple test categories.",
        callback=handle_multiple_input,
    ),
    temperature: float = typer.Option(
        0.001, help="The temperature parameter for the model."
    ),
    include_input_log: bool = typer.Option(
        False,
        "--include-input-log",
        help="Include the fully-transformed input to the model inference endpoint in the inference log; only relevant for debugging input integrity and format.",
    ),
    exclude_state_log: bool = typer.Option(
        False,
        "--exclude-state-log",
        help="Exclude info about the state of each API system after each turn in the inference log; only relevant for multi-turn categories.",
    ),
    num_gpus: int = typer.Option(1, help="The number of GPUs to use."),
    num_threads: Optional[int] = typer.Option(
        None, help="The number of threads to use."
    ),
    gpu_memory_utilization: float = typer.Option(
        0.9, help="The GPU memory utilization."
    ),
    backend: str = typer.Option("sglang", help="The backend to use for the model."),
    skip_server_setup: bool = typer.Option(
        False,
        "--skip-server-setup",
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the LOCAL_SERVER_ENDPOINT and LOCAL_SERVER_PORT environment variables.",
    ),
    local_model_path: Optional[str] = typer.Option(
        None,
        "--local-model-path",
        help="Specify the path to a local directory containing the model's config/tokenizer/weights for fully offline inference. Use this only if the model weights are stored in a location other than the default HF_HOME directory.",
    ),
    result_dir: str = typer.Option(
        RESULT_PATH,
        "--result-dir",
        help="Path to the folder where output files will be stored; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
    allow_overwrite: bool = typer.Option(
        False,
        "--allow-overwrite",
        "-o",
        help="Allow overwriting existing results for regeneration.",
    ),
    run_ids: bool = typer.Option(
        False,
        "--run-ids",
        help="If true, also run the test entry mentioned in the test_case_ids_to_generate.json file, in addition to the --test_category argument.",
    ),
):
    """
    Generate the LLM response for one or more models on a test-category (same as openfunctions_evaluation.py).
    """

    args = SimpleNamespace(
        model=model,
        test_category=test_category,
        temperature=temperature,
        include_input_log=include_input_log,
        exclude_state_log=exclude_state_log,
        num_gpus=num_gpus,
        num_threads=num_threads,
        gpu_memory_utilization=gpu_memory_utilization,
        backend=backend,
        skip_server_setup=skip_server_setup,
        local_model_path=local_model_path,
        result_dir=result_dir,
        allow_overwrite=allow_overwrite,
        run_ids=run_ids,
    )
    load_dotenv(
        dotenv_path=DOTENV_PATH, verbose=True, override=True
    )  # Load the .env file
    generation_main(args)


@cli.command()
def results(
    result_dir: str = typer.Option(
        None,
        "--result-dir",
        help="Relative path to the model response folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
):
    """
    List the results available for evaluation.
    """

    def display_name(name: str):
        """
        Undo the / -> _ transformation if it happened.

        Args:
            name (str): The name of the model in the result directory.

        Returns:
            str: The original name of the model.
        """
        if name not in MODEL_CONFIG_MAPPING:
            candidate = name.replace("_", "/")
            if candidate in MODEL_CONFIG_MAPPING:
                return candidate
            print(f"Unknown model name: {name}")
        return name

    if result_dir is None:
        result_dir = RESULT_PATH
    else:
        result_dir = (PROJECT_ROOT / result_dir).resolve()

    results_data = []
    for dir in result_dir.iterdir():
        # Check if it is a directory and not a file
        if not dir.is_dir():
            continue

        results_data.append(
            (
                display_name(dir.name),
                datetime.fromtimestamp(dir.stat().st_ctime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )
        )

    print(
        tabulate(
            results_data,
            headers=["Model name", "Creation time"],
            tablefmt="pretty",
        )
    )


@cli.command()
def evaluate(
    model: List[str] = typer.Option(
        None, help="A list of model names to evaluate.", callback=handle_multiple_input
    ),
    test_category: List[str] = typer.Option(
        ["all"],
        help="A list of test categories to run the evaluation on.",
        callback=handle_multiple_input,
    ),
    result_dir: str = typer.Option(
        None,
        "--result-dir",
        help="Relative path to the model response folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
    score_dir: str = typer.Option(
        None,
        "--score-dir",
        help="Relative path to the evaluation score folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
    partial_eval: bool = typer.Option(
        False,
        "--partial-eval",
        help="Run evaluation on a partial set of benchmark entries (eg. entries present in the model result files) without raising for missing IDs.",
    ),
):
    """
    Evaluate results from run of one or more models on a test-category (same as eval_runner.py).
    """

    load_dotenv(
        dotenv_path=DOTENV_PATH, verbose=True, override=True
    )  # Load the .env file
    evaluation_main(model, test_category, result_dir, score_dir, partial_eval)


@cli.command()
def scores(
    score_dir: str = typer.Option(
        None,
        "--score-dir",
        help="Relative path to the evaluation score folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
):
    """
    Display the leaderboard.
    """

    def truncate(text, length=22):
        return (text[:length] + "...") if len(text) > length else text

    if score_dir is None:
        score_dir = SCORE_PATH
    else:
        score_dir = (PROJECT_ROOT / score_dir).resolve()
    # files = ["./score/data_non_live.csv", "./score/data_live.csv", "./score/data_overall.csv"]
    file = score_dir / "data_overall.csv"

    selected_columns = [
        "Rank",
        "Model",
        "Overall Acc",
        "Non-Live AST Acc",
        "Non-Live Exec Acc",
        "Live Acc",
        "Multi Turn Acc",
        "Relevance Detection",
        "Irrelevance Detection",
    ]

    if file.exists():
        with open(file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Read the header row
            column_indices = [headers.index(col) for col in selected_columns]
            data = [
                [row[i] for i in column_indices] for row in reader
            ]  # Read the rest of the data
            selected_columns = selected_columns[:-2] + [
                "Relevance",
                "Irrelevance",
            ]  # Shorten the column names
            print(tabulate(data, headers=selected_columns, tablefmt="grid"))
    else:
        print(f"\nFile {file} not found.\n")


if __name__ == "__main__":
    cli()
