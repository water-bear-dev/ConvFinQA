"""
Main typer app for ConvFinQA
"""

import typer
from dotenv import load_dotenv
from rich import print as rich_print

from .data import DatasetLoader
from .eval import compare_eval_results, evaluate_split
from .format import format_document
from .history import TurnState
from .solver import answer_turn

load_dotenv()

app = typer.Typer(
    name="main",
    help="ConvFinQA conversational financial QA CLI",
    add_completion=True,
    no_args_is_help=True,
)

DATASET_PATH = "data/convfinqa_dataset.json"
loader = DatasetLoader(DATASET_PATH)


@app.command()
def show(
    record_id: str = typer.Argument(..., help="ID of the record to show"),
) -> None:
    """Show formatted document and gold dialogue for debugging."""
    record = loader.get_record(record_id)
    if not record:
        rich_print(f"[red]Record with ID '{record_id}' not found.[/red]")
        raise typer.Exit(code=1)

    rich_print(f"[bold green]Record ID: {record.id}[/bold green]\n")
    rich_print("[bold blue]Formatted Document:[/bold blue]")
    rich_print(format_document(record.doc))

    rich_print("\n[bold blue]Dialogue Turns:[/bold blue]")
    for i, (question, answer) in enumerate(
        zip(record.dialogue.conv_questions, record.dialogue.conv_answers)
    ):
        rich_print(f"[yellow]Turn {i + 1}:[/yellow]")
        rich_print(f"  [bold]Q:[/bold] {question}")
        rich_print(f"  [bold]A:[/bold] {answer}")


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record (multi-turn; keeps numeric history)."""
    record = loader.get_record(record_id)
    if not record:
        rich_print(f"[red]Record with ID '{record_id}' not found.[/red]")
        raise typer.Exit(code=1)

    history: list[TurnState] = []
    rich_print(f"[bold green]Chatting about record: {record_id}[/bold green]")
    rich_print("[dim]Type 'exit' or 'quit' to end the conversation.[/dim]\n")
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        result = answer_turn(record, history, message, turn_idx=len(history))
        history.append(
            TurnState(question=message, answer=result.answer, metric=result.metric)
        )
        rich_print(
            f"[blue][bold]assistant:[/bold] {result.answer} "
            f"(route={result.route}, Reasoning: {result.reasoning})[/blue]"
        )


@app.command()
def eval(
    split: str = typer.Option("dev", help="Dataset split to evaluate"),
    limit: int = typer.Option(50, help="Max number of conversations to evaluate"),
    output: str | None = typer.Option(
        None, help="Optional path to write full eval JSON (results + failures + summary)"
    ),
    baseline: str | None = typer.Option(
        None, help="Optional baseline eval JSON to compare against after run"
    ),
) -> None:
    """Replay scripted conversations and report execution accuracy."""
    summary = evaluate_split(split=split, limit=limit, output_path=output)

    rich_print("[bold blue]Evaluation Summary[/bold blue]")
    rich_print(
        f"Split={summary.split}  Records={summary.total_records}  "
        f"Turns={summary.total_turns}  Correct={summary.correct_turns}"
    )
    rich_print(f"[bold]Turn-level Exe Acc:[/bold] {summary.acc:.2%}")
    rich_print(
        f"[bold]Turn 0–1:[/bold] {summary.acc_turn01:.2%}    "
        f"[bold]Turn 2+:[/bold] {summary.acc_turn2plus:.2%}"
    )
    rich_print(
        f"[bold]Type I:[/bold] {summary.acc_type1:.2%}    "
        f"[bold]Type II:[/bold] {summary.acc_type2:.2%}"
    )
    if summary.acc_by_route:
        rich_print(
            "[bold]By route:[/bold] "
            + "  ".join(f"{k}={v:.2%}" for k, v in summary.acc_by_route.items())
        )
    rich_print(f"[bold]Retries:[/bold] {summary.retry_count}")
    if summary.error_tag_counts:
        rich_print(f"[bold]Error tags:[/bold] {summary.error_tag_counts}")

    if summary.failures:
        rich_print("\n[bold blue]Sample Failures (first few)[/bold blue]")
        for f in summary.failures[:5]:
            rich_print(
                f"- {f['record_id']} (turn {f['turn']}, route={f.get('route')}): "
                f"gold={f['gold_raw']} pred={f['pred']}"
            )

    if output:
        rich_print(f"\n[dim]Wrote eval results to: {output}[/dim]")

    if baseline and output:
        comparison = compare_eval_results(baseline, output)
        rich_print("\n[bold blue]Comparison vs baseline[/bold blue]")
        for metric, delta in comparison["deltas"].items():
            rich_print(f"  {metric}: {delta:+.2%}")
        rich_print(
            f"  Fixed: {comparison['fixed_count']}  "
            f"Broken: {comparison['broken_count']}"
        )


@app.command("eval-compare")
def eval_compare(
    baseline: str = typer.Argument(..., help="Baseline eval JSON path"),
    candidate: str = typer.Argument(..., help="Candidate eval JSON path"),
) -> None:
    """Compare two eval result files and print metric deltas."""
    comparison = compare_eval_results(baseline, candidate)
    rich_print("[bold blue]Eval Comparison[/bold blue]")
    rich_print(f"Baseline acc: {comparison['baseline'].get('acc', 0):.2%}")
    rich_print(f"Candidate acc: {comparison['candidate'].get('acc', 0):.2%}")
    for metric, delta in comparison["deltas"].items():
        rich_print(f"  {metric}: {delta:+.2%}")
    rich_print(
        f"Fixed turns: {comparison['fixed_count']}  "
        f"Broken turns: {comparison['broken_count']}"
    )
    if comparison["fixed_sample"]:
        rich_print("\n[green]Sample fixed turns:[/green]")
        for row in comparison["fixed_sample"][:5]:
            rich_print(
                f"  {row['record_id']} turn {row['turn']}: "
                f"gold={row['gold']} pred={row['pred']}"
            )
    if comparison["broken_sample"]:
        rich_print("\n[red]Sample broken turns:[/red]")
        for row in comparison["broken_sample"][:5]:
            rich_print(
                f"  {row['record_id']} turn {row['turn']}: "
                f"gold={row['gold']} pred={row['pred']}"
            )


if __name__ == "__main__":
    app()
