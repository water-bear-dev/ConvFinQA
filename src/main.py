"""
Main typer app for ConvFinQA
"""

import typer
from dotenv import load_dotenv
from rich import print as rich_print

from .data import DatasetLoader
from .eval import evaluate_split
from .format import format_document
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

    history: list[tuple[str, float]] = []
    # History stores (question, numeric_answer) so later turns can refer to earlier answers
    # (e.g., "what is the difference?" or "what percentage change is that?").
    rich_print(f"[bold green]Chatting about record: {record_id}[/bold green]")
    rich_print("[dim]Type 'exit' or 'quit' to end the conversation.[/dim]\n")
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        answer, reasoning = answer_turn(record, history, message)
        history.append((message, answer))
        rich_print(
            f"[blue][bold]assistant:[/bold] {answer} "
            f"(Reasoning: {reasoning})[/blue]"
        )


@app.command()
def eval(
    split: str = typer.Option("dev", help="Dataset split to evaluate"),
    limit: int = typer.Option(50, help="Max number of conversations to evaluate"),
) -> None:
    """Replay scripted conversations and report execution accuracy."""
    summary = evaluate_split(split=split, limit=limit)

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

    if summary.failures:
        rich_print("\n[bold blue]Sample Failures (first few)[/bold blue]")
        for f in summary.failures[:5]:
            rich_print(
                f"- {f['record_id']} (turn {f['turn']}, type2={f['has_type2']}): "
                f"gold={f['gold_raw']} pred={f['pred']}"
            )


if __name__ == "__main__":
    app()
