"""
RuleGround CLI Entry Point

Commands:
    ruleground train    - Train the model (all 3 stages)
    ruleground eval     - Evaluate a trained model
    ruleground extract  - Extract predicates from rationales
    ruleground demo     - Run inference on a video
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="ruleground",
    help="RuleGround: Rule-Grounded Representations for Sports Video Understanding",
    add_completion=False,
)


@app.command()
def train(
    config: str = typer.Option(..., "-c", "--config", help="Path to YAML config"),
    output: str = typer.Option("experiments", "-o", "--output", help="Output directory"),
    gpus: int = typer.Option(1, "-g", "--gpus", help="Number of GPUs"),
    resume: str = typer.Option(None, "-r", "--resume", help="Resume from checkpoint"),
    overrides: list[str] = typer.Option(None, "--set", help="Config overrides (key=value)"),
):
    """Train RuleGround model (supervised + GRPO + RSA)."""
    from ruleground.cli.train import run_training

    run_training(
        config_path=config,
        output_dir=output,
        num_gpus=gpus,
        resume_path=resume,
        overrides=overrides,
    )


@app.command(name="eval")
def evaluate(
    checkpoint: str = typer.Option(..., "-ckpt", "--checkpoint", help="Model checkpoint"),
    config: str = typer.Option(..., "-c", "--config", help="Config YAML"),
    split: str = typer.Option("test", "-s", "--split", help="Dataset split"),
    output: str = typer.Option(None, "-o", "--output", help="Results output path"),
    per_sport: bool = typer.Option(True, help="Per-sport breakdown"),
):
    """Evaluate a trained model on SportR."""
    from ruleground.cli.eval import run_evaluation

    run_evaluation(
        checkpoint_path=checkpoint,
        config_path=config,
        split=split,
        output_path=output,
        per_sport=per_sport,
    )


@app.command()
def extract(
    data: str = typer.Option(..., "-d", "--data", help="Path to rationales JSON"),
    output: str = typer.Option(..., "-o", "--output", help="Output predicates JSON"),
    backend: str = typer.Option("anthropic", "-b", "--backend", help="LLM backend"),
    model: str = typer.Option("claude-sonnet-4-20250514", "-m", "--model", help="Model name"),
    validate: bool = typer.Option(False, help="Cross-model validation"),
):
    """Extract predicates from human rationales using LLM."""
    from ruleground.cli.extract import run_extraction

    run_extraction(
        data_path=data,
        output_path=output,
        backend=backend,
        model_name=model,
        validate=validate,
    )


if __name__ == "__main__":
    app()
