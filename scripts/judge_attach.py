import rich
import os
import subprocess
import json
import wandb
import random
from rich.console import Console
from rich.prompt import Prompt
from pathlib import Path
import threading
from tqdm import tqdm

console = Console()

def get_recent_runs(limit=5):
    """Get the most recent runs from eleutherai/sparsify."""
    try:
        api = wandb.Api()
        runs = api.runs("eleutherai/sparsify", per_page=limit, filters = {"state": "running"})
        recent_runs = []
        for run in runs:
            recent_runs.append({
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "config": run.config
            })
        return recent_runs[::-1]
    except Exception as e:
        console.print(f"[red]Error fetching recent runs: {e}[/red]")
        return []

def get_wandb_run_info(run_id):
    """Get run information from wandb using the run ID."""
    try:
        api = wandb.Api()
        run = api.run(f"eleutherai/sparsify/{run_id}")
        return {
            "name": run.name,
            "config": run.config,
            "state": run.state
        }
    except Exception as e:
        console.print(f"[red]Error fetching wandb run {run_id}: {e}[/red]")
        return None

def judge_single_prompt(run_name, prompt, prompt_idx, model_type="gpt2"):
    """Judge a model on a single prompt."""
    # Change to the project root directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # Load evaluation data
    eval_data = json.load(open(f"data/{model_type}-eval-data/data.json"))

    # Determine script name based on model type
    script_name = {
        "gpt2": "gpt",
        "gemma2-2b": "gemma",
        "llama-1b": "llama",
    }[model_type]

    # Get extra args if any
    extra_args = {
        "gpt2": {},
        "gemma2-2b": {},
        "llama-1b": {},
    }[model_type]

    judge_ctx = "j1"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(eval_data["model_name"])

    prompt_text = tokenizer.decode(prompt[:-1], skip_special_tokens=True)

    # For gpt2, remove the gpt2-sweep prefix if it exists in the run name
    if model_type == "gpt2" and run_name.startswith("gpt2-sweep/"):
        run_name_for_script = run_name[len("gpt2-sweep/"):]
    else:
        run_name_for_script = run_name

    args = [f"scripts/judge-{script_name}", run_name_for_script, *extra_args.get(run_name, "").split()]
    if model_type == "gpt2":
        prompt_text = "<|endoftext|>" + prompt_text
        args.append("--remove_prefix=1")

    prompt_name = f"{judge_ctx}-{prompt_idx}"

    # For gpt2, remove the gpt2-sweep prefix if it exists in the run name for path construction
    if model_type == "gpt2" and run_name.startswith("gpt2-sweep/"):
        model_name = run_name[len("gpt2-sweep/"):].replace("/", "_")
    else:
        model_name = run_name.replace("/", "_")

    results_path = f"./results/{model_type}-eval/{model_name}/{prompt_name}-{model_type}/results.json"

    # Check if already evaluated
    if os.path.exists(results_path):
        console.print(f"[yellow]Prompt {prompt_idx} already evaluated, loading existing results...[/yellow]")
        return results_path

    console.print(f"[yellow]Evaluating prompt {prompt_idx}...[/yellow]")

    try:
        pipes = subprocess.Popen(
            args,
            env=os.environ | dict(
                pn=prompt_name,
                pt=prompt_text,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        stdout, stderr = pipes.communicate()
        if pipes.returncode != 0:
            console.print(f"[red]Error running {run_name} on prompt {prompt_idx}: {pipes.returncode}[/red]")
            console.print(stdout.decode("utf-8"))
            console.print(stderr.decode("utf-8"))
            return None
        else:
            console.print(f"[green]Successfully evaluated prompt {prompt_idx}[/green]")
            return results_path
    except Exception as e:
        console.print(f"[red]Exception during evaluation: {e}[/red]")
        return None

def load_single_score(results_path):
    """Load score from a single evaluation result."""
    try:
        json_data = json.load(open(results_path))
        if "replacement_score_unpruned" in json_data:
            return json_data["replacement_score_unpruned"]
        else:
            console.print(f"[red]No replacement_score_unpruned found in {results_path}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Error loading score from {results_path}: {e}[/red]")
        return None

def log_to_wandb(run_id, metric_name, metric_value):
    """Log a metric back to the original wandb run."""
    try:
        api = wandb.Api()
        run = api.run(f"eleutherai/sparsify/{run_id}")

        # Create a new wandb run to log the metric
        with wandb.init(
            project="sparsify",
            entity="eleutherai",
            id=run_id,
            resume="must"
        ) as wandb_run:
            wandb_run.log({metric_name: metric_value})
            console.print(f"[green]Successfully logged {metric_name}: {metric_value} to run {run_id}[/green]")

    except Exception as e:
        console.print(f"[red]Error logging to wandb: {e}[/red]")

def main():
    console.print("[bold blue]Wandb Run Judge and Attach - Single Prompt Mode[/bold blue]")
    console.print("=" * 60)

    # Get recent runs
    console.print("[yellow]Fetching recent runs from eleutherai/sparsify...[/yellow]")
    recent_runs = get_recent_runs(5)

    if not recent_runs:
        console.print("[red]Failed to fetch recent runs. Exiting.[/red]")
        return

    # Display recent runs
    console.print("\n[bold cyan]Recent runs:[/bold cyan]")
    for i, run in enumerate(recent_runs, 1):
        created_str = run['created_at'] if run['created_at'] else 'Unknown'
        console.print(f"{i}. [green]{run['name']}[/green] (ID: {run['id']}) - {run['state']} - {created_str}")

    # Let user pick a run
    choice = Prompt.ask(
        f"\nSelect a run (1-{len(recent_runs)}) or enter a custom run ID",
        default="1"
    )

    # Determine run_id based on choice
    if choice.isdigit() and 1 <= int(choice) <= len(recent_runs):
        run_id = recent_runs[int(choice) - 1]['id']
        run_info = {
            "name": recent_runs[int(choice) - 1]['name'],
            "config": recent_runs[int(choice) - 1]['config'],
            "state": recent_runs[int(choice) - 1]['state']
        }
    else:
        # User entered a custom run ID
        run_id = choice
        console.print(f"[yellow]Looking up custom run {run_id}...[/yellow]")
        run_info = get_wandb_run_info(run_id)
        if not run_info:
            console.print("[red]Failed to fetch run information. Exiting.[/red]")
            return

    console.print(f"[green]Selected run: {run_info['name']}[/green]")
    console.print(f"Run state: {run_info['state']}")

    # Determine checkpoint path
    checkpoint_path = f"checkpoints/{run_info['name']}"
    console.print(f"Checkpoint path: {checkpoint_path}")

    # Wait for checkpoint to exist
    console.print(f"[yellow]Waiting for checkpoint path {checkpoint_path} to exist...[/yellow]")
    while not os.path.exists(checkpoint_path):
        console.print(f"[yellow]Checkpoint not found, waiting 30 seconds...[/yellow]")
        import time
        time.sleep(30)

    console.print(f"[green]Checkpoint found at {checkpoint_path}[/green]")

    # Determine model type from config or ask user
    model_type = run_info.get('config', {}).get('model_type', 'gpt2')
    if not model_type:
        model_type = Prompt.ask("Enter model type", choices=["gpt2", "gemma2-2b", "llama-1b"], default="gpt2")

    console.print(f"[yellow]Using model type: {model_type}[/yellow]")

    # Load evaluation data to get prompts
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    eval_data = json.load(open(f"data/{model_type}-eval-data/data.json"))
    prompts = eval_data["prompts"]

    console.print(f"[green]Loaded {len(prompts)} prompts for evaluation[/green]")

    # Interactive prompt evaluation loop
    while True:
        # Pick a random prompt
        prompt_idx = random.randint(0, len(prompts) - 1)
        prompt = prompts[prompt_idx]

        console.print(f"\n[bold cyan]Selected prompt {prompt_idx} for evaluation[/bold cyan]")

        # Judge the model on this single prompt
        results_path = judge_single_prompt(run_info['name'], prompt, prompt_idx, model_type)

        if results_path:
            # Load the score
            score = load_single_score(results_path)

            if score is not None:
                console.print(f"[green]Prompt {prompt_idx} score: {score:.4f}[/green]")

                # Log to wandb immediately
                console.print(f"[yellow]Logging score to wandb run {run_id}...[/yellow]")
                log_to_wandb(run_id, "replacement_score_unpruned", score)

        # Continue automatically with next random prompt
        console.print(f"[cyan]Continuing with next random prompt...[/cyan]")

    console.print("[bold green]Evaluation complete![/bold green]")

if __name__ == "__main__":
    main()
