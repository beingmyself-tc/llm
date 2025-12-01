import yaml
import subprocess
import sys
import time
import os
import signal
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
processes = []

def load_config():
    with open("models.yaml", "r") as f:
        return yaml.safe_load(f)

def start_server(model_config):
    cmd = [
        "mlx_lm.server",
        "--model", model_config["repo_id"],
        "--port", str(model_config["port"])
    ]
    
    if "draft_model" in model_config:
        cmd.extend(["--draft-model", model_config["draft_model"]])
        
    console.print(f"[green]Starting {model_config['name']} on port {model_config['port']}...[/green]")
    if "draft_model" in model_config:
        console.print(f"  [dim]Using draft model: {model_config['draft_model']}[/dim]")

    # Start process
    log_file = open(f"logs/{model_config['id']}.log", "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        preexec_fn=os.setsid # Create new process group
    )
    return process, log_file

def stop_servers():
    console.print("\n[bold red]Stopping all servers...[/bold red]")
    for p, f in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            f.close()
        except:
            pass
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="MLX Model Manager")
    parser.add_argument("--list", action="store_true", help="List configured models")
    args = parser.parse_args()

    config = load_config()
    
    if args.list:
        table = Table(title="Configured MLX Models")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Port", style="green")
        table.add_column("Model Repo", style="dim")
        
        for m in config["models"]:
            table.add_row(m["id"], m["name"], str(m["port"]), m["repo_id"])
        
        console.print(table)
        return

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, lambda x, y: stop_servers())

    console.print(Panel.fit("[bold blue]MLX Local Model Server Manager[/bold blue]\nPress Ctrl+C to stop all servers"))

    for model in config["models"]:
        p, f = start_server(model)
        processes.append((p, f))

    console.print("\n[bold]Servers are running![/bold]")
    console.print("You can connect using any OpenAI-compatible client (Chatbox, Continue, etc.)")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_servers()

if __name__ == "__main__":
    main()
