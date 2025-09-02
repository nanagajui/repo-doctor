import click
import asyncio
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .agents import ProfileAgent, AnalysisAgent, ResolutionAgent
from .models.system import SystemProfile
from .models.analysis import Analysis
from .models.resolution import Resolution

console = Console()

@click.group()
@click.version_option()
def main():
    """Repo Doctor - Diagnose and resolve GitHub repository compatibility issues."""
    pass

@main.command()
@click.argument('repo_url')
@click.option('--strategy', type=click.Choice(['docker', 'conda', 'venv', 'auto']), default='auto',
              help='Environment generation strategy')
@click.option('--validate/--no-validate', default=True, help='Validate generated solution')
@click.option('--gpu-mode', type=click.Choice(['strict', 'flexible', 'cpu_fallback']), default='flexible',
              help='GPU compatibility mode')
@click.option('--output', type=click.Path(), help='Output directory for generated files')
def check(repo_url, strategy, validate, gpu_mode, output):
    """Check repository compatibility and generate environment."""
    asyncio.run(_check_async(repo_url, strategy, validate, gpu_mode, output))


async def _check_async(repo_url: str, strategy: str, validate: bool, gpu_mode: str, output: str):
    """Async implementation of check command."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Step 1: Profile system
        profile_task = progress.add_task("🔍 Profiling system...", total=None)
        profile_agent = ProfileAgent()
        system_profile = profile_agent.profile()
        progress.remove_task(profile_task)
        
        # Display system profile
        _display_system_profile(system_profile)
        
        # Step 2: Analyze repository
        analysis_task = progress.add_task("📦 Analyzing repository...", total=None)
        github_token = os.getenv('GITHUB_TOKEN')
        analysis_agent = AnalysisAgent(github_token)
        
        try:
            analysis = await analysis_agent.analyze(repo_url)
            progress.remove_task(analysis_task)
            
            # Display analysis results
            _display_analysis_results(analysis)
            
            # Step 3: Generate resolution
            resolution_task = progress.add_task("💡 Generating solution...", total=None)
            resolution_agent = ResolutionAgent()
            
            try:
                resolution = resolution_agent.resolve(analysis, strategy if strategy != 'auto' else None)
                progress.remove_task(resolution_task)
                
                # Display resolution
                _display_resolution(resolution)
                
                # Step 4: Save files
                if output:
                    _save_generated_files(resolution, output)
                else:
                    _save_generated_files(resolution, ".")
                
                # Step 5: Validation (if requested)
                if validate:
                    validation_task = progress.add_task("🧪 Validating solution...", total=None)
                    try:
                        validation_result = resolution_agent.validate_solution(resolution, analysis)
                        progress.remove_task(validation_task)
                        _display_validation_results(validation_result)
                    except Exception as e:
                        progress.remove_task(validation_task)
                        console.print(f"[yellow]⚠[/yellow] Validation failed: {str(e)}")
                
                # Display knowledge base insights
                _display_knowledge_insights(resolution_agent, analysis)
                
                console.print("\n[bold green]✅ Analysis complete![/bold green]")
                
            except Exception as e:
                progress.remove_task(resolution_task)
                console.print(f"[red]❌ Resolution failed: {str(e)}[/red]")
                
        except Exception as e:
            progress.remove_task(analysis_task)
            console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")


def _display_system_profile(profile: SystemProfile):
    """Display system profile information."""
    console.print("\n[bold blue]System Profile:[/bold blue]")
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Platform", profile.platform.title())
    table.add_row("CPU Cores", str(profile.hardware.cpu_cores))
    table.add_row("Memory", f"{profile.hardware.memory_gb:.1f} GB")
    table.add_row("Architecture", profile.hardware.architecture)
    
    if profile.has_gpu():
        gpu_info = profile.hardware.gpus[0]
        table.add_row("GPU", f"{gpu_info.name} ({gpu_info.memory_gb:.1f} GB)")
        if gpu_info.cuda_version:
            table.add_row("CUDA", gpu_info.cuda_version)
    else:
        table.add_row("GPU", "None detected")
    
    table.add_row("Container Runtime", profile.container_runtime or "None")
    table.add_row("Python", profile.software.python_version)
    table.add_row("Compute Score", f"{profile.compute_score:.1f}/100")
    
    console.print(table)


def _display_analysis_results(analysis: Analysis):
    """Display repository analysis results."""
    console.print(f"\n[bold blue]Repository Analysis:[/bold blue]")
    
    # Repository info
    repo_table = Table(show_header=False, box=None, padding=(0, 2))
    repo_table.add_column("Property", style="cyan")
    repo_table.add_column("Value", style="white")
    
    repo_table.add_row("Repository", f"{analysis.repository.owner}/{analysis.repository.name}")
    repo_table.add_row("Language", analysis.repository.language or "Unknown")
    repo_table.add_row("Stars", str(analysis.repository.stars))
    repo_table.add_row("Dependencies Found", str(len(analysis.dependencies)))
    repo_table.add_row("Analysis Time", f"{analysis.analysis_time:.2f}s")
    repo_table.add_row("Confidence", f"{analysis.confidence_score:.1%}")
    
    console.print(repo_table)
    
    # Dependencies
    if analysis.dependencies:
        console.print(f"\n[bold cyan]Dependencies ({len(analysis.dependencies)}):[/bold cyan]")
        dep_table = Table()
        dep_table.add_column("Package", style="white")
        dep_table.add_column("Version", style="yellow")
        dep_table.add_column("Type", style="green")
        dep_table.add_column("GPU", style="red")
        dep_table.add_column("Source", style="dim")
        
        for dep in analysis.dependencies[:10]:  # Show first 10
            dep_table.add_row(
                dep.name,
                dep.version or "Any",
                dep.type.value,
                "✓" if dep.gpu_required else "",
                dep.source
            )
        
        console.print(dep_table)
        
        if len(analysis.dependencies) > 10:
            console.print(f"[dim]... and {len(analysis.dependencies) - 10} more[/dim]")
    
    # Compatibility issues
    if analysis.compatibility_issues:
        console.print(f"\n[bold yellow]Compatibility Issues ({len(analysis.compatibility_issues)}):[/bold yellow]")
        for issue in analysis.compatibility_issues:
            severity_color = {"critical": "red", "warning": "yellow", "info": "blue"}.get(issue.severity, "white")
            console.print(f"[{severity_color}]• {issue.message}[/{severity_color}]")
            if issue.suggested_fix:
                console.print(f"  [dim]→ {issue.suggested_fix}[/dim]")


def _display_resolution(resolution: Resolution):
    """Display resolution information."""
    console.print(f"\n[bold green]Solution Generated:[/bold green]")
    
    console.print(f"Strategy: [cyan]{resolution.strategy.type.value.title()}[/cyan]")
    console.print(f"Files: [white]{len(resolution.generated_files)}[/white]")
    console.print(f"Setup Time: [yellow]~{resolution.strategy.estimated_setup_time}s[/yellow]")
    
    if resolution.generated_files:
        console.print(f"\n[bold cyan]Generated Files:[/bold cyan]")
        for file in resolution.generated_files:
            console.print(f"• [white]{file.path}[/white] - {file.description}")


def _save_generated_files(resolution: Resolution, output_dir: str):
    """Save generated files to output directory."""
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = []
    for file in resolution.generated_files:
        file_path = output_path / file.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(file.content)
        
        if file.executable:
            os.chmod(file_path, 0o755)
        
        saved_files.append(str(file_path))
    
    if saved_files:
        console.print(f"\n[bold green]Files saved to {output_dir}:[/bold green]")
        for file_path in saved_files:
            console.print(f"• [white]{file_path}[/white]")
    
    # Save instructions
    if resolution.instructions:
        instructions_path = output_path / "SETUP_INSTRUCTIONS.md"
        with open(instructions_path, 'w') as f:
            f.write(resolution.instructions)
        console.print(f"• [white]{instructions_path}[/white]")

@main.command()
@click.argument('repo_url')
@click.option('--from-ci', is_flag=True, help='Learn from CI configuration')
def learn(repo_url, from_ci):
    """Learn patterns from repository for future analysis."""
    console.print(f"[blue]📚[/blue] Learning from {repo_url}...")
    # TODO: Implement learning system

@main.command()
@click.option('--show-failures', is_flag=True, help='Show common failure patterns')
def patterns(show_failures):
    """Show learned patterns and compatibility data."""
    if show_failures:
        console.print("[red]❌[/red] Common failure patterns:")
        # TODO: Show failure patterns from knowledge base
    else:
        console.print("[green]✅[/green] Success patterns:")
        # TODO: Show success patterns from knowledge base

@main.command()
@click.option('--clear', is_flag=True, help='Clear all cached data')
def cache(clear):
    """Manage knowledge base cache."""
    if clear:
        console.print("[yellow]🗑[/yellow] Clearing cache...")
        # TODO: Implement cache clearing
    else:
        console.print("[blue]📊[/blue] Cache statistics:")
        # TODO: Show cache stats

def _display_validation_results(validation_result):
    """Display validation results."""
    console.print("\n[bold blue]Validation Results:[/bold blue]")
    
    if validation_result.status.value == "success":
        console.print("[green]✅ Validation successful![/green]")
        console.print(f"Duration: {validation_result.duration:.1f} seconds")
    else:
        console.print("[red]❌ Validation failed[/red]")
        if validation_result.error_message:
            console.print(f"Error: {validation_result.error_message}")
    
    if validation_result.logs:
        console.print("\n[bold]Validation Log:[/bold]")
        for log_line in validation_result.logs[-10:]:  # Show last 10 log lines
            console.print(f"  {log_line}")


def _display_knowledge_insights(resolution_agent, analysis):
    """Display insights from knowledge base."""
    similar_solutions = resolution_agent.get_similar_solutions(analysis, limit=3)
    
    if similar_solutions:
        console.print("\n[bold blue]Similar Repositories:[/bold blue]")
        for similar in similar_solutions:
            repo_info = similar['analysis']['repository']
            similarity = similar['similarity']
            console.print(f"  • {repo_info['owner']}/{repo_info['name']} (similarity: {similarity:.1%})")
    
    success_patterns = resolution_agent.get_success_patterns('docker')
    if success_patterns:
        console.print("\n[bold blue]Success Patterns:[/bold blue]")
        console.print(f"  • Docker strategy success rate: {success_patterns.get('count', 0)} successful builds")
        avg_time = success_patterns.get('avg_setup_time', 0)
        if avg_time > 0:
            console.print(f"  • Average setup time: {avg_time:.0f} seconds")


if __name__ == "__main__":
    main()
