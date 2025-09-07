import asyncio
import os
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .agents import AnalysisAgent, ProfileAgent, ResolutionAgent
from .models.analysis import Analysis
from .models.resolution import Resolution
from .models.system import SystemProfile
from .utils.config import Config
from .utils.env import EnvLoader, load_environment
from .utils.logging_config import setup_logging
from .presets import PRESETS, get_preset, list_presets, apply_preset_to_config

# Import learning system components
try:
    from .learning import EnhancedAnalysisAgent, EnhancedResolutionAgent, LearningDashboard
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

console = Console()


@click.group()
@click.version_option()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Set logging level"
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Log to file (structured JSON format)"
)
def main(log_level: str, log_file: Optional[str]) -> None:
    """Repo Doctor - Diagnose and resolve GitHub repository compatibility issues.
    
    Generated files are saved to outputs/{owner}-{repo}/ by default.
    Use --output to specify a custom directory.
    """
    # Set up logging based on CLI options
    setup_logging(level=log_level, log_file=log_file, console=True)


@main.command()
@click.argument("repo_url")
@click.option(
    "--preset",
    type=click.Choice(list(PRESETS.keys())),
    help="Use a preset configuration (ml-research, production, development, quick)"
)
@click.option(
    "--output", "-o",
    type=click.Path(), 
    help="Output directory for generated files"
)
@click.option(
    "--quick", 
    is_flag=True,
    help="Quick mode - skip validation for faster results"
)
@click.option(
    "--advanced",
    is_flag=True,
    help="Show advanced options"
)
def check(repo_url: str, preset: Optional[str], output: Optional[str], quick: bool, advanced: bool) -> None:
    """Check repository compatibility and generate environment.
    
    Examples:
        repo-doctor check pytorch/pytorch
        repo-doctor check pytorch/pytorch --preset production
        repo-doctor check pytorch/pytorch --quick
        repo-doctor check pytorch/pytorch --advanced
    """
    if advanced:
        # Show help for advanced command
        ctx = click.get_current_context()
        console.print("[yellow]Use 'repo-doctor check-advanced' for advanced options[/yellow]")
        console.print("\nAvailable presets:")
        for name, desc in list_presets().items():
            console.print(f"  {name}: {desc}")
        ctx.abort()
    
    # Apply preset if specified
    config = Config.load()
    if preset:
        apply_preset_to_config(config, preset)
        console.print(f"[green]Using preset: {PRESETS[preset]['name']}[/green]")
    elif quick:
        # Apply quick preset for fast mode
        apply_preset_to_config(config, "quick")
        console.print("[yellow]Quick mode enabled - skipping validation[/yellow]")
    
    # Use defaults from config
    strategy = config.defaults.strategy
    validate = config.defaults.validation and not quick
    gpu_mode = config.defaults.gpu_mode
    enable_llm = config.integrations.llm.enabled
    llm_url = config.integrations.llm.base_url
    llm_model = config.integrations.llm.model
    
    try:
        asyncio.run(
            _check_async(
                repo_url,
                strategy,
                validate,
                gpu_mode,
                output,
                enable_llm,
                llm_url,
                llm_model,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Operation cancelled by user")
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise click.Abort()


@main.command("presets")
def show_presets() -> None:
    """Show available preset configurations."""
    console.print("\n[bold]Available Preset Configurations[/bold]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Preset", style="green")
    table.add_column("Description")
    table.add_column("Strategy", style="yellow")
    table.add_column("GPU Mode", style="magenta")
    table.add_column("Validation", style="blue")
    
    for name, config in PRESETS.items():
        table.add_row(
            name,
            config["description"],
            config["strategy"],
            config["gpu_mode"],
            "✓" if config["validation"] else "✗"
        )
    
    console.print(table)
    console.print("\n[dim]Use: repo-doctor check <repo> --preset <preset-name>[/dim]\n")


@main.command("check-advanced")
@click.argument("repo_url")
@click.option(
    "--strategy",
    type=click.Choice(["docker", "conda", "venv", "auto"]),
    default="auto",
    help="Environment generation strategy",
)
@click.option(
    "--validate/--no-validate", default=True, help="Validate generated solution"
)
@click.option(
    "--gpu-mode",
    type=click.Choice(["strict", "flexible", "cpu_fallback"]),
    default="flexible",
    help="GPU compatibility mode",
)
@click.option(
    "--output", type=click.Path(), help="Output directory for generated files (default: outputs/{owner}-{repo})"
)
@click.option(
    "--enable-llm/--disable-llm", default=None, help="Enable/disable LLM assistance"
)
@click.option("--llm-url", help="LLM server URL (overrides config)")
@click.option("--llm-model", help="LLM model name (overrides config)")
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option("--cache-ttl", type=int, help="Cache TTL in seconds")
def check_advanced(
    repo_url: str, strategy: str, validate: bool, gpu_mode: str, output: Optional[str], 
    enable_llm: Optional[bool], llm_url: Optional[str], llm_model: Optional[str],
    no_cache: bool, cache_ttl: Optional[int]
) -> None:
    """Advanced check with all configuration options."""
    try:
        asyncio.run(
            _check_async(
                repo_url,
                strategy,
                validate,
                gpu_mode,
                output,
                enable_llm,
                llm_url,
                llm_model,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Operation cancelled by user")
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise click.Abort()


async def _check_async(
    repo_url: str,
    strategy: str,
    validate: bool,
    gpu_mode: str,
    output: str,
    enable_llm: Optional[bool] = None,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
):
    """Async implementation of check command."""
    try:
        # Load and configure settings
        config = Config.load()

        # Override LLM settings from CLI options
        if enable_llm is not None:
            config.integrations.llm.enabled = enable_llm
        if llm_url:
            config.integrations.llm.base_url = llm_url
        if llm_model:
            config.integrations.llm.model = llm_model

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # STREAM B Optimization: Run Profile and Analysis prep in parallel
            parallel_task = progress.add_task("🚀 Parallel initialization...", total=None)
            
            try:
                # Create tasks for parallel execution
                async def profile_system():
                    """Profile system in parallel."""
                    profile_agent = ProfileAgent(config)
                    # Prefer async method if available and a coroutine function
                    try:
                        import inspect
                        if hasattr(profile_agent, "profile_async") and inspect.iscoroutinefunction(profile_agent.profile_async):
                            return await profile_agent.profile_async()
                        # Fallback to running sync method in a thread (works with mocks)
                        return await asyncio.to_thread(profile_agent.profile)
                    except Exception:
                        # Last resort: call sync directly (in case event loop/thread off)
                        return profile_agent.profile()
                
                async def prepare_analysis():
                    """Prepare analysis agent in parallel."""
                    github_token = EnvLoader.get_github_token()
                    _check_token_configuration()
                    
                    # Use enhanced agent if learning is enabled and available
                    learning_enabled = (hasattr(config, 'learning') and 
                                      getattr(config.learning, 'enabled', False)) or config.integrations.llm.enabled
                    if LEARNING_AVAILABLE and learning_enabled:
                        return EnhancedAnalysisAgent(config, github_token, use_cache=True)
                    else:
                        return AnalysisAgent(config, github_token, use_cache=True)
                
                # Run both tasks in parallel
                profile_future = asyncio.create_task(profile_system())
                analysis_prep_future = asyncio.create_task(prepare_analysis())
                
                # Wait for both to complete
                system_profile, analysis_agent = await asyncio.gather(
                    profile_future, 
                    analysis_prep_future,
                    return_exceptions=False
                )
                try:
                    progress.remove_task(parallel_task)
                except Exception:
                    pass
                
                # Display system profile
                _display_system_profile(system_profile)

                # Display LLM status if configured
                if config.integrations.llm.enabled:
                    llm_url = config.integrations.llm.base_url or "auto-detected"
                    console.print(
                        f"[dim]🤖 LLM assistance enabled: {config.integrations.llm.model} @ {llm_url}[/dim]"
                    )
                    
            except Exception as e:
                progress.remove_task(parallel_task)
                console.print(f"[yellow]⚠[/yellow] Parallel initialization failed: {str(e)}")
                # Fall back to sequential execution
                profile_agent = ProfileAgent(config)
                system_profile = profile_agent.profile_sync()
                github_token = EnvLoader.get_github_token()
                _check_token_configuration()
                
                # Use enhanced agent if learning is enabled and available
                learning_enabled = (hasattr(config, 'learning') and 
                                  getattr(config.learning, 'enabled', False)) or config.integrations.llm.enabled
                if LEARNING_AVAILABLE and learning_enabled:
                    analysis_agent = EnhancedAnalysisAgent(config, github_token, use_cache=True)
                else:
                    analysis_agent = AnalysisAgent(config, github_token, use_cache=True)

            # Step 2: Analyze repository (now with pre-initialized agent)
            analysis_task = progress.add_task("📦 Analyzing repository...", total=None)
            try:
                # Check GitHub API rate limit before analysis
                if hasattr(analysis_agent, 'github_helper') and analysis_agent.github_helper:
                    # Check if using cached helper
                    if hasattr(analysis_agent.github_helper, 'get_cache_stats'):
                        cache_stats = analysis_agent.github_helper.get_cache_stats()
                        console.print(f"[dim]📊 Cache: {cache_stats['hit_rate']} hit rate, {cache_stats['api_calls_saved']} API calls saved[/dim]")
                    
                    # Original rate limit check
                    if hasattr(analysis_agent.github_helper, 'get_rate_limit_status'):
                        rate_limit_status = analysis_agent.github_helper.get_rate_limit_status()
                        console.print(f"[dim]{rate_limit_status}[/dim]")
                        analysis_agent.github_helper.warn_if_rate_limit_low(threshold=200)
                
                import inspect
                maybe = analysis_agent.analyze(repo_url, system_profile)
                analysis = await maybe if inspect.isawaitable(maybe) else maybe
                try:
                    progress.remove_task(analysis_task)
                except Exception:
                    pass

                # Display analysis results
                _display_analysis_results(analysis)

                # Step 3: Generate resolution
                resolution_task = progress.add_task("💡 Generating solution...", total=None)
                try:
                    # Use enhanced agent if learning is enabled and available
                    learning_enabled = (hasattr(config, 'learning') and 
                                      getattr(config.learning, 'enabled', False)) or config.integrations.llm.enabled
                    if LEARNING_AVAILABLE and learning_enabled:
                        resolution_agent = EnhancedResolutionAgent(config=config)
                    else:
                        resolution_agent = ResolutionAgent(config=config)
                    
                    # Handle both sync and async resolve methods
                    import inspect
                    maybe_resolution = resolution_agent.resolve(
                        analysis, strategy if strategy != "auto" else None
                    )
                    resolution = await maybe_resolution if inspect.isawaitable(maybe_resolution) else maybe_resolution
                    try:
                        progress.remove_task(resolution_task)
                    except Exception:
                        pass

                    # Display resolution
                    _display_resolution(resolution)

                    # Step 4: Save files
                    if output:
                        _save_generated_files(resolution, output)
                    else:
                        # Use default outputs directory with repository name
                        repo_name = f"{analysis.repository.owner}-{analysis.repository.name}"
                        default_output = f"outputs/{repo_name}"
                        _save_generated_files(resolution, default_output)

                    # Step 5: Validation (if requested)
                    if validate:
                        validation_task = progress.add_task(
                            "🧪 Validating solution...", total=None
                        )
                        try:
                            result_maybe = resolution_agent.validate_solution(
                                resolution, analysis
                            )
                            validation_result = await result_maybe if inspect.isawaitable(result_maybe) else result_maybe
                            try:
                                progress.remove_task(validation_task)
                            except Exception:
                                pass
                            _display_validation_results(validation_result)
                        except Exception as e:
                            try:
                                progress.remove_task(validation_task)
                            except Exception:
                                pass
                            console.print(f"[yellow]⚠[/yellow] Validation failed: {str(e)}")

                    # Display knowledge base insights
                    try:
                        _display_knowledge_insights(resolution_agent, analysis)
                    except KeyError as e:
                        console.print(f"[dim]Knowledge insights unavailable: {str(e)}[/dim]")
                    except Exception as e:
                        console.print(f"[dim]Knowledge insights unavailable: {str(e)}[/dim]")

                    console.print("\n[bold green]✅ Analysis complete![/bold green]")

                except Exception as e:
                    progress.remove_task(resolution_task)
                    console.print(f"[red]❌ Resolution failed: {str(e)}[/red]")
                    raise

            except Exception as e:
                progress.remove_task(analysis_task)
                console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")
                raise

    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        raise


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

    repo_table.add_row(
        "Repository", f"{analysis.repository.owner}/{analysis.repository.name}"
    )
    repo_table.add_row("Language", analysis.repository.language or "Unknown")
    repo_table.add_row("Stars", str(analysis.repository.stars))
    repo_table.add_row("Dependencies Found", str(len(analysis.dependencies)))
    repo_table.add_row("Analysis Time", f"{analysis.analysis_time:.2f}s")
    repo_table.add_row("Confidence", f"{analysis.confidence_score:.1%}")

    console.print(repo_table)

    # Dependencies
    if analysis.dependencies:
        console.print(
            f"\n[bold cyan]Dependencies ({len(analysis.dependencies)}):[/bold cyan]"
        )
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
                dep.source,
            )

        console.print(dep_table)

        if len(analysis.dependencies) > 10:
            console.print(f"[dim]... and {len(analysis.dependencies) - 10} more[/dim]")

    # Compatibility issues
    if analysis.compatibility_issues:
        console.print(
            f"\n[bold yellow]Compatibility Issues ({len(analysis.compatibility_issues)}):[/bold yellow]"
        )
        for issue in analysis.compatibility_issues:
            severity_color = {
                "critical": "red",
                "warning": "yellow",
                "info": "blue",
            }.get(issue.severity, "white")
            console.print(f"[{severity_color}]• {issue.message}[/{severity_color}]")
            if issue.suggested_fix:
                console.print(f"  [dim]→ {issue.suggested_fix}[/dim]")


def _display_resolution(resolution: Resolution):
    """Display resolution information."""
    console.print(f"\n[bold green]Solution Generated:[/bold green]")

    console.print(f"Strategy: [cyan]{resolution.strategy.type.value.title()}[/cyan]")
    console.print(f"Files: [white]{len(resolution.generated_files)}[/white]")
    console.print(
        f"Setup Time: [yellow]~{resolution.strategy.estimated_setup_time}s[/yellow]"
    )

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

        with open(file_path, "w") as f:
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
        with open(instructions_path, "w") as f:
            f.write(resolution.instructions)
        console.print(f"• [white]{instructions_path}[/white]")


@main.command()
@click.argument("repo_url")
@click.option("--from-ci", is_flag=True, help="Learn from CI configuration")
def learn(repo_url: str, from_ci: bool) -> None:
    """Learn patterns from repository for future analysis."""
    console.print(f"[blue]📚[/blue] Learning from {repo_url}...")
    # TODO: Implement learning system


@main.command()
@click.option("--show-failures", is_flag=True, help="Show common failure patterns")
def patterns(show_failures: bool) -> None:
    """Show learned patterns and compatibility data."""
    if show_failures:
        console.print("[red]❌[/red] Common failure patterns:")
        # TODO: Show failure patterns from knowledge base
    else:
        console.print("[green]✅[/green] Success patterns:")
        # TODO: Show success patterns from knowledge base


@main.command()
@click.option("--clear", is_flag=True, help="Clear all cached data")
def cache(clear: bool) -> None:
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
    try:
        similar_solutions = resolution_agent.get_similar_solutions(analysis, limit=3)
    except Exception:
        similar_solutions = []

    if isinstance(similar_solutions, list) and all(isinstance(item, dict) for item in similar_solutions):
        if similar_solutions:
            console.print("\n[bold blue]Similar Repositories:[/bold blue]")
            for similar in similar_solutions:
                try:
                    repo_info = similar.get("analysis", {}).get("repository", {})
                    similarity = similar.get("similarity", 0.0)
                    owner = repo_info.get("owner", "unknown")
                    name = repo_info.get("name", "unknown")
                    console.print(
                        f"  • {owner}/{name} (similarity: {similarity:.1%})"
                    )
                except Exception:
                    continue

    try:
        success_patterns = resolution_agent.get_success_patterns("docker")
    except Exception:
        success_patterns = None
    if isinstance(success_patterns, dict) and success_patterns:
        console.print("\n[bold blue]Success Patterns:[/bold blue]")
        console.print(
            f"  • Docker strategy success rate: {success_patterns.get('count', 0)} successful builds"
        )
        avg_time = success_patterns.get("avg_setup_time", 0)
        if isinstance(avg_time, (int, float)) and avg_time > 0:
            console.print(f"  • Average setup time: {avg_time:.0f} seconds")


def _check_token_configuration():
    """Check and warn about missing API tokens."""
    github_token = EnvLoader.get_github_token()
    hf_token = EnvLoader.get_hf_token()
    
    warnings = []
    
    if not github_token:
        warnings.append(
            "🔑 [yellow]GITHUB_TOKEN[/yellow] not found. API rate limit will be 60/hour instead of 5000/hour."
        )
        warnings.append("   Set with: [cyan]export GITHUB_TOKEN=your_token_here[/cyan] or add to .env file")
    
    if not hf_token:
        warnings.append(
            "🤗 [yellow]HF_TOKEN[/yellow] not found. Hugging Face model downloads may be limited."
        )
        warnings.append("   Set with: [cyan]export HF_TOKEN=your_token_here[/cyan] or add to .env file")
    
    if warnings:
        console.print("\n[bold yellow]⚠ Token Configuration Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  {warning}")
        console.print()


@main.command()
@click.option("--github/--no-github", default=True, help="Check GitHub token")
@click.option("--hf/--no-hf", default=True, help="Check Hugging Face token")
def tokens(github: bool, hf: bool) -> None:
    """Check API token configuration and rate limits."""
    console.print("[bold blue]🔑 Token Configuration Status:[/bold blue]\n")
    
    # Show environment loading status
    env_info = load_environment()
    if env_info["dotenv_available"]:
        if env_info["env_loaded"]:
            console.print("[green]📄[/green] .env file loaded successfully\n")
        else:
            console.print("[yellow]📄[/yellow] .env file support available but no .env file found\n")
    else:
        console.print("[dim]📄 .env file support not available (install python-dotenv)[/dim]\n")
    
    if github:
        github_token = EnvLoader.get_github_token()
        if github_token:
            console.print("[green]✅[/green] GitHub token found")
            try:
                from .utils.github import GitHubHelper
                helper = GitHubHelper(github_token)
                status = helper.get_rate_limit_status()
                console.print(f"   {status}")
            except Exception as e:
                console.print(f"   [red]Error checking rate limit: {e}[/red]")
        else:
            console.print("[red]❌[/red] GitHub token not found")
            console.print("   [dim]Set with: export GITHUB_TOKEN=your_token_here or add to .env file[/dim]")
    
    if hf:
        hf_token = EnvLoader.get_hf_token()
        if hf_token:
            console.print("[green]✅[/green] Hugging Face token found")
        else:
            console.print("[red]❌[/red] Hugging Face token not found")
            console.print("   [dim]Set with: export HF_TOKEN=your_token_here or add to .env file[/dim]")
    
    console.print("\n[dim]💡 Tokens improve API rate limits and access to private repositories[/dim]")


@main.command()
def llm() -> None:
    """Test LLM discovery and connection."""
    from .utils.llm_discovery import smart_llm_config
    from .utils.llm import LLMClient
    import asyncio
    
    async def test_llm():
        console.print("[bold blue]🤖 LLM Discovery and Connection Test[/bold blue]\n")
        
        # Test discovery
        discovery_result = await smart_llm_config.get_config()
        
        if discovery_result.get("enabled", False):
            console.print(f"[green]✅ LLM Server Found![/green]")
            console.print(f"   URL: {discovery_result['base_url']}")
            console.print(f"   Model: {discovery_result['model']}")
            console.print(f"   Discovery Method: {discovery_result.get('discovery_method', 'unknown')}")
            
            if 'server_info' in discovery_result:
                server_info = discovery_result['server_info']
                console.print(f"   Server Type: {server_info.get('server_type', 'unknown')}")
                console.print(f"   Models Available: {server_info.get('model_count', 0)}")
            
            # Test connection
            console.print(f"\n[blue]Testing connection...[/blue]")
            client = LLMClient(use_smart_discovery=True)
            available = await client._check_availability()
            
            if available:
                console.print(f"[green]✅ Connection successful![/green]")
                
                # Test completion
                console.print(f"[blue]Testing completion...[/blue]")
                try:
                    response = await client.generate_completion(
                        "What is Python? Answer in one sentence.",
                        max_tokens=50
                    )
                    if response:
                        console.print(f"[green]✅ Response: {response[:100]}...[/green]")
                    else:
                        console.print(f"[yellow]⚠ No response received[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ Completion failed: {e}[/red]")
            else:
                console.print(f"[red]❌ Connection failed[/red]")
        else:
            console.print(f"[red]❌ No LLM server found[/red]")
            console.print(f"[dim]Make sure an LLM server is running on one of the candidate URLs[/dim]")
    
    asyncio.run(test_llm())


@main.command()
def health() -> None:
    """Check system health and agent status."""
    from .agents.contracts import AgentHealthMonitor
    from .agents import ProfileAgent, AnalysisAgent, ResolutionAgent
    from .utils.config import Config
    
    console.print("[bold blue]🏥 System Health Check:[/bold blue]\n")
    
    health_monitor = AgentHealthMonitor()
    config = Config.load()
    
    # Check Profile Agent
    try:
        profile_agent = ProfileAgent(config)
        profile = profile_agent.profile_sync()
        health_monitor.check_agent_health("profile_agent", "healthy", {
            "compute_score": profile.compute_score,
            "has_gpu": profile.has_gpu(),
            "has_cuda": profile.has_cuda(),
        })
    except Exception as e:
        health_monitor.check_agent_health("profile_agent", "unhealthy", {"error": str(e)})
    
    # Check Analysis Agent (basic connectivity)
    try:
        github_token = EnvLoader.get_github_token()
        analysis_agent = AnalysisAgent(config, github_token)
        
        if analysis_agent.github_helper:
            rate_limit = analysis_agent.github_helper.check_rate_limit()
            if rate_limit["remaining"] < 10:
                health_monitor.check_agent_health("analysis_agent", "degraded", {
                    "github_rate_limit": rate_limit["remaining"],
                    "issue": "Low GitHub API rate limit"
                })
            else:
                health_monitor.check_agent_health("analysis_agent", "healthy", {
                    "github_rate_limit": rate_limit["remaining"]
                })
        else:
            health_monitor.check_agent_health("analysis_agent", "degraded", {
                "issue": "No GitHub helper available"
            })
            
    except Exception as e:
        health_monitor.check_agent_health("analysis_agent", "unhealthy", {"error": str(e)})
    
    # Check Resolution Agent
    try:
        resolution_agent = ResolutionAgent(config=config)
        strategy_count = len(resolution_agent.strategies)
        health_monitor.check_agent_health("resolution_agent", "healthy", {
            "strategies_available": strategy_count
        })
    except Exception as e:
        health_monitor.check_agent_health("resolution_agent", "unhealthy", {"error": str(e)})
    
    # Display results
    system_health = health_monitor.get_system_health()
    
    status_colors = {
        "healthy": "green",
        "degraded": "yellow", 
        "unhealthy": "red",
        "unknown": "dim"
    }
    
    overall_color = status_colors.get(system_health["status"], "dim")
    console.print(f"[bold {overall_color}]Overall Status: {system_health['status'].upper()}[/bold {overall_color}]\n")
    
    summary = system_health["summary"]
    console.print(f"[green]✅ Healthy agents: {summary['healthy']}[/green]")
    console.print(f"[yellow]⚠ Degraded agents: {summary['degraded']}[/yellow]")
    console.print(f"[red]❌ Unhealthy agents: {summary['unhealthy']}[/red]\n")
    
    # Show individual agent status
    for agent_name, status in system_health["agents"].items():
        status_color = status_colors.get(status["status"], "dim")
        icon = {"healthy": "✅", "degraded": "⚠", "unhealthy": "❌", "unknown": "❓"}[status["status"]]
        
        console.print(f"[{status_color}]{icon} {agent_name.replace('_', ' ').title()}: {status['status']}[/{status_color}]")
        
        if status["details"]:
            for key, value in status["details"].items():
                console.print(f"   {key}: {value}")
    
    console.print("\n[dim]Use 'repo-doctor tokens' to check API token configuration[/dim]")


@main.command()
@click.option(
    "--knowledge-base",
    type=click.Path(),
    help="Path to knowledge base directory"
)
def learning_dashboard(knowledge_base: Optional[str]) -> None:
    """Show learning system performance dashboard."""
    if not LEARNING_AVAILABLE:
        console.print("[red]❌ Learning system not available. Please install required dependencies.[/red]")
        return
    
    try:
        from pathlib import Path
        from .learning import MLKnowledgeBase, LearningDashboard
        
        # Set up knowledge base path
        kb_path = Path(knowledge_base) if knowledge_base else Path.home() / ".repo-doctor" / "knowledge"
        kb_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize learning dashboard
        ml_kb = MLKnowledgeBase(kb_path)
        dashboard = LearningDashboard(ml_kb)
        
        # Get metrics
        metrics = dashboard.get_dashboard_metrics()
        
        # Display dashboard
        console.print("\n[bold blue]🧠 Learning System Dashboard[/bold blue]\n")
        
        # Status panel
        status_color = "green" if metrics.learning_enabled else "yellow"
        status_text = "Enabled" if metrics.learning_enabled else "Disabled"
        console.print(f"[{status_color}]Status: {status_text}[/{status_color}]\n")
        
        # Metrics table
        table = Table(title="Learning Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Description", style="dim")
        
        table.add_row("Total Analyses", str(metrics.total_analyses), "Number of analyses performed")
        table.add_row("Success Rate", f"{metrics.success_rate:.1%}", "Percentage of successful analyses")
        table.add_row("Model Accuracy", f"{metrics.model_accuracy:.1%}", "ML model prediction accuracy")
        table.add_row("Patterns Discovered", str(metrics.pattern_count), "Number of patterns found")
        table.add_row("Learning Velocity", f"{metrics.learning_velocity:.2f}", "Learning improvement rate")
        table.add_row("Insight Quality", f"{metrics.insight_quality:.1%}", "Quality of generated insights")
        table.add_row("Storage Size", f"{metrics.storage_size_mb:.1f} MB", "Knowledge base size")
        
        console.print(table)
        
        # Recent insights
        console.print("\n[bold blue]🔍 Recent Insights[/bold blue]")
        insights = dashboard.get_recent_insights(limit=5)
        
        if insights:
            for i, insight in enumerate(insights, 1):
                console.print(f"[cyan]{i}.[/cyan] {insight['title']}")
                console.print(f"   [dim]{insight['description']}[/dim]")
                console.print(f"   [dim]Confidence: {insight['confidence']:.1%} | Source: {insight['source']}[/dim]\n")
        else:
            console.print("[dim]No recent insights available. Run some analyses to generate insights.[/dim]")
        
        # Learning recommendations
        console.print("\n[bold blue]💡 Learning Recommendations[/bold blue]")
        recommendations = dashboard.get_learning_recommendations()
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                console.print(f"[yellow]{i}.[/yellow] {rec['recommendation']}")
                console.print(f"   [dim]Priority: {rec['priority']} | Impact: {rec['impact']}[/dim]\n")
        else:
            console.print("[dim]No recommendations available at this time.[/dim]")
            
    except Exception as e:
        console.print(f"[red]❌ Error loading learning dashboard: {e}[/red]")
        console.print("[dim]Make sure the learning system is properly initialized.[/dim]")


if __name__ == "__main__":
    main()
