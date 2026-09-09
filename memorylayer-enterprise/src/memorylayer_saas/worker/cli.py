# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""CLI entry point for the MemoryLayer distributed worker.

Usage::

    memorylayer-worker start --aether-addr localhost:50051
    memorylayer-worker start --workspace _system --specifier worker-1
"""
import asyncio
import uuid

import click

from scitrera_app_framework import get_variables


# -----------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------
MEMORYLAYER_WORKER_AETHER_ADDR = "MEMORYLAYER_WORKER_AETHER_ADDR"
DEFAULT_MEMORYLAYER_WORKER_AETHER_ADDR = "localhost:50051"

MEMORYLAYER_WORKER_WORKSPACE = "MEMORYLAYER_WORKER_WORKSPACE"
DEFAULT_MEMORYLAYER_WORKER_WORKSPACE = "_system"

MEMORYLAYER_WORKER_SPECIFIER = "MEMORYLAYER_WORKER_SPECIFIER"

MEMORYLAYER_WORKER_TASK_TYPES = "MEMORYLAYER_WORKER_TASK_TYPES"


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """MemoryLayer Worker - distributed task execution for MemoryLayer Enterprise."""
    v = get_variables()
    if verbose:
        v.set("LOGGING_LEVEL", "DEBUG")


@cli.command()
@click.option(
    "--aether-addr",
    default=None,
    help="Aether gateway address [default: localhost:50051]",
)
@click.option(
    "--workspace",
    default=None,
    help="Aether workspace [default: _system]",
)
@click.option(
    "--specifier",
    default=None,
    help="Worker specifier/ID [default: auto-generated UUID]",
)
@click.option(
    "--task-types",
    default=None,
    help="Comma-separated task types to handle (empty = all)",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    help="Log level [default: info]",
)
def start(
    aether_addr: str | None,
    workspace: str | None,
    specifier: str | None,
    task_types: str | None,
    log_level: str | None,
) -> None:
    """Start the MemoryLayer distributed worker."""
    from memorylayer_saas.dependencies import preconfigure, initialize_services, shutdown_services
    from memorylayer_saas.worker.runner import WorkerRunner

    v = get_variables()

    if log_level is not None:
        v.set("LOGGING_LEVEL", log_level.upper())

    # Resolve configuration: CLI flag > env var > default
    if aether_addr is None:
        aether_addr = v.environ(
            MEMORYLAYER_WORKER_AETHER_ADDR,
            default=DEFAULT_MEMORYLAYER_WORKER_AETHER_ADDR,
        )
    if workspace is None:
        workspace = v.environ(
            MEMORYLAYER_WORKER_WORKSPACE,
            default=DEFAULT_MEMORYLAYER_WORKER_WORKSPACE,
        )
    if specifier is None:
        env_specifier = v.environ(MEMORYLAYER_WORKER_SPECIFIER, default="")
        specifier = env_specifier if env_specifier else f"worker-{uuid.uuid4().hex[:8]}"

    # Parse task type filter
    task_type_filter: set[str] | None = None
    if task_types is None:
        env_types = v.environ(MEMORYLAYER_WORKER_TASK_TYPES, default="")
        if env_types:
            task_type_filter = {t.strip() for t in env_types.split(",") if t.strip()}
    elif task_types:
        task_type_filter = {t.strip() for t in task_types.split(",") if t.strip()}

    click.echo(f"Starting MemoryLayer worker (specifier={specifier})")
    click.echo(f"  Aether: {aether_addr}")
    click.echo(f"  Workspace: {workspace}")
    click.echo(f"  Task types: {', '.join(sorted(task_type_filter)) if task_type_filter else 'all'}")

    async def _run() -> None:
        # Preconfigure enterprise plugins (registers all plugins including task handlers)
        preconfigure()

        # Initialize all services (plugin framework + async readiness)
        await initialize_services()

        runner = WorkerRunner(
            v=v,
            aether_addr=aether_addr,
            workspace=workspace,
            specifier=specifier,
            task_type_filter=task_type_filter,
        )
        runner.initialize()

        try:
            await runner.run()
        finally:
            await shutdown_services()

    asyncio.run(_run())


def main() -> None:
    """Main entry point for the ``memorylayer-worker`` console script."""
    cli()


if __name__ == "__main__":
    main()
