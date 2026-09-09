"""MemoryLayer Enterprise CLI entrypoint.

This module provides the CLI for running MemoryLayer with enterprise features.
It layers enterprise plugins on top of the OSS memorylayer-server.

Usage:
    # Run with default enterprise configuration
    python -m memorylayer_saas.cli serve

    # Or via the installed command (if configured in pyproject.toml)
    memorylayer-enterprise serve
"""

import os
import sys

import click
from scitrera_app_framework import get_variables


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """MemoryLayer Enterprise - Memory infrastructure for LLM-powered agents."""
    v = get_variables()  # get variables instance prior to preconfigure() call
    if verbose:
        v.set("LOGGING_LEVEL", "DEBUG")


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
def serve(host: str, port: int, workers: int):
    """Start the MemoryLayer Enterprise server."""
    import uvicorn
    from memorylayer_server.config import (
        MEMORYLAYER_SERVER_HOST, MEMORYLAYER_SERVER_PORT, DEFAULT_MEMORYLAYER_SERVER_HOST, DEFAULT_MEMORYLAYER_SERVER_PORT
    )
    from memorylayer_saas.dependencies import preconfigure
    from memorylayer_server.lifecycle.fastapi import fastapi_app_factory

    # NOTE: THIS IS ALMOST EXACTLY THE SAME AS OSS! and that's CORRECT!
    # The only difference is that we get preconfigure from .dependencies instead of memorylayer_server.dependencies
    # which means that we integrate all the enterprise plugins automatically.

    # preconfigure ensures that plugins are registered
    v, _ = preconfigure()  # TODO: ideally we would support controlling variables instance?
    if host is None:
        host = v.environ(MEMORYLAYER_SERVER_HOST, default=DEFAULT_MEMORYLAYER_SERVER_HOST)
    if port is None:
        port = v.environ(MEMORYLAYER_SERVER_PORT, default=DEFAULT_MEMORYLAYER_SERVER_PORT, type_fn=int)

    # get FastAPI app instance
    app = fastapi_app_factory(v)

    click.echo(f"Starting memorylayer.ai server on {host}:{port}")
    # Collapse k8s probe access lines into a periodic rollup. Probes otherwise
    # monopolise the access log (measured: 596 of 600 lines), which hid the
    # callers during a live incident. Must go through log_config -- uvicorn
    # runs dictConfig at startup and would discard a pre-installed filter.
    # 0 disables the summary and drops probe lines silently.
    from memorylayer_server.access_log import (
        DEFAULT_PROBE_ROLLUP_SECONDS,
        build_uvicorn_log_config,
    )

    probe_rollup_s = v.environ(
        "MEMORYLAYER_ACCESS_LOG_PROBE_ROLLUP_S",
        default=DEFAULT_PROBE_ROLLUP_SECONDS,
        type_fn=int,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        workers=workers,
        log_config=build_uvicorn_log_config(probe_rollup_s),
    )


def _get_default_enterprise_env() -> dict[str, str]:
    """Return a dict of enterprise-specific environment variables and their defaults."""
    from .storage.postgresql import (
        MEMORYLAYER_POSTGRESQL_URL, DEFAULT_MEMORYLAYER_POSTGRESQL_URL,
        MEMORYLAYER_POSTGRESQL_POOL_SIZE, DEFAULT_MEMORYLAYER_POSTGRESQL_POOL_SIZE,
    )
    from .services.tiering.default import (
        MEMORYLAYER_TIERING_MAX_IMPORTANCE, DEFAULT_MAX_IMPORTANCE,
        MEMORYLAYER_TIERING_MAX_ACCESS_COUNT, DEFAULT_MAX_ACCESS_COUNT,
        MEMORYLAYER_TIERING_OLDER_THAN_DAYS, DEFAULT_OLDER_THAN_DAYS,
        MEMORYLAYER_TIERING_WARMUP_ACCESS_THRESHOLD, DEFAULT_WARMUP_ACCESS_THRESHOLD,
    )
    return {
        MEMORYLAYER_POSTGRESQL_URL: DEFAULT_MEMORYLAYER_POSTGRESQL_URL,
        MEMORYLAYER_POSTGRESQL_POOL_SIZE: DEFAULT_MEMORYLAYER_POSTGRESQL_POOL_SIZE,
        MEMORYLAYER_TIERING_MAX_IMPORTANCE: str(DEFAULT_MAX_IMPORTANCE),
        MEMORYLAYER_TIERING_MAX_ACCESS_COUNT: str(DEFAULT_MAX_ACCESS_COUNT),
        MEMORYLAYER_TIERING_OLDER_THAN_DAYS: str(DEFAULT_OLDER_THAN_DAYS),
        MEMORYLAYER_TIERING_WARMUP_ACCESS_THRESHOLD: str(DEFAULT_WARMUP_ACCESS_THRESHOLD),
    }


@cli.command()
def info():
    """Show enterprise configuration information."""

    click.echo("MemoryLayer Enterprise Configuration")
    click.echo("=" * 40)
    click.echo()

    # Show default environment variables
    click.echo("Default Environment Variables:")
    defaults = _get_default_enterprise_env()
    for key, value in defaults.items():
        current = os.environ.get(key, "(not set)")
        click.echo(f"  {key}")
        click.echo(f"    Default: {value}")
        click.echo(f"    Current: {current}")
        click.echo()

    # Show registered plugins
    click.echo("Enterprise Plugins:")
    click.echo("  - PostgreSQLStoragePlugin (storage/postgresql)")
    click.echo("  - EnterpriseMemoryServicePlugin (services/enterprise_memory)")
    click.echo("  - TieringServicePlugin (services/tiering)")
    click.echo("  - CompressionServicePlugin (services/compression)")
    click.echo()

    click.echo("Enterprise Features:")
    click.echo("  - Cold tier storage with LEANN compression")
    click.echo("  - Automatic memory tiering")
    click.echo("  - PostgreSQL with pgvector support")


@cli.command()
@click.option('--url', help='PostgreSQL connection URL')
def migrate(url: str):
    """Run database migrations (Alembic)."""
    import subprocess
    from .dependencies import preconfigure

    # Ensure plugins are registered (this sets up environment properly)
    v, _ = preconfigure()

    # Override PostgreSQL URL if provided
    if url:
        os.environ['MEMORYLAYER_POSTGRESQL_URL'] = url

    alembic_args = ['alembic', 'upgrade', 'head']

    click.echo("Running database migrations...")
    result = subprocess.run(alembic_args, cwd=os.path.dirname(__file__))
    sys.exit(result.returncode)


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
