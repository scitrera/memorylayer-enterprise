"""Source staging excludes stale build products and unrelated private files."""
import importlib.util
from pathlib import Path


def test_clean_source_staging_omits_stale_packages_and_checkout_secrets(tmp_path):
    spec = importlib.util.spec_from_file_location("build_context", Path(__file__).with_name("build_context.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "checkout"
    for path, content in {
        "src/example/__init__.py": "value = 1",
        "src/example/__pycache__/old.pyc": "stale",
        "src/example.egg-info/SOURCES.txt": "stale",
        "build/lib/private_extension/__init__.py": "stale private module",
        ".env": "PRIVATE=example",
        "pyproject.toml": "[project]\nname = 'example'",
        "LICENSE": "license text",
    }.items():
        target = package / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    staged = tmp_path / "clean"
    module.stage_sources(package, staged)
    files = {str(p.relative_to(staged)) for p in staged.rglob('*') if p.is_file()}
    assert files == {'src/example/__init__.py', 'pyproject.toml', 'LICENSE'}
