"""Build a clean GitHub folder and ZIP from an explicit list of current files.

Run from the project root: python -m checks.build_release
Existing output is not overwritten unless --replace is provided.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def release_files():
    """Map release-relative destinations to sources in either workspace layout."""
    names = {
        'README.md', 'pyproject.toml', '.gitignore', '.gitattributes',
        '.github/workflows/tests.yml',
        'hsvd/__init__.py', 'hsvd/core.py', 'hsvd/_numeric.py', 'hsvd/_types.py',
        'examples/__init__.py', 'examples/cases.py', 'examples/cpca_real_data.py',
    }
    for module in ('hsvd', 'error_handling', 'exact_oracle', 'full_construction',
                   'contest_jacobi', 'contest_protocol', 'cpca_real_data'):
        names.add(f'tests/test_{module}.py')
    for module in ('contest_experiments', 'render_contest_tables',
                   'verify_paper_identities', 'verify_full_construction',
                   'verify_error_bounds', 'build_release'):
        names.add(f'checks/{module}.py')
    for name in ('results.json', 'inputs.npz', 'requirements-tested.txt', 'tables.tex'):
        names.add(f'results/contest/{name}')
    for name in ('data.csv', 'PROVENANCE.md', 'results.json', 'inputs.npz',
                 'computed.npz', 'projection.csv', 'pca_vs_cpca.pdf', 'pca_vs_cpca.png'):
        names.add(f'results/applications/cpca_real/{name}')
    files = {name: ROOT / name for name in names}
    files['paper/nirs_computer_science.tex'] = ROOT / 'paper/nirs_computer_science.tex'
    files['paper/nirs_computer_science.pdf'] = ROOT / 'paper/nirs_computer_science.pdf'
    files['paper/figures/pca_vs_cpca.pdf'] = ROOT / 'results/applications/cpca_real/pca_vs_cpca.pdf'
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Release is incomplete: {missing}')
    return dict(sorted(files.items()))


def build(destination: Path, replace: bool = False):
    destination = destination.resolve()
    folder = destination / 'HSVD'
    archive = destination / 'HSVD.zip'
    if folder == ROOT or ROOT.is_relative_to(folder):
        raise ValueError('Output must not replace the project or its parent')
    if (folder.exists() or archive.exists()) and not replace:
        raise FileExistsError('Output already exists. Choose another destination or use --replace.')
    files = release_files()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hsvd-release-', dir=destination) as temp:
        staging = Path(temp) / 'HSVD'
        staging.mkdir()
        for name, source in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest = ''.join(
            f'{hashlib.sha256((staging/name).read_bytes()).hexdigest()}  {name}\n'
            for name in files
        )
        (staging / 'SHA256SUMS.txt').write_text(manifest, encoding='utf-8')
        pending_zip = Path(temp) / 'HSVD.zip'
        with zipfile.ZipFile(pending_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as package:
            for name in [*files, 'SHA256SUMS.txt']:
                package.write(staging/name, f'HSVD/{name}')
        with zipfile.ZipFile(pending_zip) as package:
            if package.testzip() is not None:
                raise ValueError('ZIP integrity check failed')
            for name in files:
                if package.read(f'HSVD/{name}') != (staging/name).read_bytes():
                    raise ValueError(f'Archive differs from folder: {name}')
        if folder.exists():
            shutil.rmtree(folder)
        staging.replace(folder)
        pending_zip.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(f'Created {folder} and {archive} ({len(files)} files + manifest)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT/'github')
    parser.add_argument('--replace', action='store_true', help='Replace previously generated output')
    args = parser.parse_args()
    build(args.destination, args.replace)


if __name__ == '__main__':
    main()
