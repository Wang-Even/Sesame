"""Restore hash-verified SuperMimic and mjlab sources used by the course runs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('supermimic_official_5efbf114', 'mjlab_research_dd51a9fc')


def verify(directory, expected):
    for relative, digest in expected.items():
        file = directory / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Missing or modified upstream file: {file}')


def restore(name, destination, archive_dir):
    record = json.loads((ROOT/'third_party'/f'{name}.provenance.json').read_text())
    expected = record['files_sha256']
    target = destination / name
    if target.exists():
        verify(target, expected)
        print(f'Verified {target} ({len(expected)} files)')
        return
    repo = record['repository'].removeprefix('https://github.com/')
    commit = record['commit']
    url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
    prefix = f'{repo.split("/")[-1]}-{commit}/'
    with tempfile.TemporaryDirectory(dir=destination) as temp:
        work = Path(temp)
        archive = work/'archive.tar.gz'
        if archive_dir:
            archive = archive_dir / f'{name}.tar.gz'
        else:
            print(f'Downloading {url}', flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as output:
                shutil.copyfileobj(response, output)
        extracted = work/'source'
        with tarfile.open(archive, 'r:gz') as bundle:
            for relative in expected:
                path = PurePosixPath(relative)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError(f'Invalid manifest path: {relative}')
                member = bundle.getmember(prefix+relative)
                # The upstream AGENTS.md alias points to CLAUDE.md. Materialize the
                # same verified bytes as a regular file, never extract archive links.
                if member.issym() and relative == 'AGENTS.md' and member.linkname == 'CLAUDE.md':
                    member = bundle.getmember(prefix+'CLAUDE.md')
                if not member.isfile():
                    raise ValueError(f'Unexpected archive member: {member.name}')
                output = extracted / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as stream, output.open('wb') as file:
                    shutil.copyfileobj(stream, file)
                if member.mode & 0o111:
                    output.chmod(0o755)
        verify(extracted, expected)
        extracted.rename(target)
    print(f'Restored {target} ({len(expected)} verified files)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT/'third_party')
    parser.add_argument('--archive-dir', type=Path, help='Use local archives named <package>.tar.gz')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in PACKAGES:
        restore(name, args.destination, args.archive_dir)


if __name__ == '__main__':
    main()
