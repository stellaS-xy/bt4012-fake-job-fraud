"""Package or verify/unpack the final model without importing TensorFlow.

Only model/preprocessing/calibration files enter the ZIP, never posting data,
case IDs, prediction labels, environment secrets or inner-fold caches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

CORE = frozenset({'mlp_features.joblib', 'mlp.keras', 'mlp_calibrator.joblib',
                  'trees_features.joblib', 'trees.joblib', 'trees_calibrator.joblib'})
FILES = CORE | {'metadata.json', 'runtime_requirements.txt', 'export_verified.json'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_artifact(directory):
    directory = Path(directory)
    if any(not (directory/name).is_file() or (directory/name).is_symlink() for name in FILES):
        raise ValueError('A complete, verified model export is required.')
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata.get('model') != 'calibrated_mlp_extra_trees' or metadata.get('weights') != [.5, .5]:
        raise ValueError('Unexpected model/ensemble weights.')
    hashes = metadata.get('file_hashes', {})
    if set(hashes) != CORE or any(digest(directory/name) != hashes[name] for name in CORE):
        raise ValueError('Model export checksums differ.')
    verified = json.loads((directory/'export_verified.json').read_text())
    if verified.get('test_used') is not False or verified.get('development_probe_rows', 0) < 1:
        raise ValueError('Missing development-only model round-trip verification.')
    return metadata


def package_artifact(directory, destination):
    directory, destination = Path(directory), Path(destination)
    verify_artifact(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(FILES):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, (directory/name).read_bytes())
    result = digest(temporary)
    if destination.exists() and digest(destination) != result:
        temporary.unlink()
        raise FileExistsError('A different release ZIP already exists; do not overwrite it.')
    temporary.replace(destination)
    destination.with_suffix('.sha256').write_text(f'{result}  {destination.name}\n', encoding='utf-8')
    return result


def unpack_artifact(archive_path, destination, expected_sha):
    archive_path, destination = Path(archive_path), Path(destination)
    if len(expected_sha) != 64 or digest(archive_path) != expected_sha.lower():
        raise ValueError('ZIP SHA256 differs from the published checksum.')
    if destination.exists():
        raise FileExistsError('Destination already exists; existing models are never overwritten.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ensemble_unpack_', dir=destination.parent) as temp:
        stage = Path(temp)/'model'; stage.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            if set(archive.namelist()) != FILES or len(archive.namelist()) != len(FILES):
                raise ValueError('Unexpected ZIP contents; refusing extraction.')
            for name in sorted(FILES):
                with archive.open(name) as incoming, (stage/name).open('wb') as output:
                    for chunk in iter(lambda: incoming.read(1024*1024), b''):
                        output.write(chunk)
        verify_artifact(stage)
        os.replace(stage, destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pack = sub.add_parser('pack')
    pack.add_argument('--artifact', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    unpack = sub.add_parser('unpack')
    unpack.add_argument('--archive', type=Path, required=True)
    unpack.add_argument('--destination', type=Path, required=True)
    unpack.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        print(package_artifact(args.artifact, args.output))
    else:
        print('Verified model extracted:', unpack_artifact(args.archive, args.destination, args.sha256))
