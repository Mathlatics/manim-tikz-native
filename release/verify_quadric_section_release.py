"""Rebuild and verify the finite-cone v1 release manifest.

This verifier deliberately lives under ``release/``.  ``MANIFEST.in`` prunes
that directory, so the checksum sidecar and the code that verifies it cannot
change the wheel or sdist bytes they attest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
from importlib.metadata import version as distribution_version
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence
import zipfile


MANIFEST_SCHEMA = "manim-quadric-section-release-manifest/v1"
EVIDENCE_SCHEMA = "manim-quadric-section-release-verification/v1"
NORMALIZATION_POLICY = (
    "sorted regular files and links; USTAR; uid/gid 0; root owner/group; "
    "SOURCE_DATE_EPOCH mtime; gzip -n -9 equivalent; gzip OS byte 255"
)
ALLOWED_CURRENT_DRIFT_PREFIXES = (".github/", "release/")
_HEX_OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReleaseManifestVerificationError(RuntimeError):
    """The release sidecar cannot be tied to one reproducible source tree."""


@dataclass(frozen=True, slots=True)
class ProvenanceEvidence:
    implementation_base_commit: str
    implementation_head_commit: str
    implementation_tree_sha: str
    merged_main_commit: str
    current_checkout_commit: str
    current_evidence_only_drift: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "implementationBaseCommit": self.implementation_base_commit,
            "implementationHeadCommit": self.implementation_head_commit,
            "implementationTreeSha": self.implementation_tree_sha,
            "mergedMainCommit": self.merged_main_commit,
            "currentCheckoutCommit": self.current_checkout_commit,
            "currentEvidenceOnlyDrift": list(self.current_evidence_only_drift),
        }


@dataclass(frozen=True, slots=True)
class BuildRunEvidence:
    run_index: int
    wheel_sha256: str
    raw_sdist_sha256: str
    normalized_sdist_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "runIndex": self.run_index,
            "wheelSha256": self.wheel_sha256,
            "rawSdistSha256": self.raw_sdist_sha256,
            "normalizedSdistSha256": self.normalized_sdist_sha256,
        }


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            tuple(arguments),
            cwd=cwd,
            env=None if environment is None else dict(environment),
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ReleaseManifestVerificationError(
            f"command failed: {' '.join(arguments)}{suffix}"
        ) from exc


def _git(repository: Path, *arguments: str) -> str:
    return _run(
        ("git", "-C", str(repository), *arguments),
    ).stdout.strip()


def _required_string(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseManifestVerificationError(
            f"release manifest field {key!r} must be a non-empty string"
        )
    return value.strip()


def _required_object_id(record: Mapping[str, object], key: str) -> str:
    value = _required_string(record, key)
    if _HEX_OBJECT_ID.fullmatch(value) is None:
        raise ReleaseManifestVerificationError(
            f"release manifest field {key!r} is not a full Git object ID"
        )
    return value


def _required_sha256(record: Mapping[str, object], key: str) -> str:
    value = _required_string(record, key)
    if _HEX_SHA256.fullmatch(value) is None:
        raise ReleaseManifestVerificationError(
            f"release manifest field {key!r} is not a SHA-256 digest"
        )
    return value


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseManifestVerificationError(
            f"cannot read release manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReleaseManifestVerificationError(
            "release manifest root must be a JSON object"
        )
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ReleaseManifestVerificationError(
            f"unsupported release manifest schema: {value.get('schema')!r}"
        )
    return value


def _assert_commit_exists(repository: Path, commit: str, label: str) -> None:
    try:
        _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    except ReleaseManifestVerificationError as exc:
        raise ReleaseManifestVerificationError(
            f"{label} {commit!r} is not available in the checkout"
        ) from exc


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        detail = (result.stderr or result.stdout or "").strip()
        raise ReleaseManifestVerificationError(
            "cannot evaluate Git ancestry"
            + (f": {detail}" if detail else "")
        )
    return result.returncode == 0


def verify_provenance(
    repository: Path,
    manifest: Mapping[str, object],
) -> ProvenanceEvidence:
    base = _required_object_id(manifest, "implementation_base_commit")
    head = _required_object_id(manifest, "implementation_head_commit")
    tree = _required_object_id(manifest, "implementation_tree_sha")
    merged = _required_object_id(manifest, "merged_main_commit")
    current = _git(repository, "rev-parse", "HEAD")
    for label, commit in (
        ("implementation base commit", base),
        ("implementation head commit", head),
        ("merged main commit", merged),
        ("current checkout commit", current),
    ):
        _assert_commit_exists(repository, commit, label)

    if not _is_ancestor(repository, base, head):
        raise ReleaseManifestVerificationError(
            "implementation_base_commit is not an ancestor of "
            "implementation_head_commit"
        )
    actual_head_tree = _git(repository, "rev-parse", f"{head}^{{tree}}")
    if actual_head_tree != tree:
        raise ReleaseManifestVerificationError(
            "implementation_tree_sha does not match implementation_head_commit: "
            f"expected {tree}, got {actual_head_tree}"
        )
    if not _is_ancestor(repository, head, merged):
        raise ReleaseManifestVerificationError(
            "merged_main_commit does not contain implementation_head_commit"
        )
    actual_merged_tree = _git(repository, "rev-parse", f"{merged}^{{tree}}")
    if actual_merged_tree != tree:
        raise ReleaseManifestVerificationError(
            "merged_main_commit does not preserve the attested implementation "
            f"tree: expected {tree}, got {actual_merged_tree}"
        )
    if not _is_ancestor(repository, merged, current):
        raise ReleaseManifestVerificationError(
            "current checkout does not descend from merged_main_commit"
        )

    drift = tuple(
        line.strip()
        for line in _git(
            repository,
            "diff",
            "--name-only",
            f"{merged}..{current}",
        ).splitlines()
        if line.strip()
    )
    unexpected = tuple(
        path
        for path in drift
        if not path.startswith(ALLOWED_CURRENT_DRIFT_PREFIXES)
    )
    if unexpected:
        raise ReleaseManifestVerificationError(
            "release manifest is stale; current checkout changes package or "
            "production inputs after merged_main_commit: "
            + ", ".join(unexpected)
        )
    return ProvenanceEvidence(
        implementation_base_commit=base,
        implementation_head_commit=head,
        implementation_tree_sha=tree,
        merged_main_commit=merged,
        current_checkout_commit=current,
        current_evidence_only_drift=drift,
    )


def _parse_version_field(value: object, package: str) -> str:
    if not isinstance(value, str):
        raise ReleaseManifestVerificationError(
            f"build environment field for {package!r} must be a string"
        )
    prefix = f"{package} "
    if not value.startswith(prefix) or not value[len(prefix) :].strip():
        raise ReleaseManifestVerificationError(
            f"build environment field must use {prefix!r}<version>"
        )
    return value[len(prefix) :].strip().split()[0]


def verify_build_environment(
    build_artifacts: Mapping[str, object],
) -> dict[str, object]:
    expected_python = _required_string(build_artifacts, "builder_python")
    expected_versions = {
        "build": _parse_version_field(
            build_artifacts.get("build_frontend"), "build"
        ),
        "setuptools": _parse_version_field(
            build_artifacts.get("build_backend"), "setuptools"
        ),
        "wheel": _parse_version_field(
            build_artifacts.get("wheel_builder"), "wheel"
        ),
        "twine": _parse_version_field(
            build_artifacts.get("twine_check"), "twine"
        ),
    }
    actual_python = platform.python_version()
    actual_versions = {
        package: distribution_version(package) for package in expected_versions
    }
    if actual_python != expected_python:
        raise ReleaseManifestVerificationError(
            "builder Python does not match the manifest: "
            f"expected {expected_python}, got {actual_python}"
        )
    mismatches = tuple(
        f"{package}: expected {expected}, got {actual_versions[package]}"
        for package, expected in expected_versions.items()
        if actual_versions[package] != expected
    )
    if mismatches:
        raise ReleaseManifestVerificationError(
            "build tool versions do not match the manifest: "
            + "; ".join(mismatches)
        )
    if build_artifacts.get("build_isolation") is not False:
        raise ReleaseManifestVerificationError(
            "release verification requires build_isolation=false and pinned "
            "backend versions"
        )
    return {
        "python": actual_python,
        "packages": actual_versions,
        "buildIsolation": False,
    }


def _export_commit(repository: Path, commit: str, destination: Path) -> None:
    archive = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            commit,
        ),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ReleaseManifestVerificationError(
                    f"Git archive contains an unsafe path: {member.name!r}"
                )
        source.extractall(destination, filter="data")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sdist(path: Path, source_date_epoch: int) -> bytes:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = tuple(member.name for member in members)
        if len(names) != len(set(names)):
            raise ReleaseManifestVerificationError(
                "sdist contains duplicate archive paths"
            )
        payloads = {
            member.name: archive.extractfile(member).read()
            for member in members
            if member.isfile()
        }

    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as normalized:
        for original in sorted(members, key=lambda member: member.name):
            if not (
                original.isdir()
                or original.isfile()
                or original.issym()
                or original.islnk()
            ):
                raise ReleaseManifestVerificationError(
                    f"unsupported sdist entry type: {original.name!r}"
                )
            info = tarfile.TarInfo(original.name)
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = source_date_epoch
            if original.isdir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.size = 0
                normalized.addfile(info)
            elif original.isfile():
                payload = payloads[original.name]
                info.type = tarfile.REGTYPE
                info.mode = 0o644
                info.size = len(payload)
                normalized.addfile(info, io.BytesIO(payload))
            else:
                info.type = original.type
                info.mode = 0o777
                info.linkname = original.linkname
                info.size = 0
                normalized.addfile(info)
    compressed = bytearray(
        gzip.compress(stream.getvalue(), compresslevel=9, mtime=0)
    )
    if len(compressed) < 10 or compressed[:3] != b"\x1f\x8b\x08":
        raise ReleaseManifestVerificationError(
            "normalized sdist did not produce a valid gzip header"
        )
    # Python 3.12 delegates mtime=0 compression to zlib, whose header records
    # the host OS (3 on Linux, 19 on macOS). RFC 1952 reserves 255 for
    # unknown, which makes the otherwise identical archive cross-platform.
    compressed[9] = 255
    return bytes(compressed)


def _archive_paths(path: Path) -> tuple[str, ...]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return tuple(archive.namelist())
    with tarfile.open(path, "r:gz") as archive:
        return tuple(member.name for member in archive.getmembers())


def _assert_sidecars_are_excluded(path: Path) -> None:
    offenders = []
    for raw in _archive_paths(path):
        parts = PurePosixPath(raw).parts
        if "release" in parts or ".github" in parts:
            offenders.append(raw)
    if offenders:
        raise ReleaseManifestVerificationError(
            f"{path.name} contains external release sidecars: "
            + ", ".join(offenders[:8])
        )


def _artifact_record(
    build_artifacts: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    record = build_artifacts.get(key)
    if not isinstance(record, dict):
        raise ReleaseManifestVerificationError(
            f"build_artifacts.{key} must be an object"
        )
    _required_string(record, "filename")
    _required_sha256(record, "sha256")
    return record


def _preserve_failure_artifacts(
    directory: Path | None,
    *,
    run_index: int,
    wheel_path: Path,
    sdist_path: Path,
    normalized_sdist: bytes,
) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"run-{run_index}"
    shutil.copy2(wheel_path, directory / f"{prefix}-{wheel_path.name}")
    shutil.copy2(sdist_path, directory / f"{prefix}-{sdist_path.name}")
    (directory / f"{prefix}-normalized-{sdist_path.name}").write_bytes(
        normalized_sdist
    )


def rebuild_and_verify(
    repository: Path,
    manifest: Mapping[str, object],
    provenance: ProvenanceEvidence,
    *,
    build_count: int,
    failure_artifacts_directory: Path | None = None,
) -> tuple[BuildRunEvidence, ...]:
    build_artifacts = manifest.get("build_artifacts")
    if not isinstance(build_artifacts, dict):
        raise ReleaseManifestVerificationError(
            "release manifest build_artifacts must be an object"
        )
    source_date_epoch = build_artifacts.get("source_date_epoch")
    if (
        isinstance(source_date_epoch, bool)
        or not isinstance(source_date_epoch, int)
        or source_date_epoch <= 0
    ):
        raise ReleaseManifestVerificationError(
            "source_date_epoch must be a positive integer"
        )
    wheel_record = _artifact_record(build_artifacts, "wheel")
    sdist_record = _artifact_record(build_artifacts, "sdist")
    if sdist_record.get("normalization") != NORMALIZATION_POLICY:
        raise ReleaseManifestVerificationError(
            "sdist normalization policy does not match the verifier"
        )
    wheel_filename = _required_string(wheel_record, "filename")
    sdist_filename = _required_string(sdist_record, "filename")
    expected_wheel = _required_sha256(wheel_record, "sha256")
    expected_sdist = _required_sha256(sdist_record, "sha256")
    if build_count <= 0:
        raise ReleaseManifestVerificationError("build_count must be positive")

    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    results: list[BuildRunEvidence] = []
    with TemporaryDirectory(prefix="quadric-section-release-") as temporary:
        root = Path(temporary)
        for run_index in range(1, build_count + 1):
            source = root / f"source-{run_index}"
            output = root / f"dist-{run_index}"
            _export_commit(
                repository,
                provenance.merged_main_commit,
                source,
            )
            output.mkdir()
            _run(
                (
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--outdir",
                    str(output),
                ),
                cwd=source,
                environment=environment,
            )
            wheel_path = output / wheel_filename
            sdist_path = output / sdist_filename
            if not wheel_path.is_file() or not sdist_path.is_file():
                raise ReleaseManifestVerificationError(
                    "build did not produce the manifest filenames"
                )
            _assert_sidecars_are_excluded(wheel_path)
            _assert_sidecars_are_excluded(sdist_path)
            wheel_sha = _sha256_file(wheel_path)
            raw_sdist_sha = _sha256_file(sdist_path)
            normalized_sdist = normalize_sdist(
                sdist_path,
                source_date_epoch,
            )
            normalized_sdist_sha = _sha256_bytes(normalized_sdist)
            if wheel_sha != expected_wheel:
                _preserve_failure_artifacts(
                    failure_artifacts_directory,
                    run_index=run_index,
                    wheel_path=wheel_path,
                    sdist_path=sdist_path,
                    normalized_sdist=normalized_sdist,
                )
                raise ReleaseManifestVerificationError(
                    "wheel SHA-256 does not match the manifest: "
                    f"expected {expected_wheel}, got {wheel_sha}"
                )
            if normalized_sdist_sha != expected_sdist:
                _preserve_failure_artifacts(
                    failure_artifacts_directory,
                    run_index=run_index,
                    wheel_path=wheel_path,
                    sdist_path=sdist_path,
                    normalized_sdist=normalized_sdist,
                )
                raise ReleaseManifestVerificationError(
                    "normalized sdist SHA-256 does not match the manifest: "
                    f"expected {expected_sdist}, got {normalized_sdist_sha}"
                )
            if run_index == 1:
                _run(
                    (
                        sys.executable,
                        "-m",
                        "twine",
                        "check",
                        str(wheel_path),
                        str(sdist_path),
                    ),
                )
            results.append(
                BuildRunEvidence(
                    run_index=run_index,
                    wheel_sha256=wheel_sha,
                    raw_sdist_sha256=raw_sdist_sha,
                    normalized_sdist_sha256=normalized_sdist_sha,
                )
            )
    if len({item.wheel_sha256 for item in results}) != 1:
        raise ReleaseManifestVerificationError(
            "repeated wheel builds are not byte-identical"
        )
    if len({item.normalized_sdist_sha256 for item in results}) != 1:
        raise ReleaseManifestVerificationError(
            "repeated normalized sdist builds are not identical"
        )
    return tuple(results)


def verify_release(
    repository: Path,
    manifest_path: Path,
    *,
    build_count: int,
    failure_artifacts_directory: Path | None = None,
) -> dict[str, object]:
    manifest = _load_manifest(manifest_path)
    provenance = verify_provenance(repository, manifest)
    build_artifacts = manifest.get("build_artifacts")
    if not isinstance(build_artifacts, dict):
        raise ReleaseManifestVerificationError(
            "release manifest build_artifacts must be an object"
        )
    environment = verify_build_environment(build_artifacts)
    runs = rebuild_and_verify(
        repository,
        manifest,
        provenance,
        build_count=build_count,
        failure_artifacts_directory=failure_artifacts_directory,
    )
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "manifest": str(manifest_path),
        "provenance": provenance.to_dict(),
        "buildEnvironment": environment,
        "buildRuns": [item.to_dict() for item in runs],
    }


def _write_evidence(path: Path | None, payload: Mapping[str, object]) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(path)


def main(argv: Sequence[str] | None = None) -> int:
    default_repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=default_repository,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/quadric-section-v1-release-manifest.json"),
    )
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--build-count", type=int, default=2)
    parser.add_argument("--failure-artifacts-directory", type=Path)
    arguments = parser.parse_args(argv)
    repository = Path(
        _git(arguments.repository.resolve(), "rev-parse", "--show-toplevel")
    )
    manifest_path = arguments.manifest
    if not manifest_path.is_absolute():
        manifest_path = repository / manifest_path
    try:
        payload = verify_release(
            repository,
            manifest_path.resolve(),
            build_count=arguments.build_count,
            failure_artifacts_directory=arguments.failure_artifacts_directory,
        )
    except Exception as exc:
        failure = {
            "schema": EVIDENCE_SCHEMA,
            "status": "fail",
            "manifest": str(manifest_path),
            "errorType": type(exc).__name__,
            "error": str(exc),
        }
        if arguments.failure_artifacts_directory is not None:
            failure["failureArtifactsDirectory"] = str(
                arguments.failure_artifacts_directory
            )
        _write_evidence(arguments.evidence_json, failure)
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1
    _write_evidence(arguments.evidence_json, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
