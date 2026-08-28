# Release evidence sidecars

Files in this directory are deliberately excluded from the wheel and sdist.
They attest released bytes, so including them in those bytes would create a
self-reference: changing a recorded checksum would change the archive whose
checksum is being recorded.

`quadric-section-v1-release-manifest.json` records four Git identities:

- `implementation_base_commit`: the reviewed base of the implementation PR;
- `implementation_head_commit`: the exact reviewed implementation commit;
- `implementation_tree_sha`: the Git tree owned by that implementation head;
- `merged_main_commit`: the main-branch merge that contains the head without
  changing its tree.

The release verifier checks the ancestry and both tree identities, then
exports `merged_main_commit` into a clean temporary source directory. It uses
the pinned Python/build/setuptools/wheel versions and `SOURCE_DATE_EPOCH`,
builds twice without isolation, checks the wheel byte hash, normalizes the two
sdists according to the manifest policy, checks their hashes, runs `twine
check`, and writes machine-readable evidence.

The normalized gzip header always uses RFC 1952 OS value 255. Python 3.12
otherwise writes a host-specific byte even when the decompressed tar payload
is identical, which would give macOS and Linux different checksums.

Run it from the repository root with the pinned release environment:

```bash
python release/verify_quadric_section_release.py \
  --evidence-json /tmp/quadric-section-release-verification.json \
  --failure-artifacts-directory /tmp/quadric-section-release-failure
```

The optional failure directory receives the wheel, raw sdist, and normalized
sdist from a hash-mismatching run. Successful verification leaves it empty.

The scheduled, manually dispatched, and published-release
`Extended Quadric Acceptance` workflow runs the same command. A later checkout
may differ from `merged_main_commit` only under `.github/` and `release/` until
the manifest is refreshed. Any production, test, documentation, packaging, or
example change therefore makes stale release evidence fail closed.
