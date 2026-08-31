# Release evidence sidecars

Files in this directory are deliberately excluded from the wheel and sdist.
They attest built bytes, so including them in those bytes would create a
self-reference: changing a recorded checksum would change the archive whose
checksum is being recorded.

The two quadric-section sidecars have different lifetimes:

- `quadric-section-v1-release-manifest.json` is the frozen historical record
  for the published v0.1.1 bytes and component revisions. Main-branch work
  must not relabel it.
- `quadric-section-v1-current-main-manifest.json` follows the latest reviewed
  main-branch implementation. Scheduled and manually dispatched extended
  acceptance verifies this file, so any later production, test,
  documentation, packaging, or example change makes it stale and fails
  closed until a separate evidence-only refresh is reviewed.

Each manifest records four Git identities:

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

Verify the latest main-branch evidence from the repository root with the
pinned release environment:

```bash
python release/verify_quadric_section_release.py \
  --manifest release/quadric-section-v1-current-main-manifest.json \
  --evidence-json /tmp/quadric-section-release-verification.json \
  --failure-artifacts-directory /tmp/quadric-section-release-failure
```

To reproduce a published historical manifest, run the same verifier from the
manifest's attested `merged_main_commit` checkout. The verifier intentionally
does not treat a later production checkout as the released source tree.

The optional failure directory receives the wheel, raw sdist, and normalized
sdist from a hash-mismatching run. Successful verification leaves it empty.

The scheduled, manually dispatched, and published-release
`Extended Quadric Acceptance` workflow verifies the current-main sidecar. A
later checkout may differ from its `merged_main_commit` only under `.github/`
and `release/` until that sidecar is refreshed. The historical release
manifest remains unchanged.
