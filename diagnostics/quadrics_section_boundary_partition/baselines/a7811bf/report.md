# Quadric section-boundary pre-fix baseline

- Source commit: `a7811bf4f7d4adbf2e4078e30d647b40bea6693a`
- Required PR #12 ancestor: `a7811bf4f7d4adbf2e4078e30d647b40bea6693a`
- Production quadric diff from PR #12: `empty`
- Fixed Mobject identities: `18740`
- Identity stable across five states: `True`

The Cairo seam count uses fill-only production rendering. Legitimate role boundaries are eroded by 3 pixels; a remaining interior pixel is counted when its RGB distance from every valid flat-fill composite exceeds 8.0. Solid painter-role errors are deliberately excluded.

| State | Plane fragments | Ray classifications | Behind area | Outside area | Between area | Front area | Opaque seam px | Translucent seam px | Canonical SHA-256 | IDs stable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| mainly_behind | 422 | 874 | 12.806784628647 | 3.175219649890 | 0.069991137279 | 0.014341441439 | 389 | 358 | `0834bdbc1eb1f2f10849f1c8ad6c37da826251ed471c57fc36a93a1c86171902` | True |
| intersects | 3107 | 6059 | 6.833164566799 | 5.183437374153 | 3.803862014364 | 0.459692842944 | 270 | 299 | `720b2cc7743e462a40ed59326b460aeab6bf8a2508a5af0c97fe4170c904163a` | True |
| near_tangent | 3020 | 5821 | 6.997013031991 | 6.715393454866 | 2.161607335257 | 0.133115931945 | 311 | 312 | `a935b20fd9ae6214bc6fc619920670d4aa1c2ea0ea25d783c15711dc879597bd` | True |
| exact_parabola | 3602 | 6919 | 2.569137486543 | 6.480303391568 | 4.640983161268 | 2.316705714680 | 112 | 278 | `aee533e885e45027da182073bf31ec916328de51a9b3996e2f81a588acce62fc` | True |
| mainly_front | 3464 | 6598 | 1.503886280955 | 5.358588488503 | 3.866940083588 | 4.261640179077 | 90 | 290 | `ebab821d4b2db71e14d850e895449dfcbdbd04d63201fd460f33af2e4f439322` | True |

## Initial reference attachments

- `report.md`: `7afad6406a45a12d77da4bd6a018eead4f3560aa26752056c903c0b40f665ccc` (10256 bytes)
- `evidence.json`: `9e970c790b83ccfce89d8776c576bf655ebab8943795941219752f8c2742c112` (76519 bytes)
- `seam_evidence_contact_v2.png`: `4d7ca554632168edc3529dd483e1343707b550e8da0837375b31f1ca69824c62` (131598 bytes)
