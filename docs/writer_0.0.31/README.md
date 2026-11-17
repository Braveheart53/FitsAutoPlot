# vszh5_writer 0.0.31 evidence

| File | What it is | SHA-256 |
|---|---|---|
| `vszh5_writer_0.0.30.py` | registry master 0.0.30, byte for byte (the base of 0.0.31) | `26194c8e96572fce2106bb5588d33d7c3f024fbd5169ff626cab2e40aa01f604` |
| `vszh5_writer_0.0.30_to_0.0.31.diff` | unified diff from 0.0.30 to 0.0.31 (1364 lines) | — |
| `../../vszh5_writer.py` | 0.0.31 (RF project master `claude/vszh5_writer.py`) | `23c0612df1fb983e6e583397754e5e009350b04c310b839ebeab017ca4652a07` |

Rebuild and check (Linux, macOS or Git Bash):

```sh
cp vszh5_writer_0.0.30.py rebuilt.py
patch rebuilt.py < vszh5_writer_0.0.30_to_0.0.31.diff
sha256sum rebuilt.py ../../vszh5_writer.py      # both 23c0612d...ca4652a07
```

`tests/test_writer_0031.py` loads `vszh5_writer_0.0.30.py` (SHA-256 checked) and shows that pages
without a date axis or data ranges are written identically by 0.0.30 and 0.0.31.
Release notes: `RELEASE_suite-2026-11-09.md` section 15; RF project
`claude/vszh5-writer-0.0.31-notes.md`.
