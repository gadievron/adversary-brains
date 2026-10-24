# adversary-brains — PoC kit (`brainkit`)

Reference implementation of the adversary-brain schema v0.1: a deterministic detection
engine (`brainkit`), six hand-authored example brains (one YAML each), the JSON Schema they
validate against, and the test suite that exercises the engine.

An *adversary brain* is a small typed graph of entities and required-or-supporting links that
stands as a logical signature of a threat actor's activity, with fire / screen / suspend
semantics that separate "not there" from "not seen."

## Adversary Brains paper
The original paper this repo is a part of can be found here: 

## Check out what we do at Knostic
- Check out what we do at Knostic to secure agents and their supply chain: [https://knostic.ai/](https://knostic.ai/)
- We also have quite a bit of open source: [https://github.com/knostic](https://github.com/knostic)
- One example of which being OpenAnt, the first open source LLM-based vulnerability discovery and remediation project: [https://github.com/knostic/openant/](https://github.com/knostic/openant/)

## What's here
| Path | What it is |
|------|-----------|
| `brainkit/` | The engine: schema validator, matcher (flat `match` + `sequence`), `screen`, `weights`, `coverage`, `extract`/`assess` (LLM record/replay), `discover`/`loop` lifecycle. `validate` is the only CLI subcommand (by design). |
| `brains/` | Six canonical brains (one YAML each): SUNBURST, lure-chain, Scattered Spider, Volt Typhoon, APT1, Akira. |
| `schema/brain-0.1.schema.json` | The normative contract every shipped brain validates against. |
| `data/techniques.json` | Curated ATT&CK technique-id index for the `extract` validation gate. |
| `brainkit/llm-transcripts/` | Committed LLM record/replay transcripts (ADR-010) — replayed deterministically by the tests, no network needed. |
| `tests/` | The deterministic suite (no endpoint, no network). |
| `requirements.txt` | Exact dependency pins: PyYAML, jsonschema, pytest. Python ≥ 3.10. |

## Run
```bash
pip install -r requirements.txt
export PYTHONPATH=.
python -m brainkit validate brains/ --schema schema/brain-0.1.schema.json   # the CLI (validate-only)
python -m pytest tests/ -q                                                   # exercise the whole engine
```
No package install is needed — `python -m brainkit` runs straight from the source tree. The only
CLI subcommand is `validate`; the matcher and lifecycle code are a library the tests drive.

## LLM (optional)
`extract` / `assess` use a record/replay LLM client (ADR-010): the committed transcripts under
`brainkit/llm-transcripts/` are replayed deterministically, so the suite runs with **no endpoint**.
Set `BRAINS_LLM_ENDPOINT` to an OpenAI-compatible URL to record fresh transcripts against a live model.

## License
Apache-2.0 (see `LICENSE`).
