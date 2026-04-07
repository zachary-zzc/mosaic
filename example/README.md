# Example

Minimal example data and scripts for verifying the MOSAIC graph construction and query pipeline.

## Files

- **conv_small.json** — Minimal conversation data (single session, 5 messages), same format as `locomo_conv*.json`.
- **qa_small.json** — 2 example QA pairs for query testing.
- **run_example.py** — Load a pre-built example graph and QA, run retrieval (optionally with LLM answering and judging).

## Quick Start (from project root `LongtermMemory/`)

### Run example (requires mosaic dependencies)

```bash
# Install mosaic dependencies
pip install -r mosaic/requirements.txt

# Test loading and TF-IDF retrieval only (no LLM calls)
PYTHONPATH=mosaic python example/run_example.py --no-llm

# Full QA with LLM (requires API key in mosaic/config/config.cfg)
PYTHONPATH=mosaic python example/run_example.py --max-questions 2
```

### API Configuration

Set `ali_api_key` and `ali_base_url` in `mosaic/config/config.cfg`, or use the `MOSAIC_CONFIG_PATH` environment variable to specify a custom config file.

## Notes

- The full graph construction pipeline (from `conv_small.json` to graph) requires LLM; see `mosaic/src/save.py`.
- The pre-built graph in `output/` allows quick testing of the query pipeline.
- Use `--no-llm` to verify graph loading and retrieval without API access.
