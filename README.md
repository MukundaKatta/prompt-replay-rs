# prompt-replay

[![CI](https://github.com/MukundaKatta/prompt-replay-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/prompt-replay-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Record and replay LLM prompts for A/B testing and model swaps.

`prompt-replay` is a small, dependency-light Rust library for capturing the exact
prompt sent to a language model together with the response it produced. Later you
can replay the same prompt through a different model or provider and **diff the
outputs** to see what changed. This is useful for regression testing prompts,
evaluating model upgrades, and comparing providers on identical inputs.

## Features

- **Record** model name, messages, tools, parameters, and the raw response in an
  append-only, in-memory store. Each record gets a unique id and a timestamp.
- **Query** records by id or by model.
- **Diff** two responses with three modes:
  - `DiffMode::Exact` — exact JSON value equality.
  - `DiffMode::JsonDiff` — a structural diff that lists every changed, added, or
    removed path.
  - `DiffMode::TextSimilarity` — a character-bigram (Sørensen–Dice) similarity
    score in `0.0..=1.0`, with text extracted from Anthropic-style `content`
    blocks when present.
- **Serialize** the whole store (or a single record) to `serde_json::Value` for
  persistence or inspection.

The only runtime dependency is [`serde_json`](https://crates.io/crates/serde_json).

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
prompt-replay = "0.1"
serde_json = "1"
```

Or with cargo:

```sh
cargo add prompt-replay serde_json
```

## Usage

Record a prompt/response pair, then replay the same prompt against a second model
and diff the two responses:

```rust
use prompt_replay::{DiffMode, ReplayStore};
use serde_json::json;

let mut store = ReplayStore::new();

// Record the baseline call.
let id = store.record(
    "claude-sonnet-4-6",
    &[json!({"role": "user", "content": "Say hello"})],
    &[],                  // tools
    &Default::default(),  // params (a serde_json::Map)
    &json!({"content": [{"type": "text", "text": "Hi there!"}]}),
);

let baseline = store.find(&id).unwrap();

// ...later, the same prompt is sent to another model and you get a new response.
let candidate = json!({"content": [{"type": "text", "text": "Hello!"}]});

// Compare the two responses.
let exact = store.diff(&baseline.response, &candidate, DiffMode::Exact);
assert!(!exact.equal); // the wording differs

let similarity = store.diff(&baseline.response, &candidate, DiffMode::TextSimilarity);
let score = similarity.details.as_f64().unwrap();
println!("text similarity: {score:.3}");

// Structural diff: see exactly which JSON paths changed.
let structural = store.diff(&baseline.response, &candidate, DiffMode::JsonDiff);
for path in structural.details.as_array().unwrap() {
    println!("changed: {path}");
}
```

## API overview

### `ReplayStore`

An append-only, in-memory collection of records.

| Method | Description |
| ------ | ----------- |
| `ReplayStore::new()` | Create an empty store. |
| `record(model, messages, tools, params, response) -> String` | Store a record; returns its id. |
| `find(id) -> Option<&PromptRecord>` | Look up a record by id. |
| `all() -> &[PromptRecord]` | All records in insertion order. |
| `by_model(model) -> Vec<&PromptRecord>` | Records matching a model name. |
| `len()` / `is_empty()` | Size helpers. |
| `diff(a, b, mode) -> DiffResult` | Compare two responses. |
| `to_json() -> serde_json::Value` | Serialize every record to a JSON array. |

### `PromptRecord`

Public fields: `id`, `model`, `messages`, `tools`, `params`, `response`,
`timestamp`. Call `to_json()` to serialize a single record.

### `DiffMode` and `DiffResult`

`DiffMode` selects the comparison strategy (`Exact`, `JsonDiff`,
`TextSimilarity`). `diff` returns a `DiffResult` with:

- `mode` — the mode that was used.
- `equal` — whether the two responses are considered equal.
- `details` — mode-specific detail: `Null`/the differing values for `Exact`, a
  JSON array of changed paths for `JsonDiff`, and the similarity score for
  `TextSimilarity`.

## Development

```sh
cargo build
cargo test            # unit + integration + doc tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## License

Licensed under the [MIT](LICENSE) license.
