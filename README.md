# prompt-replay

Record and replay LLM prompts for A/B testing and model swaps.

`prompt-replay` is a small, dependency-light Rust library that captures the exact
prompt sent to a language model (messages, tools, parameters) together with its
response. You can later replay the same prompt through a different model or
provider and diff the outputs to see what changed.

## Why

When you swap models, tweak prompts, or compare providers, you want a precise,
reproducible answer to "did the output actually change, and how?". This crate
gives you an append-only store of prompt/response pairs plus three ways to
compare responses.

## Features

- **Record** prompt + response pairs into an in-memory append-only store, each
  tagged with a unique id, model name, and timestamp.
- **Look up** records by id or filter them by model.
- **Diff** two responses with three modes:
  - `Exact` — strict value equality.
  - `JsonDiff` — structural JSON diff that reports the changed/added/removed paths.
  - `TextSimilarity` — character-bigram (Sørensen–Dice) similarity score in `0.0..=1.0`,
    with built-in extraction of text from Anthropic-style `content` arrays.
- **Serialize** the store (or an individual record) to `serde_json::Value`.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
prompt-replay = "0.1"
serde_json = "1"
```

## Usage

```rust
use prompt_replay::{ReplayStore, DiffMode};
use serde_json::json;

let mut store = ReplayStore::new();

// Record a prompt and the response it produced.
let id = store.record(
    "claude-sonnet-4-6",
    &[json!({"role": "user", "content": "hello"})],
    &[],                       // tools
    &Default::default(),       // params
    &json!({"content": [{"type": "text", "text": "Hi there!"}]}),
);

assert_eq!(store.len(), 1);

// Fetch it back.
let record = store.find(&id).unwrap();
assert_eq!(record.model, "claude-sonnet-4-6");

// Compare two responses with a structural JSON diff.
let a = json!({"content": [{"type": "text", "text": "Hi there!"}]});
let b = json!({"content": [{"type": "text", "text": "Hello!"}]});
let result = store.diff(&a, &b, DiffMode::JsonDiff);
assert!(!result.equal);
println!("changed paths: {}", result.details);
```

### Diff modes

```rust
use prompt_replay::{ReplayStore, DiffMode};
use serde_json::json;

let store = ReplayStore::new();
let a = json!({"content": [{"type": "text", "text": "hello world"}]});
let b = json!({"content": [{"type": "text", "text": "hello there"}]});

// Exact equality.
let exact = store.diff(&a, &b, DiffMode::Exact);

// Structural JSON diff: details is an array of changed paths.
let structural = store.diff(&a, &b, DiffMode::JsonDiff);

// Text similarity: details is a score in 0.0..=1.0.
let similarity = store.diff(&a, &b, DiffMode::TextSimilarity);
println!("similarity = {}", similarity.details);
```

## API overview

- `ReplayStore` — append-only store of records.
  - `new()`, `record(...)`, `find(id)`, `all()`, `by_model(model)`, `len()`,
    `is_empty()`, `diff(a, b, mode)`, `to_json()`.
- `PromptRecord` — a recorded prompt + response pair (`id`, `model`, `messages`,
  `tools`, `params`, `response`, `timestamp`), with `to_json()`.
- `DiffMode` — `Exact`, `JsonDiff`, `TextSimilarity`.
- `DiffResult` — `mode`, `equal`, and `details`.

## Tech stack

- Rust (edition 2021)
- [`serde_json`](https://crates.io/crates/serde_json) for JSON values

## Building and testing

```sh
cargo build
cargo test
```

## License

Licensed under the MIT License. See the `license` field in `Cargo.toml`.
