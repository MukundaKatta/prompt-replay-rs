//! Integration tests exercising the public API of `prompt-replay` end to end.

use prompt_replay::{DiffMode, ReplayStore};
use serde_json::{json, Map, Value};

/// Build a params map the way a caller would.
fn params(temp: f64) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("temperature".to_string(), json!(temp));
    m.insert("max_tokens".to_string(), json!(256));
    m
}

#[test]
fn record_roundtrips_all_fields() {
    let mut store = ReplayStore::new();
    let messages = vec![json!({"role": "user", "content": "ping"})];
    let tools = vec![json!({"name": "get_time", "description": "current time"})];
    let response = json!({"content": [{"type": "text", "text": "pong"}]});

    let id = store.record("model-x", &messages, &tools, &params(0.2), &response);

    let rec = store.find(&id).expect("record should exist");
    assert_eq!(rec.model, "model-x");
    assert_eq!(rec.messages, messages);
    assert_eq!(rec.tools, tools);
    assert_eq!(rec.params.get("temperature"), Some(&json!(0.2)));
    assert_eq!(rec.response, response);
    assert!(rec.timestamp > 0.0, "timestamp should be set");
    assert!(id.starts_with("prec_"), "id should carry the prec_ prefix");
}

#[test]
fn all_returns_insertion_order() {
    let mut store = ReplayStore::new();
    let a = store.record("m", &[], &[], &Default::default(), &json!(1));
    let b = store.record("m", &[], &[], &Default::default(), &json!(2));
    let c = store.record("m", &[], &[], &Default::default(), &json!(3));

    let ids: Vec<&str> = store.all().iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec![a.as_str(), b.as_str(), c.as_str()]);
    assert_eq!(store.len(), 3);
    assert!(!store.is_empty());
}

#[test]
fn store_to_json_serializes_every_record() {
    let mut store = ReplayStore::new();
    store.record("m1", &[], &[], &params(0.0), &json!({"ok": true}));
    store.record("m2", &[], &[], &Default::default(), &json!({"ok": false}));

    let dumped = store.to_json();
    let arr = dumped.as_array().expect("to_json should be an array");
    assert_eq!(arr.len(), 2);
    // Each entry preserves the model and exposes the documented fields.
    assert_eq!(arr[0]["model"], "m1");
    assert_eq!(arr[1]["model"], "m2");
    for entry in arr {
        for key in [
            "id",
            "model",
            "messages",
            "tools",
            "params",
            "response",
            "timestamp",
        ] {
            assert!(entry.get(key).is_some(), "missing field {key} in {entry}");
        }
    }
}

#[test]
fn exact_diff_reports_both_values_when_unequal() {
    let store = ReplayStore::new();
    let a = json!({"text": "left"});
    let b = json!({"text": "right"});

    let result = store.diff(&a, &b, DiffMode::Exact);
    assert!(!result.equal);
    assert_eq!(result.mode, DiffMode::Exact);
    assert_eq!(result.details["a"], a);
    assert_eq!(result.details["b"], b);

    let same = store.diff(&a, &a, DiffMode::Exact);
    assert!(same.equal);
    assert!(same.details.is_null());
}

#[test]
fn json_diff_finds_nested_array_changes() {
    let store = ReplayStore::new();
    let a = json!({"items": [{"v": 1}, {"v": 2}]});
    let b = json!({"items": [{"v": 1}, {"v": 99}]});

    let result = store.diff(&a, &b, DiffMode::JsonDiff);
    assert!(!result.equal);
    let paths: Vec<String> = result
        .details
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap().to_string())
        .collect();
    assert!(
        paths.iter().any(|p| p.contains("items[1]")),
        "expected a path under items[1], got {paths:?}"
    );
}

#[test]
fn json_diff_reports_added_and_removed_array_elements() {
    let store = ReplayStore::new();
    let a = json!([1, 2, 3]);
    let b = json!([1, 2]);

    let shrunk = store.diff(&a, &b, DiffMode::JsonDiff);
    assert!(!shrunk.equal);
    let removed = shrunk
        .details
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p.as_str().unwrap().contains("removed"));
    assert!(removed, "dropping an element should be reported as removed");

    let grown = store.diff(&b, &a, DiffMode::JsonDiff);
    let added = grown
        .details
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p.as_str().unwrap().contains("added"));
    assert!(added, "appending an element should be reported as added");
}

#[test]
fn text_similarity_is_between_zero_and_one_for_related_text() {
    let store = ReplayStore::new();
    let a = json!({"content": [{"type": "text", "text": "the quick brown fox"}]});
    let b = json!({"content": [{"type": "text", "text": "the quick brown dog"}]});

    let result = store.diff(&a, &b, DiffMode::TextSimilarity);
    let score = result.details.as_f64().unwrap();
    assert!(score > 0.0 && score < 1.0, "score was {score}");
    assert!(!result.equal);
    assert_eq!(result.mode, DiffMode::TextSimilarity);
}

#[test]
fn text_similarity_joins_multiple_text_blocks() {
    let store = ReplayStore::new();
    // Multi-block content should still be extracted and compared as text,
    // not fall back to raw JSON serialization.
    let a = json!({"content": [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]});
    let b = json!({"content": [{"type": "text", "text": "hello world"}]});

    let result = store.diff(&a, &b, DiffMode::TextSimilarity);
    let score = result.details.as_f64().unwrap();
    assert!(
        (score - 1.0).abs() < 1e-9,
        "joined blocks should match the single block exactly, got {score}"
    );
    assert!(result.equal);
}

#[test]
fn by_model_only_returns_matching_records() {
    let mut store = ReplayStore::new();
    store.record("alpha", &[], &[], &Default::default(), &json!(null));
    store.record("beta", &[], &[], &Default::default(), &json!(null));
    store.record("alpha", &[], &[], &Default::default(), &json!(null));

    assert_eq!(store.by_model("alpha").len(), 2);
    assert_eq!(store.by_model("beta").len(), 1);
    assert!(store.by_model("gamma").is_empty());
}
