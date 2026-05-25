/*!
prompt-replay: record and replay LLM prompts for A/B testing and model swaps.

Record the exact prompt sent to an LLM and its response. Later replay the same
prompt through a different model or provider and diff the outputs. Three diff
modes: exact equality, JSON structural diff, and character-level similarity.

```rust
use prompt_replay::{ReplayStore, DiffMode};
use serde_json::json;

let mut store = ReplayStore::new();
let id = store.record(
    "claude-sonnet-4-6",
    &[json!({"role": "user", "content": "hello"})],
    &[],
    &Default::default(),
    &json!({"content": [{"type": "text", "text": "Hi there!"}]}),
);
assert_eq!(store.len(), 1);
```
*/

use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("prec_{:016x}_{:04x}", ts.wrapping_mul(2654435761), n)
}

fn now_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---- PromptRecord ---------------------------------------------------------

/// A recorded prompt + response pair.
#[derive(Debug, Clone)]
pub struct PromptRecord {
    pub id: String,
    pub model: String,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub params: Map<String, Value>,
    pub response: Value,
    pub timestamp: f64,
}

impl PromptRecord {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "model": self.model,
            "messages": self.messages,
            "tools": self.tools,
            "params": self.params,
            "response": self.response,
            "timestamp": self.timestamp,
        })
    }
}

// ---- DiffMode -------------------------------------------------------------

/// How to compare two responses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffMode {
    /// Exact value equality.
    Exact,
    /// Structural JSON diff: returns list of changed paths.
    JsonDiff,
    /// Character-level similarity score (0.0–1.0).
    TextSimilarity,
}

// ---- DiffResult -----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiffResult {
    pub mode: DiffMode,
    pub equal: bool,
    /// For Exact: empty if equal. For JsonDiff: list of differing paths. For TextSimilarity: similarity score.
    pub details: Value,
}

fn extract_text(response: &Value) -> String {
    // Try Anthropic content array shape.
    if let Some(arr) = response.get("content").and_then(|v| v.as_array()) {
        let parts: Vec<&str> = arr
            .iter()
            .filter_map(|b| {
                if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                    b.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
            .collect();
        if !parts.is_empty() {
            return parts.join(" ");
        }
    }
    // Fallback: serialize to string.
    serde_json::to_string(response).unwrap_or_default()
}

fn text_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    // Simple bigram similarity (Sørensen–Dice coefficient).
    let bigrams = |s: &str| -> std::collections::HashSet<[char; 2]> {
        let chars: Vec<char> = s.chars().collect();
        chars.windows(2).map(|w| [w[0], w[1]]).collect()
    };
    let ba = bigrams(a);
    let bb = bigrams(b);
    let intersection = ba.intersection(&bb).count();
    if ba.is_empty() && bb.is_empty() {
        return 1.0;
    }
    (2.0 * intersection as f64) / (ba.len() + bb.len()) as f64
}

fn json_diff_paths(a: &Value, b: &Value, path: &str, diffs: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            let mut all_keys: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
            all_keys.extend(ma.keys());
            all_keys.extend(mb.keys());
            for k in all_keys {
                let sub = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{}.{}", path, k)
                };
                match (ma.get(k), mb.get(k)) {
                    (Some(va), Some(vb)) => json_diff_paths(va, vb, &sub, diffs),
                    (Some(_), None) => diffs.push(format!("{}: removed", sub)),
                    (None, Some(_)) => diffs.push(format!("{}: added", sub)),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(aa), Value::Array(ab)) => {
            let max_len = aa.len().max(ab.len());
            for i in 0..max_len {
                let sub = format!("{}[{}]", path, i);
                match (aa.get(i), ab.get(i)) {
                    (Some(va), Some(vb)) => json_diff_paths(va, vb, &sub, diffs),
                    (Some(_), None) => diffs.push(format!("{}: removed", sub)),
                    (None, Some(_)) => diffs.push(format!("{}: added", sub)),
                    (None, None) => {}
                }
            }
        }
        _ => {
            if a != b {
                diffs.push(path.to_owned());
            }
        }
    }
}

// ---- ReplayStore ----------------------------------------------------------

/// Append-only store for prompt records.
#[derive(Debug, Default, Clone)]
pub struct ReplayStore {
    records: Vec<PromptRecord>,
}

impl ReplayStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a prompt + response. Returns the new record's id.
    pub fn record(
        &mut self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        params: &Map<String, Value>,
        response: &Value,
    ) -> String {
        let rec = PromptRecord {
            id: new_id(),
            model: model.to_owned(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
            params: params.clone(),
            response: response.clone(),
            timestamp: now_f64(),
        };
        let id = rec.id.clone();
        self.records.push(rec);
        id
    }

    /// Find a record by id.
    pub fn find(&self, id: &str) -> Option<&PromptRecord> {
        self.records.iter().find(|r| r.id == id)
    }

    /// All records in insertion order.
    pub fn all(&self) -> &[PromptRecord] {
        &self.records
    }

    /// Records for a specific model.
    pub fn by_model(&self, model: &str) -> Vec<&PromptRecord> {
        self.records.iter().filter(|r| r.model == model).collect()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Compare two responses using the given diff mode.
    pub fn diff(&self, a: &Value, b: &Value, mode: DiffMode) -> DiffResult {
        match mode {
            DiffMode::Exact => DiffResult {
                equal: a == b,
                details: if a == b {
                    Value::Null
                } else {
                    serde_json::json!({"a": a, "b": b})
                },
                mode: DiffMode::Exact,
            },
            DiffMode::JsonDiff => {
                let mut diffs = Vec::new();
                json_diff_paths(a, b, "", &mut diffs);
                let equal = diffs.is_empty();
                DiffResult {
                    equal,
                    details: Value::Array(diffs.into_iter().map(Value::String).collect()),
                    mode: DiffMode::JsonDiff,
                }
            }
            DiffMode::TextSimilarity => {
                let ta = extract_text(a);
                let tb = extract_text(b);
                let score = text_similarity(&ta, &tb);
                DiffResult {
                    equal: (score - 1.0).abs() < f64::EPSILON,
                    details: Value::from(score),
                    mode: DiffMode::TextSimilarity,
                }
            }
        }
    }

    pub fn to_json(&self) -> Value {
        Value::Array(self.records.iter().map(|r| r.to_json()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_store() -> (ReplayStore, String) {
        let mut store = ReplayStore::new();
        let id = store.record(
            "model-a",
            &[json!({"role": "user", "content": "hi"})],
            &[],
            &Default::default(),
            &json!({"content": [{"type": "text", "text": "Hello!"}]}),
        );
        (store, id)
    }

    #[test]
    fn record_creates_entry() {
        let (store, _) = make_store();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn find_by_id() {
        let (store, id) = make_store();
        let rec = store.find(&id).unwrap();
        assert_eq!(rec.id, id);
        assert_eq!(rec.model, "model-a");
    }

    #[test]
    fn find_missing_returns_none() {
        let store = ReplayStore::new();
        assert!(store.find("nope").is_none());
    }

    #[test]
    fn by_model_filters() {
        let mut store = ReplayStore::new();
        store.record("m1", &[], &[], &Default::default(), &json!(null));
        store.record("m1", &[], &[], &Default::default(), &json!(null));
        store.record("m2", &[], &[], &Default::default(), &json!(null));
        assert_eq!(store.by_model("m1").len(), 2);
        assert_eq!(store.by_model("m2").len(), 1);
    }

    #[test]
    fn is_empty_initially() {
        let store = ReplayStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn diff_exact_equal() {
        let store = ReplayStore::new();
        let a = json!({"text": "hello"});
        let r = store.diff(&a, &a.clone(), DiffMode::Exact);
        assert!(r.equal);
    }

    #[test]
    fn diff_exact_unequal() {
        let store = ReplayStore::new();
        let a = json!({"text": "hello"});
        let b = json!({"text": "world"});
        let r = store.diff(&a, &b, DiffMode::Exact);
        assert!(!r.equal);
    }

    #[test]
    fn diff_json_diff_detects_change() {
        let store = ReplayStore::new();
        let a = json!({"a": 1, "b": 2});
        let b = json!({"a": 1, "b": 3});
        let r = store.diff(&a, &b, DiffMode::JsonDiff);
        assert!(!r.equal);
        let paths = r.details.as_array().unwrap();
        assert!(paths.iter().any(|p| p.as_str().unwrap().contains("b")));
    }

    #[test]
    fn diff_json_diff_equal() {
        let store = ReplayStore::new();
        let a = json!({"a": 1});
        let r = store.diff(&a, &a.clone(), DiffMode::JsonDiff);
        assert!(r.equal);
        assert_eq!(r.details.as_array().unwrap().len(), 0);
    }

    #[test]
    fn diff_text_similarity_identical() {
        let store = ReplayStore::new();
        let a = json!({"content": [{"type": "text", "text": "hello world"}]});
        let r = store.diff(&a, &a.clone(), DiffMode::TextSimilarity);
        assert!(r.equal);
        assert!((r.details.as_f64().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn diff_text_similarity_different() {
        let store = ReplayStore::new();
        let a = json!({"content": [{"type": "text", "text": "hello"}]});
        let b = json!({"content": [{"type": "text", "text": "goodbye"}]});
        let r = store.diff(&a, &b, DiffMode::TextSimilarity);
        assert!(r.details.as_f64().unwrap() < 1.0);
    }

    #[test]
    fn diff_json_added_key() {
        let store = ReplayStore::new();
        let a = json!({"a": 1});
        let b = json!({"a": 1, "b": 2});
        let r = store.diff(&a, &b, DiffMode::JsonDiff);
        assert!(!r.equal);
    }

    #[test]
    fn diff_json_removed_key() {
        let store = ReplayStore::new();
        let a = json!({"a": 1, "b": 2});
        let b = json!({"a": 1});
        let r = store.diff(&a, &b, DiffMode::JsonDiff);
        assert!(!r.equal);
    }

    #[test]
    fn to_json_is_array() {
        let (store, _) = make_store();
        let j = store.to_json();
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 1);
    }

    #[test]
    fn record_to_json_has_fields() {
        let (store, id) = make_store();
        let rec = store.find(&id).unwrap();
        let j = rec.to_json();
        assert_eq!(j["model"], "model-a");
        assert!(!j["id"].as_str().unwrap().is_empty());
    }

    #[test]
    fn unique_ids() {
        let mut s = ReplayStore::new();
        let a = s.record("m", &[], &[], &Default::default(), &json!(null));
        let b = s.record("m", &[], &[], &Default::default(), &json!(null));
        assert_ne!(a, b);
    }

    #[test]
    fn text_similarity_empty_strings() {
        assert_eq!(text_similarity("", "hello"), 0.0);
        assert_eq!(text_similarity("hello", ""), 0.0);
    }

    #[test]
    fn text_similarity_same() {
        assert_eq!(text_similarity("hello", "hello"), 1.0);
    }
}
