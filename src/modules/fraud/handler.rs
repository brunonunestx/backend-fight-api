use axum::{extract::State, Json};
use std::sync::Arc;
use crate::AppState;

use super::types::Transaction;

pub async fn detect_fraud(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<Transaction>,
) -> Json<serde_json::Value> {
    let is_fraud: bool = app_state.fraud_service.detect_fraud(&payload);
    Json(serde_json::json!({ "fraud": is_fraud }))
}
