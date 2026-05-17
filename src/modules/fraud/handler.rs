use axum::{extract::State, Json};
use std::sync::Arc;
use crate::AppState;

use super::types::{FraudResult, Transaction};

pub async fn detect_fraud(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<Transaction>,
) -> Json<FraudResult> {
    let service = Arc::clone(&app_state.fraud_service);
    tokio::task::spawn_blocking(move || Json(service.detect_fraud(&payload)))
        .await
        .unwrap()
}
