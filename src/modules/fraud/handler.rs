use axum::{extract::State, Json};
use std::sync::Arc;
use crate::AppState;

use super::types::{FraudResult, Transaction};

pub async fn detect_fraud(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<Transaction>,
) -> Json<FraudResult> {
    Json(app_state.fraud_service.detect_fraud(&payload))
}
