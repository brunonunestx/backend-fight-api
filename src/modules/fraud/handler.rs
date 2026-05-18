use axum::{extract::State, Json};
use std::sync::Arc;
use tokio::sync::oneshot;
use crate::AppState;

use super::types::{FraudResult, Transaction};

pub async fn detect_fraud(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<Transaction>,
) -> Json<FraudResult> {
    let (tx, rx) = oneshot::channel();
    let service = app_state.fraud_service.clone();
    app_state.pool.spawn(move || {
        let result = service.detect_fraud(&payload);
        let _ = tx.send(result);
    });
    Json(rx.await.unwrap())
}
