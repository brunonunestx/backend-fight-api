use super::handler::detect_fraud;
use axum::{Router, routing::post};
use std::sync::Arc;
use crate::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/fraud-score", post(detect_fraud))
}
