mod modules;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use axum::Router;
use std::sync::Arc;

use modules::fraud::{routes::router as fraud_router, service::FraudService};

pub struct AppState {
    pub fraud_service: Arc<FraudService>,}

#[tokio::main]
async fn main() {
    let service = Arc::new(FraudService::new());
    let state = AppState { fraud_service: service.clone() };

    let app = Router::new()
        .merge(fraud_router())
        .route("/ready", axum::routing::get(|| async { "OK" }))
        .with_state(Arc::new(state));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}
