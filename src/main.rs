mod modules;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use axum::Router;
use rayon::ThreadPool;
use std::sync::Arc;

use modules::fraud::{routes::router as fraud_router, service::FraudService};

pub struct AppState {
    pub fraud_service: Arc<FraudService>,
    pub pool: Arc<ThreadPool>,
}

#[tokio::main]
async fn main() {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap(),
    );
    let service = Arc::new(FraudService::new());
    let state = Arc::new(AppState { fraud_service: service, pool });

    let app = Router::new()
        .merge(fraud_router())
        .route("/ready", axum::routing::get(|| async { "OK" }))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}
