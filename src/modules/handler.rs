use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request, Response};
use hyper::body::Incoming;
use crate::modules::fraud::service::{FraudService, DetectFraudResult};

pub async fn handler(
    req: Request<Incoming>,
    fraud_service: Arc<FraudService>,
) -> Result<Response<Full<Bytes>>, Box<dyn std::error::Error + Send + Sync>> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/ready") => {
            let body_bytes = req.collect().await?.to_bytes();
            Ok(Response::new(Full::new(body_bytes)))
        }
        (&Method::POST, "/fraud-score") => {
            let body = req.collect().await?.to_bytes();
            let result: DetectFraudResult = fraud_service.detect_fraud(&body);

            let json = serde_json::to_vec(&result)?;
            Ok(Response::builder()
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(json)))
                .unwrap())
        }
        _ => Ok(Response::new(Full::new(Bytes::from("Method Not Allowed")))),
    }
}
