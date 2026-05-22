use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request, Response};
use hyper::body::Incoming;
use crate::modules::fraud::service::FraudService;

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
            let fraud_score = fraud_service.detect_fraud(&body);

            Ok(Response::new(Full::new(Bytes::from(fraud_score.to_string()))))
        }
        _ => Ok(Response::new(Full::new(Bytes::from("Method Not Allowed")))),
    }
}
