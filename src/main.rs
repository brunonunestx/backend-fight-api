use std::sync::Arc;

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use mimalloc::MiMalloc;
use tokio::net::TcpListener;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod modules;
use modules::fraud::service::FraudService;
use modules::handler::handler;

#[tokio::main]
async fn main() {
    let service = Arc::new(FraudService::new());
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();

    println!("server running");

    loop {
        let (stream, _) = listener.accept().await.unwrap();
        let io = TokioIo::new(stream);
        let service = Arc::clone(&service);

        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(move |req| handler(req, Arc::clone(&service))))
                .await
            {
                eprintln!("connection error: {:?}", err);
            }
        });
    }
}
