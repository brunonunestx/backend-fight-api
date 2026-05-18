FROM rust:1.87-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY scripts ./scripts
RUN RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --bin backend-fight

FROM alpine:3.21
WORKDIR /app
COPY --from=builder /app/target/release/backend-fight ./backend-fight
COPY src/data ./src/data
COPY src/dataset ./src/dataset
CMD ["./backend-fight"]
