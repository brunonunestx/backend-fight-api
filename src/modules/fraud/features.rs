use std::collections::HashMap;

use chrono::{Datelike, Timelike};

use super::types::{Transaction, fnv_hash};

const MAX_AMOUNT: f32 = 10_000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1_440.0;
const MAX_KM: f32 = 1_000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

fn limit(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn quantize(value: f32) -> u8 {
    ((value + 1.0) * 127.5) as u8
}

pub fn quantize_vector(vector: &[f32; 14]) -> [u8; 14] {
    std::array::from_fn(|i| quantize(vector[i]))
}

pub fn vectorize(tx: &Transaction, mcc_risk: &HashMap<String, f32>) -> [f32; 14] {
    let requested_at = tx.transaction.requested_at;

    let hour = requested_at.hour() as f32 / 23.0;
    let day = requested_at.weekday().num_days_from_monday() as f32 / 6.0;

    let (minutes_since_last, km_from_last) = match &tx.last_transaction {
        Some(last) => {
            let minutes = (requested_at - last.timestamp).num_minutes() as f32;

            (
                limit(minutes / MAX_MINUTES),
                limit(last.km_from_current as f32 / MAX_KM),
            )
        }
        None => (-1.0, -1.0),
    };

    let unknown_merchant = if tx.customer.known_merchants.contains(&fnv_hash(&tx.merchant.id)) {
        0.0
    } else {
        1.0
    };

    let mcc_risk_val = *mcc_risk.get(&tx.merchant.mcc).unwrap_or(&0.5);

    [
        limit(tx.transaction.amount as f32 / MAX_AMOUNT),
        limit(tx.transaction.installments as f32 / MAX_INSTALLMENTS),
        limit((tx.transaction.amount as f32 / tx.customer.avg_amount as f32) / AMOUNT_VS_AVG_RATIO),
        hour,
        day,
        minutes_since_last,
        km_from_last,
        limit(tx.terminal.km_from_home as f32 / MAX_KM),
        limit(tx.customer.tx_count_24h as f32 / MAX_TX_COUNT_24H),
        if tx.terminal.is_online { 1.0 } else { 0.0 },
        if tx.terminal.card_present { 1.0 } else { 0.0 },
        unknown_merchant,
        mcc_risk_val,
        limit(tx.merchant.avg_amount as f32 / MAX_MERCHANT_AVG_AMOUNT),
    ]
}
