use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct FraudResult {
    pub approved: bool,
    pub fraud_score: f32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TxDetails {
    pub amount: f64,
    pub installments: u32,
    pub requested_at: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Customer {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    pub known_merchants: std::collections::HashSet<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Transaction {
    pub id: String,
    pub transaction: TxDetails,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<Box<LastTransaction>>,
}
