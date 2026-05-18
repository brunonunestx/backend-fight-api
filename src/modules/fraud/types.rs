use std::collections::HashSet;
use std::fmt;

use chrono::{DateTime, Utc};
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
    pub requested_at: DateTime<Utc>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Customer {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    #[serde(deserialize_with = "deserialize_merchant_set")]
    pub known_merchants: HashSet<u64>,
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
    pub timestamp: DateTime<Utc>,
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

// FNV-1a — sem alocação, sem dependência externa
pub fn fnv_hash(s: &str) -> u64 {
    let mut h = 14695981039346656037u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn deserialize_merchant_set<'de, D>(de: D) -> Result<HashSet<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Visitor;

    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = HashSet<u64>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("array of merchant id strings")
        }

        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut set = HashSet::with_capacity(seq.size_hint().unwrap_or(8));
            while let Some(id) = seq.next_element::<String>()? {
                set.insert(fnv_hash(&id));
            }
            Ok(set)
        }
    }

    de.deserialize_seq(Visitor)
}
