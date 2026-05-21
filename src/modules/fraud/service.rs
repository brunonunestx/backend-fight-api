use crate::modules::fraud::types::{FraudResult, Transaction};

const K: usize = 5;

pub struct FraudService {
}

impl FraudService {
    pub fn new() -> Self {
        FraudService {}
    }

    pub fn detect_fraud(&self, transaction: &Transaction) -> FraudResult {
        let fraud_score = 0.2;
        let fraud_count = 5;

        FraudResult { approved: fraud_count <= K / 2, fraud_score }
    }
}
