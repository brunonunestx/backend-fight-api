use std::collections::HashMap;
use std::fs;

use crate::modules::fraud::types::{FraudResult, Transaction};
use super::features::{quantize_vector, vectorize};
use super::ivf::IvfIndex;

const K: usize = 5;
const NPROBE: usize = 8;

pub struct FraudService {
    mcc_risk: HashMap<String, f32>,
    index: IvfIndex,
    vectors: Vec<u8>,
}

impl FraudService {
    pub fn new() -> Self {
        let content = fs::read_to_string("src/dataset/mcc_risk.json")
            .expect("failed to read mcc_risk.json");

        let mcc_risk: HashMap<String, f32> =
            serde_json::from_str(&content).expect("failed to parse mcc_risk.json");

        let bytes = fs::read("src/data/ivf.bin").expect("failed to read ivf.bin");
        let index: IvfIndex =
            bincode::deserialize(&bytes).expect("failed to deserialize ivf index");

        let vectors = fs::read("src/data/vectors.bin").expect("failed to read vectors.bin");

        FraudService { mcc_risk, index, vectors }
    }

    pub fn detect_fraud(&self, transaction: &Transaction) -> FraudResult {
        let vector_f32 = vectorize(transaction, &self.mcc_risk);
        let vector_u8 = quantize_vector(&vector_f32);

        let neighbours =
            self.index.search(&vector_f32, &vector_u8, &self.vectors, K, NPROBE);

        let fraud_count = neighbours.iter().filter(|(_, label)| *label == 1).count();
        let fraud_score = fraud_count as f32 / K as f32;

        FraudResult { approved: fraud_count <= K / 2, fraud_score }
    }
}
