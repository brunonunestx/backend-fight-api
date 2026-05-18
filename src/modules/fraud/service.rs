use std::fs;

use crate::modules::fraud::types::{FraudResult, Transaction};
use super::features::{quantize_vector, vectorize};
use super::ivf::IvfIndex;

const K: usize = 5;
const NPROBE: usize = 2;
const NPROBE_COARSE: usize = 2;

pub struct FraudService {
    index: IvfIndex,
    vectors: Vec<u8>,
}

impl FraudService {
    pub fn new() -> Self {
        let bytes = fs::read("src/data/ivf.bin").expect("failed to read ivf.bin");
        let index: IvfIndex =
            bincode::deserialize(&bytes).expect("failed to deserialize ivf index");

        let vectors = fs::read("src/data/vectors.bin").expect("failed to read vectors.bin");

        FraudService { index, vectors }
    }

    pub fn detect_fraud(&self, transaction: &Transaction) -> FraudResult {
        let vector_f32 = vectorize(transaction);
        let vector_u8 = quantize_vector(&vector_f32);

        let fraud_count = self.index.search(&vector_f32, &vector_u8, &self.vectors, K, NPROBE, NPROBE_COARSE);
        let fraud_score = fraud_count as f32 / K as f32;

        FraudResult { approved: fraud_count <= K / 2, fraud_score }
    }
}
