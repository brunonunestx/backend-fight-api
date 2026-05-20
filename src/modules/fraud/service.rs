use crate::modules::fraud::types::{FraudResult, Transaction};
use super::features::{quantize_vector, vectorize};
use super::ivf::GraphIndex;

const K: usize = 5;
const NPROBE: usize = 2;

pub struct FraudService {
    index: GraphIndex,
}

impl FraudService {
    pub fn new() -> Self {
        let index = GraphIndex::load("src/data/centroids.bin", "src/data/graph.bin");
        FraudService { index }
    }

    pub fn detect_fraud(&self, transaction: &Transaction) -> FraudResult {
        let vector_f32 = vectorize(transaction);
        let vector_u8 = quantize_vector(&vector_f32);

        let fraud_count = self.index.search(&vector_f32, &vector_u8, K, NPROBE);
        let fraud_score = fraud_count as f32 / K as f32;

        FraudResult { approved: fraud_count <= K / 2, fraud_score }
    }
}
