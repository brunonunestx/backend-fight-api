use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;

use crate::modules::fraud::types::Transaction;
use super::features::{quantize_vector, vectorize};
use super::lsh::{LshIndex, DIM};

const K: usize = 5;

pub struct FraudService {
    mcc_risk: HashMap<String, f32>,
    index: LshIndex,
    vectors: Vec<u8>,
    n_points: usize,
    // pool de buffers pré-alocados — requests pegam, usam e devolvem
    // Mutex mantido apenas durante pop/push, não durante a busca
    seen_pool: Mutex<Vec<Vec<u32>>>,
}

impl FraudService {
    pub fn new() -> Self {
        let content = fs::read_to_string("src/dataset/mcc_risk.json")
            .expect("failed to read mcc_risk.json");

        let mcc_risk: HashMap<String, f32> =
            serde_json::from_str(&content).expect("failed to parse mcc_risk.json");

        let bytes = fs::read("src/data/lsh.bin").expect("failed to read lsh.bin");
        let index: LshIndex = bincode::deserialize(&bytes).expect("failed to deserialize lsh index");

        let vectors = fs::read("src/data/vectors.bin").expect("failed to read vectors.bin");
        let n_points = vectors.len() / DIM;

        FraudService {
            mcc_risk,
            index,
            vectors,
            n_points,
            seen_pool: Mutex::new(Vec::new()),
        }
    }

    pub fn detect_fraud(&self, transaction: &Transaction) -> bool {
        let vector_f32 = vectorize(transaction, &self.mcc_risk);
        let vector_u8 = quantize_vector(&vector_f32);

        // pega um buffer do pool ou aloca um novo se o pool estiver vazio
        let mut seen = self.seen_pool.lock().unwrap()
            .pop()
            .unwrap_or_else(|| vec![0u32; self.n_points]);

        // busca roda sem nenhum lock
        let neighbours = self.index.search(&vector_f32, &vector_u8, &self.vectors, &mut seen, K);

        // devolve o buffer ao pool — zerado pela dirty list dentro de search
        self.seen_pool.lock().unwrap().push(seen);

        let fraud_count = neighbours.iter()
            .filter(|(_, label)| *label == 1)
            .count();

        fraud_count > K / 2
    }
}
