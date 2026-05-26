use std::collections::HashMap;
use std::sync::OnceLock;

use crate::modules::fraud::constants::*;
use crate::modules::fraud::types::Transaction;
use backend_fight::helpers::date::{get_day_of_week, get_hour, minutes_between};
use backend_fight::helpers::partition::PartitionFactory;
use backend_fight::helpers::vectors::{
    IvfIndex, Partition, brute_force_knn, ivf_knn, load_partition, quantize, VECTOR_STRIDE,
};

static MCC_RISK: OnceLock<HashMap<String, f64>> = OnceLock::new();

fn mcc_risk_map() -> &'static HashMap<String, f64> {
    MCC_RISK.get_or_init(|| {
        serde_json::from_str(include_str!("../../dataset/mcc_risk.json")).unwrap_or_default()
    })
}

const NEAR_DISTANCE: f32 = 40.0;
const NEAR_DISTANCE_SQ: i32 = (NEAR_DISTANCE * NEAR_DISTANCE) as i32;
const K: i8 = 5;
const NPROBE: usize = 10;

#[derive(serde::Serialize)]
pub struct DetectFraudResult {
    pub approved: bool,
    pub fraud_score: f32,
}

pub struct FraudService {
    partitions: HashMap<&'static str, Partition>,
    ivf_indexes: HashMap<&'static str, IvfIndex>,
}

impl FraudService {
    pub fn new() -> Self {
        let names = PartitionFactory::initialize_partitions();

        let partitions = names
            .iter()
            .filter_map(|&name| load_partition(name).map(|p| (name, p)))
            .collect();

        let ivf_indexes = names
            .iter()
            .filter_map(|&name| IvfIndex::load(name).map(|idx| (name, idx)))
            .collect();

        FraudService { partitions, ivf_indexes }
    }

    pub fn detect_fraud(&self, data: &[u8]) -> DetectFraudResult {
        let tx: Transaction<'_> = serde_json::from_slice(data).unwrap();
        let vetor = self.build_vetor(&tx);

        let mcc_risk = vetor[12] as f32;
        let partition_name = PartitionFactory::get_name(!tx.terminal.is_online, mcc_risk);

        let partition = self.partitions.get(partition_name);
        let mut offsets: [usize; K as usize] = [0; K as usize];
        let mut index_offsets = 0;

        if let Some(p) = partition {
            let mut query = [0i8; VECTOR_STRIDE];
            for (i, &v) in vetor.iter().enumerate() {
                query[i] = quantize(v as f32);
            }

            let (matched, count) = match self.ivf_indexes.get(partition_name) {
                Some(ivf) => ivf_knn(ivf, p, &query, NEAR_DISTANCE_SQ, NPROBE),
                None => brute_force_knn(p, &query, NEAR_DISTANCE_SQ),
            };

            offsets = matched;
            index_offsets = count;
        }

        let sum = offsets[..index_offsets]
            .iter()
            .map(|&i| partition.and_then(|p| p.labels.get(i)).unwrap_or(&0))
            .sum::<u8>();

        let approved = sum < (K as u8 / 2);
        let fraud_score = sum as f32 / K as f32;

        DetectFraudResult { approved, fraud_score }
    }

    fn build_vetor(&self, tx: &Transaction) -> [f64; 14] {
        let mcc_risk = *mcc_risk_map().get(tx.merchant.mcc).unwrap_or(&0.5);

        let (pos5, pos6) = match &tx.last_transaction {
            Some(last) => (
                clamp(
                    minutes_between(&last.timestamp, &tx.transaction.requested_at)
                        / MAX_MINUTES as f64,
                ),
                clamp(last.km_from_current / MAX_KM),
            ),
            None => (-1.0, -1.0),
        };

        [
            clamp(tx.transaction.amount / MAX_AMOUNT),
            clamp(tx.transaction.installments as f64 / MAX_INSTALLMENTS as f64),
            clamp((tx.transaction.amount / tx.customer.avg_amount) / AMOUNT_VS_AVG_RATIO),
            get_hour(&tx.transaction.requested_at),
            get_day_of_week(&tx.transaction.requested_at),
            pos5,
            pos6,
            clamp(tx.terminal.km_from_home / MAX_KM),
            clamp(tx.customer.tx_count_24h as f64 / MAX_TX_COUNT_24H as f64),
            if tx.terminal.is_online { 1.0 } else { 0.0 },
            if tx.terminal.card_present { 1.0 } else { 0.0 },
            if tx.customer.known_merchants.contains(&tx.merchant.id) {
                0.0
            } else {
                1.0
            },
            mcc_risk,
            clamp(tx.merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT),
        ]
    }
}

fn clamp(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}
