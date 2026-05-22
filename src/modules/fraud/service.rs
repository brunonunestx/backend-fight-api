use std::collections::HashMap;
use std::sync::OnceLock;

use crate::modules::fraud::constants::*;
use crate::modules::fraud::types::Transaction;
use backend_fight::helpers::date::{get_day_of_week, get_hour, minutes_between};
use backend_fight::helpers::partition::PartitionFactory;
use backend_fight::helpers::vectors::{load_partition, Partition};

static MCC_RISK: OnceLock<HashMap<String, f64>> = OnceLock::new();

fn mcc_risk_map() -> &'static HashMap<String, f64> {
    MCC_RISK.get_or_init(|| {
        serde_json::from_str(include_str!("../../dataset/mcc_risk.json")).unwrap_or_default()
    })
}

pub struct FraudService {
    partitions: HashMap<&'static str, Partition>,
}

impl FraudService {
    pub fn new() -> Self {
        let partitions = PartitionFactory::initialize_partitions()
            .into_iter()
            .filter_map(|name| load_partition(&name).map(|p| (name, p)))
            .collect();

        FraudService { partitions }
    }

    pub fn detect_fraud(&self, data: &[u8]) -> f64 {
        let tx: Transaction<'_> = serde_json::from_slice(data).unwrap();
        let vetor = self.build_vetor(&tx);

        let mcc_risk = vetor[12] as f32;
        let partition_name = PartitionFactory::get_name(
            !tx.terminal.is_online,
            tx.terminal.card_present,
            mcc_risk,
        );

        0.0
    }

    fn build_vetor(&self, tx: &Transaction) -> [f64; 14] {
        let mcc_risk = *mcc_risk_map().get(tx.merchant.mcc).unwrap_or(&0.5);

        let (pos5, pos6) = match &tx.last_transaction {
            Some(last) => (
                clamp(minutes_between(&last.timestamp, &tx.transaction.requested_at) / MAX_MINUTES as f64),
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
            if tx.customer.known_merchants.contains(&tx.merchant.id) { 0.0 } else { 1.0 }, // Vec<&str>::contains takes &&str
            mcc_risk,
            clamp(tx.merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT),
        ]
    }
}

fn clamp(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}
