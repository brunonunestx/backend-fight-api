pub struct PartitionFactory;

impl PartitionFactory {
    pub fn get_name(is_offline: bool, mcc_risk: f32) -> &'static str {
        let risk_bucket = if mcc_risk <= 0.25 {
            0
        } else if mcc_risk <= 0.50 {
            1
        } else if mcc_risk <= 0.75 {
            2
        } else {
            3
        };

        match (is_offline, risk_bucket) {
            (true,  0) => "OFFLINE_025",
            (true,  1) => "OFFLINE_050",
            (true,  2) => "OFFLINE_075",
            (true,  3) => "OFFLINE_100",
            (false, 0) => "ONLINE_025",
            (false, 1) => "ONLINE_050",
            (false, 2) => "ONLINE_075",
            _          => "ONLINE_100",
        }
    }

    pub fn initialize_partitions() -> Vec<&'static str> {
        vec![
            "OFFLINE_025", "OFFLINE_050", "OFFLINE_075", "OFFLINE_100",
            "ONLINE_025",  "ONLINE_050",  "ONLINE_075",  "ONLINE_100",
        ]
    }
}
