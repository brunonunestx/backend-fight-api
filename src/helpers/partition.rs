pub struct PartitionFactory;

impl PartitionFactory {
    pub fn get_name(is_offline: bool, card_present: bool, mcc_risk: f32) -> &'static str {
        let risk_bucket = if mcc_risk <= 0.25 {
            0
        } else if mcc_risk <= 0.50 {
            1
        } else if mcc_risk <= 0.75 {
            2
        } else {
            3
        };

        match (is_offline, card_present, risk_bucket) {
            (true,  true,  0) => "OFFLINE_CARD_025",
            (true,  true,  1) => "OFFLINE_CARD_050",
            (true,  true,  2) => "OFFLINE_CARD_075",
            (true,  true,  3) => "OFFLINE_CARD_100",
            (true,  false, 0) => "OFFLINE_NOCARD_025",
            (true,  false, 1) => "OFFLINE_NOCARD_050",
            (true,  false, 2) => "OFFLINE_NOCARD_075",
            (true,  false, 3) => "OFFLINE_NOCARD_100",
            (false, true,  0) => "ONLINE_CARD_025",
            (false, true,  1) => "ONLINE_CARD_050",
            (false, true,  2) => "ONLINE_CARD_075",
            (false, true,  3) => "ONLINE_CARD_100",
            (false, false, 0) => "ONLINE_NOCARD_025",
            (false, false, 1) => "ONLINE_NOCARD_050",
            (false, false, 2) => "ONLINE_NOCARD_075",
            _                 => "ONLINE_NOCARD_100",
        }
    }

    pub fn initialize_partitions() -> Vec<&'static str> {
        vec![
            "OFFLINE_CARD_025", "OFFLINE_CARD_050", "OFFLINE_CARD_075", "OFFLINE_CARD_100",
            "OFFLINE_NOCARD_025", "OFFLINE_NOCARD_050", "OFFLINE_NOCARD_075", "OFFLINE_NOCARD_100",
            "ONLINE_CARD_025", "ONLINE_CARD_050", "ONLINE_CARD_075", "ONLINE_CARD_100",
            "ONLINE_NOCARD_025", "ONLINE_NOCARD_050", "ONLINE_NOCARD_075", "ONLINE_NOCARD_100",
        ]
    }
}
