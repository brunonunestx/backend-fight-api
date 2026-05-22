use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Transaction<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(borrow)]
    pub transaction: TransactionDetails<'a>,
    #[serde(borrow)]
    pub customer: Customer<'a>,
    #[serde(borrow)]
    pub merchant: Merchant<'a>,
    pub terminal: Terminal,
    #[serde(borrow)]
    pub last_transaction: Option<LastTransaction<'a>>,
}

#[derive(Deserialize, Debug)]
pub struct TransactionDetails<'a> {
    pub amount: f64,
    pub installments: u32,
    #[serde(borrow)]
    pub requested_at: &'a str,
}

#[derive(Deserialize, Debug)]
pub struct Customer<'a> {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    #[serde(borrow)]
    pub known_merchants: Vec<&'a str>,
}

#[derive(Deserialize, Debug)]
pub struct Merchant<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(borrow)]
    pub mcc: &'a str,
    pub avg_amount: f64,
}

#[derive(Deserialize, Debug)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Deserialize, Debug)]
pub struct LastTransaction<'a> {
    #[serde(borrow)]
    pub timestamp: &'a str,
    pub km_from_current: f64,
}
