use std::fs::{self, OpenOptions};
use std::io::Write;
use backend_fight::helpers::partition::PartitionFactory;
use backend_fight::helpers::vectors::normalize;

const VECTOR_DIMENSION: usize = 14;
#[derive(Debug, serde::Deserialize)]
struct Reference {
    label: String,
    vector: [f32; VECTOR_DIMENSION],
}


fn main() {
    let content = fs::read_to_string("src/dataset/references.json").expect("Failed to read references.json");
    let mcc_risk = fs::read_to_string("src/dataset/mcc_risk.json").expect("Failed to read mcc_risk.json");

    println!("{}", mcc_risk);

    let items: Vec<Reference> = serde_json::from_str(&content).expect("Failed to parse JSON");
    let mut index = 0i32;

    for item in &items {
        println!("Appending vector number {} of {}", index + 1, items.len());

        let is_online: bool = item.vector[9] == 1.0;
        let card_present: bool = item.vector[10] == 1.0; 
        let mcc_risk = item.vector[12];    

        let partition = PartitionFactory::get_name(!is_online, card_present, mcc_risk);

        let mut vector_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("src/data/{}.vec", partition))
            .expect("Failed to open .vec file");

        let mut labels_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("src/data/{}.lbl", partition))
            .expect("Failed to open .lbl file");
        
        let normalized_vector = normalize(&mut item.vector.clone());

        let vector_bytes = unsafe {
            std::slice::from_raw_parts(
                normalized_vector.as_ptr() as *const u8,
                std::mem::size_of::<[i8; VECTOR_DIMENSION]>(),
            )
        };

        let label_bytes = if item.label == "fraud" { 1u8 } else { 0u8 };

        vector_file.write_all(vector_bytes).expect("Failed to write to .vec file");
        labels_file.write_all(&[label_bytes]).expect("Failed to write to .lbl file");
        index += 1;
    }
}
