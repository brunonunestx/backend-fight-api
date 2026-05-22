use std::fs;
use backend_fight::helpers::vectors::{normalize, calculate_distance, load_vectors_from_file};

const VECTOR_DIMENSION: usize = 14;

#[derive(Debug, serde::Deserialize)]
struct Reference {
    label: String,
    vector: [f32; VECTOR_DIMENSION],
}

fn main() {
    let content = fs::read_to_string("src/dataset/references.json").expect("Failed to read references.json");
    let items: Vec<Reference> = serde_json::from_str(&content).expect("Failed to parse JSON");

    let first = items.first().expect("No items found");
    let query = normalize(&mut first.vector.clone());

    let vec_files: Vec<_> = fs::read_dir("src/data")
        .expect("Failed to read src/data")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "vec").unwrap_or(false))
        .collect();

    let mut distances: Vec<f32> = Vec::new();

    for entry in &vec_files {
        let path = entry.path();
        let path_str = path.to_str().unwrap();
        let vectors = load_vectors_from_file(path_str);

        for vec in &vectors {
            distances.push(calculate_distance(&query, vec));
        }
    }

    distances.retain(|&d| d > 0.0);
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let top_5_percent = (distances.len() as f32 * 0.05).ceil() as usize;
    let nearest = &distances[..top_5_percent];

    println!("Total vectors: {}", distances.len());
    println!("5% nearest ({} vectors):", top_5_percent);
    println!("  min:    {:.4}", nearest.first().unwrap());
    println!("  max:    {:.4}", nearest.last().unwrap());
    println!("  avg:    {:.4}", nearest.iter().sum::<f32>() / nearest.len() as f32);
}
