use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

const DIM: usize = 14;
const L: usize = 7;
const K_HASH: usize = 8;
const W: f32 = 1.0;

// Deve ser idêntico ao LshIndex em src/modules/fraud/lsh.rs
#[derive(Serialize, Deserialize)]
struct LshIndex {
    inv_w: f32,
    projections: Vec<Vec<[f32; DIM]>>,
    offsets: Vec<Vec<f32>>,
    tables: Vec<HashMap<u64, Vec<u32>>>,
    labels: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct Record {
    vector: [f32; DIM],
    label: String,
}

fn quantize(value: f32) -> u8 {
    ((value + 1.0) * 127.5) as u8
}

fn label_to_u8(label: &str) -> u8 {
    match label {
        "legit" => 0,
        "fraud" => 1,
        _ => panic!("invalid label: {label}"),
    }
}

fn bucket_key(projections: &[[f32; DIM]], offsets: &[f32], inv_w: f32, v: &[f32; DIM]) -> u64 {
    projections
        .iter()
        .zip(offsets.iter())
        .fold(0u64, |acc, (proj, offset)| {
            let dot: f32 = proj.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let bucket = ((dot + offset) * inv_w).floor() as i32;
            acc.wrapping_mul(2654435761).wrapping_add(bucket as u64)
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = File::open("src/dataset/references.json")?;
    let records: Vec<Record> = serde_json::from_reader(BufReader::new(input))?;
    let total = records.len();
    println!("loaded {total} records");

    std::fs::create_dir_all("src/data")?;

    // --- vectors.bin (u8) + labels.bin ---
    let mut vectors_writer = BufWriter::new(File::create("src/data/vectors.bin")?);
    let mut labels_writer = BufWriter::new(File::create("src/data/labels.bin")?);

    let labels: Vec<u8> = records.iter().map(|r| label_to_u8(&r.label)).collect();

    for (i, record) in records.iter().enumerate() {
        let quantized: [u8; DIM] = std::array::from_fn(|j| quantize(record.vector[j]));
        vectors_writer.write_all(&quantized)?;
        labels_writer.write_all(&[labels[i]])?;

        if (i + 1) % 100_000 == 0 {
            println!("flat: {} / {total}", i + 1);
        }
    }
    vectors_writer.flush()?;
    labels_writer.flush()?;
    println!("flat index written (vectors.bin + labels.bin)");

    // --- lsh index ---
    println!("building lsh index (L={L}, K={K_HASH}, w={W})...");

    let mut rng = SmallRng::seed_from_u64(42);
    let normal = Normal::new(0.0f32, 1.0)?;
    let uniform = Uniform::new(0.0f32, W);

    let projections: Vec<Vec<[f32; DIM]>> = (0..L)
        .map(|_| (0..K_HASH)
            .map(|_| std::array::from_fn(|_| normal.sample(&mut rng)))
            .collect())
        .collect();

    let offsets: Vec<Vec<f32>> = (0..L)
        .map(|_| (0..K_HASH).map(|_| uniform.sample(&mut rng)).collect())
        .collect();

    let mut tables: Vec<HashMap<u64, Vec<u32>>> = vec![HashMap::new(); L];

    for (i, record) in records.iter().enumerate() {
        for t in 0..L {
            let key = bucket_key(&projections[t], &offsets[t], 1.0 / W, &record.vector);
            tables[t].entry(key).or_default().push(i as u32);
        }
        if (i + 1) % 100_000 == 0 {
            println!("lsh: {} / {total}", i + 1);
        }
    }

    let index = LshIndex { inv_w: 1.0 / W, projections, offsets, tables, labels };

    let encoded = bincode::serialize(&index)?;
    std::fs::write("src/data/lsh.bin", &encoded)?;
    println!("lsh.bin: {:.1} MB", encoded.len() as f64 / 1_048_576.0);

    println!("vectors.bin: {:.1} MB", (total * DIM) as f64 / 1_048_576.0);
    println!("done — {total} vectors indexed");
    Ok(())
}
