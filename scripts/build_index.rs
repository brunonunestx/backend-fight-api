use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

const DIM: usize = 14;
const NLIST: usize = 64;
const MAX_ITER: usize = 25;

// Deve ser idêntico ao IvfIndex em src/modules/fraud/ivf.rs
#[derive(Serialize, Deserialize)]
struct IvfIndex {
    centroids: Vec<[f32; DIM]>,
    lists: Vec<Vec<u32>>,
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

fn l2_sq(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| { let d = x - y; d * d }).sum()
}

fn nearest(v: &[f32; DIM], centroids: &[[f32; DIM]]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

fn kmeans(vectors: &[[f32; DIM]], rng: &mut SmallRng) -> Vec<[f32; DIM]> {
    let n = vectors.len();

    // k-means++ initialization
    let first = rng.gen_range(0..n);
    let mut centroids: Vec<[f32; DIM]> = vec![vectors[first]];

    for k in 1..NLIST {
        let dists: Vec<f32> = vectors
            .iter()
            .map(|v| centroids.iter().map(|c| l2_sq(v, c)).fold(f32::MAX, f32::min))
            .collect();

        let total: f32 = dists.iter().sum();
        let mut threshold = Uniform::new(0.0f32, total).sample(rng);

        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen]);
        println!("kmeans++ init: {}/{NLIST}", k + 1);
    }

    // Lloyd's iterations
    let mut assignments = vec![0usize; n];

    for iter in 0..MAX_ITER {
        let mut changed = 0usize;
        for (i, v) in vectors.iter().enumerate() {
            let c = nearest(v, &centroids);
            if c != assignments[i] {
                assignments[i] = c;
                changed += 1;
            }
        }
        println!("kmeans iter {}/{MAX_ITER}: {changed} reassigned", iter + 1);
        if changed == 0 {
            break;
        }

        let mut sums: Vec<[f32; DIM]> = vec![[0.0f32; DIM]; NLIST];
        let mut counts = vec![0usize; NLIST];

        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            for j in 0..DIM {
                sums[c][j] += v[j];
            }
            counts[c] += 1;
        }

        for c in 0..NLIST {
            if counts[c] > 0 {
                for j in 0..DIM {
                    centroids[c][j] = sums[c][j] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = File::open("src/dataset/references.json")?;
    let records: Vec<Record> = serde_json::from_reader(BufReader::new(input))?;
    let total = records.len();
    println!("loaded {total} records");

    std::fs::create_dir_all("src/data")?;

    // --- vectors.bin + labels.bin ---
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

    // --- IVF index ---
    println!("building IVF index (nlist={NLIST}, max_iter={MAX_ITER})...");

    let float_vecs: Vec<[f32; DIM]> = records.iter().map(|r| r.vector).collect();
    let mut rng = SmallRng::seed_from_u64(42);

    let centroids = kmeans(&float_vecs, &mut rng);

    println!("building inverted lists...");
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); NLIST];
    for (i, v) in float_vecs.iter().enumerate() {
        let c = nearest(v, &centroids);
        lists[c].push(i as u32);
    }

    let sizes: Vec<usize> = lists.iter().map(|l| l.len()).collect();
    println!(
        "cluster sizes: min={}, max={}, avg={:.0}",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap(),
        sizes.iter().sum::<usize>() as f64 / NLIST as f64,
    );

    let index = IvfIndex { centroids, lists, labels };
    let encoded = bincode::serialize(&index)?;
    std::fs::write("src/data/ivf.bin", &encoded)?;
    println!("ivf.bin: {:.1} MB", encoded.len() as f64 / 1_048_576.0);
    println!("vectors.bin: {:.1} MB", (total * DIM) as f64 / 1_048_576.0);
    println!("done — {total} vectors indexed");

    Ok(())
}
