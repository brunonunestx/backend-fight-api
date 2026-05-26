use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use backend_fight::helpers::partition::PartitionFactory;
use backend_fight::helpers::vectors::{normalize, load_partition, VECTOR_DIMENSION, VECTOR_STRIDE};

const IVF_K: usize = 32;
const IVF_ITERATIONS: usize = 20;

#[derive(Debug, serde::Deserialize)]
struct Reference {
    label: String,
    vector: [f32; VECTOR_DIMENSION],
}

fn main() {
    let content = fs::read_to_string("src/dataset/references.json").expect("Failed to read references.json");
    let items: Vec<Reference> = serde_json::from_str(&content).expect("Failed to parse JSON");

    // Remove stale files from both old (16-partition) and current (8-partition) schemes
    let old_partitions = [
        "OFFLINE_CARD_025", "OFFLINE_CARD_050", "OFFLINE_CARD_075", "OFFLINE_CARD_100",
        "OFFLINE_NOCARD_025", "OFFLINE_NOCARD_050", "OFFLINE_NOCARD_075", "OFFLINE_NOCARD_100",
        "ONLINE_CARD_025", "ONLINE_CARD_050", "ONLINE_CARD_075", "ONLINE_CARD_100",
        "ONLINE_NOCARD_025", "ONLINE_NOCARD_050", "ONLINE_NOCARD_075", "ONLINE_NOCARD_100",
    ];
    for name in &old_partitions {
        for ext in &["vec", "lbl", "ivf"] {
            let _ = fs::remove_file(format!("src/data/{}.{}", name, ext));
        }
    }
    for name in PartitionFactory::initialize_partitions() {
        for ext in &["vec", "lbl", "ivf"] {
            let _ = fs::remove_file(format!("src/data/{}.{}", name, ext));
        }
    }

    // Write vectors and labels into partitions
    for (i, item) in items.iter().enumerate() {
        println!("Appending vector {} of {}", i + 1, items.len());

        let is_online: bool = item.vector[9] == 1.0;
        let mcc_risk = item.vector[12];

        let partition = PartitionFactory::get_name(!is_online, mcc_risk);

        let mut vec_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("src/data/{}.vec", partition))
            .expect("Failed to open .vec file");

        let mut lbl_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("src/data/{}.lbl", partition))
            .expect("Failed to open .lbl file");

        let normalized = normalize(&mut item.vector.clone());
        let vec_bytes = unsafe {
            std::slice::from_raw_parts(normalized.as_ptr() as *const u8, VECTOR_DIMENSION)
        };
        let label_byte = if item.label == "fraud" { 1u8 } else { 0u8 };

        vec_file.write_all(vec_bytes).expect("Failed to write .vec");
        lbl_file.write_all(&[label_byte]).expect("Failed to write .lbl");
    }

    // Build IVF index for each partition
    for name in PartitionFactory::initialize_partitions() {
        build_ivf(name);
    }

    println!("Done.");
}

fn build_ivf(partition_name: &str) {
    let partition = match load_partition(partition_name) {
        Some(p) => p,
        None => {
            println!("Skipping IVF for {} (no partition file)", partition_name);
            return;
        }
    };

    let n = partition.vectors.len();
    if n < IVF_K {
        println!("Skipping IVF for {} ({} vectors < k={})", partition_name, n, IVF_K);
        return;
    }

    println!("Building IVF for {} ({} vectors)...", partition_name, n);

    let centroids = kmeans(&partition.vectors, IVF_K, IVF_ITERATIONS);

    // Assign each vector to its nearest centroid
    let mut lists: Vec<Vec<u32>> = vec![vec![]; IVF_K];
    for (vec_idx, vec) in partition.vectors.iter().enumerate() {
        let ci = nearest_centroid(&centroids, vec);
        lists[ci].push(vec_idx as u32);
    }

    write_ivf(partition_name, &centroids, &lists);

    let sizes: Vec<usize> = lists.iter().map(|l| l.len()).collect();
    println!(
        "  done — min_list={}, max_list={}, avg_list={:.1}",
        sizes.iter().min().unwrap_or(&0),
        sizes.iter().max().unwrap_or(&0),
        sizes.iter().sum::<usize>() as f64 / IVF_K as f64,
    );
}

fn l2_sq(a: &[i8; VECTOR_STRIDE], b: &[i8; VECTOR_STRIDE]) -> i32 {
    let mut sum = 0i32;
    for d in 0..VECTOR_DIMENSION {
        let diff = (a[d] as i32) - (b[d] as i32);
        sum += diff * diff;
    }
    sum
}

fn nearest_centroid(centroids: &[[i8; VECTOR_STRIDE]], v: &[i8; VECTOR_STRIDE]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (l2_sq(v, c), i))
        .min_by_key(|&(d, _)| d)
        .map(|(_, i)| i)
        .unwrap_or(0)
}

fn kmeans(vectors: &[[i8; VECTOR_STRIDE]], k: usize, iters: usize) -> Vec<[i8; VECTOR_STRIDE]> {
    let n = vectors.len();
    let step = n / k;

    // Stride-based initialization
    let mut centroids_f: Vec<[f64; VECTOR_STRIDE]> = (0..k)
        .map(|i| {
            let mut c = [0f64; VECTOR_STRIDE];
            for d in 0..VECTOR_DIMENSION {
                c[d] = vectors[i * step][d] as f64;
            }
            c
        })
        .collect();

    // Precompute i8 centroids for distance checks
    let mut centroids_i8: Vec<[i8; VECTOR_STRIDE]> = centroids_f.iter().map(quantize_centroid).collect();

    for iter in 0..iters {
        let mut sums: Vec<[f64; VECTOR_STRIDE]> = vec![[0f64; VECTOR_STRIDE]; k];
        let mut counts: Vec<usize> = vec![0; k];

        for v in vectors {
            let ci = nearest_centroid(&centroids_i8, v);
            for d in 0..VECTOR_DIMENSION {
                sums[ci][d] += v[d] as f64;
            }
            counts[ci] += 1;
        }

        let mut moved = 0usize;
        for i in 0..k {
            if counts[i] == 0 {
                continue;
            }
            for d in 0..VECTOR_DIMENSION {
                let new_val = sums[i][d] / counts[i] as f64;
                if (new_val - centroids_f[i][d]).abs() > 0.5 {
                    moved += 1;
                }
                centroids_f[i][d] = new_val;
            }
            centroids_i8[i] = quantize_centroid(&centroids_f[i]);
        }

        let empty = counts.iter().filter(|&&c| c == 0).count();
        println!("  iter {:2}: moved={}, empty_centroids={}", iter + 1, moved, empty);

        if moved == 0 {
            break;
        }
    }

    centroids_i8
}

fn quantize_centroid(c: &[f64; VECTOR_STRIDE]) -> [i8; VECTOR_STRIDE] {
    let mut v = [0i8; VECTOR_STRIDE];
    for d in 0..VECTOR_DIMENSION {
        v[d] = c[d].round().clamp(-128.0, 127.0) as i8;
    }
    v
}

fn write_ivf(partition_name: &str, centroids: &[[i8; VECTOR_STRIDE]], lists: &[Vec<u32>]) {
    let path = format!("src/data/{}.ivf", partition_name);
    let mut out = Vec::new();

    out.extend_from_slice(&(centroids.len() as u16).to_le_bytes());

    for c in centroids {
        let bytes = unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const u8, VECTOR_STRIDE)
        };
        out.extend_from_slice(bytes);
    }

    for list in lists {
        out.extend_from_slice(&(list.len() as u32).to_le_bytes());
    }

    for list in lists {
        for &idx in list {
            out.extend_from_slice(&idx.to_le_bytes());
        }
    }

    fs::write(&path, &out).expect("Failed to write .ivf");
    println!("  wrote {} ({} bytes)", path, out.len());
}
