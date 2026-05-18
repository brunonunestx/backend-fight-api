use std::{
    fs::File,
    io::{BufWriter, Write},
};

use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const DIM: usize = 14;
const NLIST: usize = 16384;
const NLIST_COARSE: usize = 128;
const MAX_ITER: usize = 25;
const MAX_ITER_COARSE: usize = 50;

#[derive(Serialize, Deserialize)]
struct IvfIndex {
    coarse_centroids: Vec<[f32; DIM]>,
    coarse_to_fine: Vec<Vec<u32>>,
    centroids: Vec<[f32; DIM]>,
    offsets: Vec<(u32, u32)>,
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

fn kmeans(vectors: &[[f32; DIM]], nlist: usize, max_iter: usize, rng: &mut SmallRng) -> Vec<[f32; DIM]> {
    let n = vectors.len();

    let first = rng.gen_range(0..n);
    let mut centroids: Vec<[f32; DIM]> = vec![vectors[first]];
    let mut min_dists: Vec<f32> = vectors.par_iter().map(|v| l2_sq(v, &centroids[0])).collect();

    for k in 1..nlist {
        let new_c = centroids[k - 1];
        min_dists.par_iter_mut().zip(vectors.par_iter()).for_each(|(md, v)| {
            let d = l2_sq(v, &new_c);
            if d < *md { *md = d; }
        });

        let total: f32 = min_dists.iter().sum();
        let mut threshold = Uniform::new(0.0f32, total).sample(rng);

        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen]);
        if k % 100 == 0 || k == nlist - 1 {
            println!("kmeans++ init: {}/{nlist}", k + 1);
        }
    }

    let mut assignments: Vec<usize> = vec![0usize; n];

    for iter in 0..max_iter {
        let new_assignments: Vec<usize> = vectors
            .par_iter()
            .map(|v| nearest(v, &centroids))
            .collect();

        let changed = new_assignments.iter().zip(assignments.iter()).filter(|(a, b)| a != b).count();
        assignments = new_assignments;

        println!("kmeans iter {}/{max_iter}: {changed} reassigned", iter + 1);
        if changed == 0 {
            break;
        }

        let nl = nlist;
        let (sums, counts) = assignments
            .par_iter()
            .zip(vectors.par_iter())
            .fold(
                || (vec![[0.0f32; DIM]; nl], vec![0usize; nl]),
                |(mut sums, mut counts), (&c, v)| {
                    for j in 0..DIM { sums[c][j] += v[j]; }
                    counts[c] += 1;
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![[0.0f32; DIM]; nl], vec![0usize; nl]),
                |(mut s1, mut c1), (s2, c2)| {
                    for c in 0..nl {
                        for j in 0..DIM { s1[c][j] += s2[c][j]; }
                        c1[c] += c2[c];
                    }
                    (s1, c1)
                },
            );

        for c in 0..nlist {
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
    let records: Vec<Record> = serde_json::from_reader(std::io::BufReader::new(input))?;
    let total = records.len();
    println!("loaded {total} records");

    std::fs::create_dir_all("src/data")?;

    let float_vecs: Vec<[f32; DIM]> = records.iter().map(|r| r.vector).collect();
    let raw_labels: Vec<u8> = records.iter().map(|r| label_to_u8(&r.label)).collect();

    // fine IVF centroids
    println!("building fine IVF centroids (nlist={NLIST}, max_iter={MAX_ITER})...");
    let mut rng = SmallRng::seed_from_u64(42);
    let centroids = kmeans(&float_vecs, NLIST, MAX_ITER, &mut rng);

    // coarse centroids
    println!("building coarse centroids (nlist_coarse={NLIST_COARSE}, max_iter={MAX_ITER_COARSE})...");
    let coarse_centroids = kmeans(&centroids, NLIST_COARSE, MAX_ITER_COARSE, &mut rng);

    println!("assigning fine centroids to coarse buckets...");
    let coarse_assignments: Vec<usize> = centroids
        .par_iter()
        .map(|v| nearest(v, &coarse_centroids))
        .collect();

    let mut coarse_to_fine: Vec<Vec<u32>> = vec![Vec::new(); NLIST_COARSE];
    for (fi, &ci) in coarse_assignments.iter().enumerate() {
        coarse_to_fine[ci].push(fi as u32);
    }

    let coarse_sizes: Vec<usize> = coarse_to_fine.iter().map(|l| l.len()).collect();
    println!(
        "coarse bucket sizes: min={}, max={}, avg={:.0}",
        coarse_sizes.iter().min().unwrap(),
        coarse_sizes.iter().max().unwrap(),
        coarse_sizes.iter().sum::<usize>() as f64 / NLIST_COARSE as f64,
    );

    // assign each vector to its nearest fine cluster
    println!("assigning vectors to fine clusters...");
    let cluster_ids: Vec<usize> = float_vecs.par_iter().map(|v| nearest(v, &centroids)).collect();

    let sizes: Vec<usize> = {
        let mut counts = vec![0usize; NLIST];
        for &c in &cluster_ids { counts[c] += 1; }
        counts
    };
    println!(
        "fine cluster sizes: min={}, max={}, avg={:.0}",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap(),
        sizes.iter().sum::<usize>() as f64 / NLIST as f64,
    );

    // compute offsets (start position of each cluster in the reordered vectors.bin)
    let mut offsets: Vec<(u32, u32)> = Vec::with_capacity(NLIST);
    let mut cursor: u32 = 0;
    for &count in &sizes {
        offsets.push((cursor, count as u32));
        cursor += count as u32;
    }

    // place vectors and labels in cluster order
    println!("reordering vectors by cluster...");
    let mut sorted_vectors: Vec<[u8; DIM]> = vec![[0u8; DIM]; total];
    let mut sorted_labels: Vec<u8> = vec![0u8; total];
    let mut write_pos: Vec<u32> = offsets.iter().map(|&(s, _)| s).collect();

    for (i, &c) in cluster_ids.iter().enumerate() {
        let p = write_pos[c] as usize;
        sorted_vectors[p] = std::array::from_fn(|j| quantize(float_vecs[i][j]));
        sorted_labels[p] = raw_labels[i];
        write_pos[c] += 1;
    }

    // write vectors.bin in cluster order
    let mut vectors_writer = BufWriter::new(File::create("src/data/vectors.bin")?);
    for v in &sorted_vectors {
        vectors_writer.write_all(v)?;
    }
    vectors_writer.flush()?;
    println!("vectors.bin written in cluster order ({:.1} MB)", (total * DIM) as f64 / 1_048_576.0);

    let index = IvfIndex { coarse_centroids, coarse_to_fine, centroids, offsets, labels: sorted_labels };
    let encoded = bincode::serialize(&index)?;
    std::fs::write("src/data/ivf.bin", &encoded)?;
    println!("ivf.bin: {:.1} MB", encoded.len() as f64 / 1_048_576.0);
    println!("done — {total} vectors indexed");

    Ok(())
}
