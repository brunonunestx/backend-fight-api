use std::{fs::File, io::{BufWriter, Write}};

use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use serde::Deserialize;

const DIM: usize = 14;
const NLIST: usize = 16384;
const N_COARSE: usize = 64; // sqrt(NLIST) — 64 coarse + 64 fine = 128 L2/query
const MAX_ITER: usize = 25;

#[derive(Debug, Deserialize)]
struct Record {
    vector: [f32; DIM],
    label: String,
}

fn label_to_u8(label: &str) -> u8 {
    match label {
        "legit" => 0,
        "fraud" => 1,
        _ => panic!("invalid label: {label}"),
    }
}

fn quantize(v: f32) -> u8 {
    ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8
}

fn l2_sq(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| { let d = x - y; d * d }).sum()
}

fn nearest_idx(v: &[f32; DIM], cs: &[[f32; DIM]]) -> usize {
    cs.iter().enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

fn kmeans(vectors: &[[f32; DIM]], nlist: usize, max_iter: usize, rng: &mut SmallRng) -> Vec<[f32; DIM]> {
    use rand_distr::{Distribution, Uniform};

    let n = vectors.len();
    assert!(nlist <= n, "nlist={nlist} > n={n}");

    let first = rng.gen_range(0..n);
    let mut centroids = vec![vectors[first]];
    let mut min_dists: Vec<f32> = vectors.par_iter().map(|v| l2_sq(v, &centroids[0])).collect();

    for k in 1..nlist {
        let prev = centroids[k - 1];
        min_dists.par_iter_mut().zip(vectors.par_iter()).for_each(|(md, v)| {
            let d = l2_sq(v, &prev);
            if d < *md { *md = d; }
        });

        let total: f32 = min_dists.iter().sum();
        let mut t = Uniform::new(0.0f32, total).sample(rng);
        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            t -= d;
            if t <= 0.0 { chosen = i; break; }
        }
        centroids.push(vectors[chosen]);

        if (k + 1) % 512 == 0 || k + 1 == nlist {
            println!("  kmeans++ init {}/{nlist}", k + 1);
        }
    }

    let mut assignments = vec![0usize; n];

    for iter in 0..max_iter {
        let new_asgn: Vec<usize> = vectors.par_iter()
            .map(|v| nearest_idx(v, &centroids))
            .collect();

        let changed = new_asgn.iter().zip(&assignments).filter(|(a, b)| a != b).count();
        assignments = new_asgn;
        println!("  iter {}/{max_iter}: {changed} reassigned", iter + 1);
        if changed == 0 { break; }

        let nl = nlist;
        let (sums, counts) = assignments.par_iter().zip(vectors.par_iter())
            .fold(
                || (vec![[0f32; DIM]; nl], vec![0usize; nl]),
                |(mut s, mut c), (&ci, v)| {
                    for j in 0..DIM { s[ci][j] += v[j]; }
                    c[ci] += 1;
                    (s, c)
                },
            )
            .reduce(
                || (vec![[0f32; DIM]; nl], vec![0usize; nl]),
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
                for j in 0..DIM { centroids[c][j] = sums[c][j] / counts[c] as f32; }
            }
        }
    }

    centroids
}

// centroids.bin: [n:u32][dim:u32][n × dim × f32]
fn write_centroids_bin(centroids: &[[f32; DIM]], path: &str) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(centroids.len() as u32).to_le_bytes())?;
    w.write_all(&(DIM as u32).to_le_bytes())?;
    for c in centroids {
        for &x in c { w.write_all(&x.to_le_bytes())?; }
    }
    w.flush()
}

// coarse.bin format:
//   [n_coarse:u32][n_fine:u32][dim:u32]   12 bytes
//   [n_coarse × dim × f32]                coarse centroids
//   per coarse bucket (n_coarse entries):
//     [count:u32]
//     [count × (cluster_id:u32, dim × f32)]
fn write_coarse_bin(
    coarse_centroids: &[[f32; DIM]],
    fine_centroids: &[[f32; DIM]],
    coarse_asgn: &[usize],   // for each fine centroid: which coarse bucket it belongs to
    path: &str,
) -> std::io::Result<()> {
    let n_coarse = coarse_centroids.len();
    let n_fine = fine_centroids.len();

    // quantize fine centroids to u8 — L1 via abs_diff (PSADBW) at query time
    let mut buckets: Vec<Vec<(u32, [u8; DIM])>> = vec![Vec::new(); n_coarse];
    for (fine_id, &ci) in coarse_asgn.iter().enumerate() {
        let q: [u8; DIM] = std::array::from_fn(|j| quantize(fine_centroids[fine_id][j]));
        buckets[ci].push((fine_id as u32, q));
    }

    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(n_coarse as u32).to_le_bytes())?;
    w.write_all(&(n_fine as u32).to_le_bytes())?;
    w.write_all(&(DIM as u32).to_le_bytes())?;

    // coarse centroids as u8
    for c in coarse_centroids {
        let q: [u8; DIM] = std::array::from_fn(|j| quantize(c[j]));
        w.write_all(&q)?;
    }

    for bucket in &buckets {
        w.write_all(&(bucket.len() as u32).to_le_bytes())?;
        for &(cluster_id, ref centroid) in bucket {
            w.write_all(&cluster_id.to_le_bytes())?;
            w.write_all(centroid)?;
        }
    }

    w.flush()
}

// ivf.bin format:
//   [n_clusters:u32][dim:u32]                       8 bytes header
//   n_clusters × [offset:u64][count:u32][_pad:u32]  16 bytes each (offset table)
//   data: cluster i at offset[i]: [count×DIM×u8][count×u8]
fn write_ivf_bin(
    n_centroids: usize,
    float_vecs: &[[f32; DIM]],
    raw_labels: &[u8],
    cluster_ids: &[usize],
    path: &str,
) -> std::io::Result<()> {
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_centroids];
    for (i, &ci) in cluster_ids.iter().enumerate() {
        clusters[ci].push(i);
    }

    let data_start = 8u64 + n_centroids as u64 * 16;
    let mut offsets = Vec::with_capacity(n_centroids);
    let mut cur = data_start;
    for cluster in &clusters {
        offsets.push(cur);
        let n = cluster.len();
        cur += (n * DIM + n) as u64;
    }

    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(n_centroids as u32).to_le_bytes())?;
    w.write_all(&(DIM as u32).to_le_bytes())?;

    for (i, cluster) in clusters.iter().enumerate() {
        w.write_all(&offsets[i].to_le_bytes())?;
        w.write_all(&(cluster.len() as u32).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
    }

    for cluster in &clusters {
        for &vi in cluster {
            let q: [u8; DIM] = std::array::from_fn(|j| quantize(float_vecs[vi][j]));
            w.write_all(&q)?;
        }
        for &vi in cluster {
            w.write_all(&[raw_labels[vi]])?;
        }
    }

    w.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();

    let input = File::open("src/dataset/references.json")?;
    let records: Vec<Record> = serde_json::from_reader(std::io::BufReader::new(input))?;
    let n = records.len();
    println!("loaded {n} records");
    println!("nlist={NLIST} (~{} vectors/cluster avg)", n / NLIST);

    std::fs::create_dir_all("src/data")?;

    let float_vecs: Vec<[f32; DIM]> = records.iter().map(|r| r.vector).collect();
    let raw_labels: Vec<u8> = records.iter().map(|r| label_to_u8(&r.label)).collect();

    println!("building {NLIST} fine centroids (max_iter={MAX_ITER})...");
    let mut rng = SmallRng::seed_from_u64(42);
    let fine_centroids = kmeans(&float_vecs, NLIST, MAX_ITER, &mut rng);

    println!("assigning vectors to fine clusters...");
    let cluster_ids: Vec<usize> = float_vecs.par_iter()
        .map(|v| nearest_idx(v, &fine_centroids))
        .collect();

    let sizes: Vec<usize> = {
        let mut c = vec![0usize; NLIST];
        for &ci in &cluster_ids { c[ci] += 1; }
        c
    };
    let mut sorted_sizes = sizes.clone();
    sorted_sizes.sort_unstable();
    let p99 = sorted_sizes[sorted_sizes.len() * 99 / 100];
    println!(
        "fine cluster sizes: min={} max={} avg={:.0} p99={}",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap(),
        sizes.iter().sum::<usize>() as f64 / NLIST as f64,
        p99,
    );

    println!("building {N_COARSE} coarse centroids over the {NLIST} fine centroids...");
    let coarse_centroids = kmeans(&fine_centroids, N_COARSE, MAX_ITER, &mut rng);

    let coarse_asgn: Vec<usize> = fine_centroids.par_iter()
        .map(|c| nearest_idx(c, &coarse_centroids))
        .collect();

    let coarse_sizes: Vec<usize> = {
        let mut c = vec![0usize; N_COARSE];
        for &ci in &coarse_asgn { c[ci] += 1; }
        c
    };
    println!(
        "coarse bucket sizes: min={} max={} avg={:.0}",
        coarse_sizes.iter().min().unwrap(),
        coarse_sizes.iter().max().unwrap(),
        coarse_sizes.iter().sum::<usize>() as f64 / N_COARSE as f64,
    );

    println!("writing centroids.bin...");
    write_centroids_bin(&fine_centroids, "src/data/centroids.bin")?;
    println!("  {:.2} MB", (NLIST * DIM * 4) as f64 / 1_048_576.0);

    println!("writing coarse.bin...");
    write_coarse_bin(&coarse_centroids, &fine_centroids, &coarse_asgn, "src/data/coarse.bin")?;
    let coarse_size = std::fs::metadata("src/data/coarse.bin")?.len();
    println!("  {:.2} KB", coarse_size as f64 / 1_024.0);

    println!("writing ivf.bin...");
    write_ivf_bin(NLIST, &float_vecs, &raw_labels, &cluster_ids, "src/data/ivf.bin")?;
    let ivf_size = std::fs::metadata("src/data/ivf.bin")?.len();
    println!("  {:.2} MB", ivf_size as f64 / 1_048_576.0);

    println!("done — {n} vectors → {N_COARSE} coarse × {NLIST} fine clusters");
    println!("search cost: {N_COARSE} + ~{} L2/query", NLIST / N_COARSE);
    Ok(())
}
