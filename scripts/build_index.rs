use std::{fs::File, io::{BufWriter, Write}};

use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use serde::Deserialize;

const DIM: usize = 14;
const SLOTS_PER_CLUSTER: usize = 10;
const M: usize = 2;
const N_ENTRY_POINTS: usize = 128;
const MAX_ITER: usize = 25;
// each coarse bucket holds ~GRAPH_COARSE_FACTOR fine centroids during graph construction
const GRAPH_COARSE_FACTOR: usize = 300;

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

fn l2(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    l2_sq(a, b).sqrt()
}

fn nearest_idx(v: &[f32; DIM], cs: &[[f32; DIM]]) -> usize {
    cs.iter().enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

fn kmeans(vectors: &[[f32; DIM]], nlist: usize, max_iter: usize, rng: &mut SmallRng) -> Vec<[f32; DIM]> {
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

        if (k + 1) % 10_000 == 0 || k + 1 == nlist {
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

// For small n: brute-force O(n²) parallel.
// For large n: coarse bucketing — find M nearest within same coarse bucket + 2 adjacent.
fn build_knn_graph(centroids: &[[f32; DIM]], rng: &mut SmallRng) -> Vec<[(u32, f32); M]> {
    let n = centroids.len();

    if n <= 20_000 {
        println!("  brute-force k-NN ({n} nodes)...");
        return centroids.par_iter().enumerate()
            .map(|(i, ci)| {
                let mut best: [(u32, f32); M] = [(u32::MAX, f32::MAX); M];
                for (j, cj) in centroids.iter().enumerate() {
                    if j == i { continue; }
                    let d = l2(ci, cj);
                    if d < best[M - 1].1 {
                        best[M - 1] = (j as u32, d);
                        best.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    }
                }
                best
            })
            .collect();
    }

    let coarse_n = (n / GRAPH_COARSE_FACTOR).clamp(64, 4096);
    println!("  coarse k-NN index: {coarse_n} buckets (~{GRAPH_COARSE_FACTOR} per bucket)...");
    let coarse_centroids = kmeans(centroids, coarse_n, 10, rng);

    let coarse_asgn: Vec<usize> = centroids.par_iter()
        .map(|c| nearest_idx(c, &coarse_centroids))
        .collect();

    let mut coarse_buckets: Vec<Vec<usize>> = vec![Vec::new(); coarse_n];
    for (i, &ci) in coarse_asgn.iter().enumerate() {
        coarse_buckets[ci].push(i);
    }

    // 2 nearest coarse neighbours per coarse centroid (for expanding search)
    let coarse_adj: Vec<[usize; 2]> = coarse_centroids.par_iter().enumerate()
        .map(|(i, c)| {
            let mut best = [(usize::MAX, f32::MAX); 2];
            for (j, other) in coarse_centroids.iter().enumerate() {
                if j == i { continue; }
                let d = l2_sq(c, other);
                if d < best[1].1 {
                    best[1] = (j, d);
                    if best[1].1 < best[0].1 { best.swap(0, 1); }
                }
            }
            [best[0].0, best[1].0]
        })
        .collect();

    println!("  finding M={M} neighbors per centroid...");
    centroids.par_iter().enumerate()
        .map(|(i, ci_vec)| {
            let ci = coarse_asgn[i];
            let mut best: [(u32, f32); M] = [(u32::MAX, f32::MAX); M];

            for bucket_ci in [ci, coarse_adj[ci][0], coarse_adj[ci][1]] {
                if bucket_ci == usize::MAX { continue; }
                for &j in &coarse_buckets[bucket_ci] {
                    if j == i { continue; }
                    let d = l2(ci_vec, &centroids[j]);
                    if d < best[M - 1].1 {
                        best[M - 1] = (j as u32, d);
                        best.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    }
                }
            }
            best
        })
        .collect()
}

fn farthest_point_sampling(centroids: &[[f32; DIM]], n: usize, rng: &mut SmallRng) -> Vec<u32> {
    let total = centroids.len();
    let mut eps: Vec<u32> = Vec::with_capacity(n);
    let first = rng.gen_range(0..total);
    eps.push(first as u32);

    let mut min_dists: Vec<f32> = centroids.iter()
        .map(|c| l2(c, &centroids[first]))
        .collect();

    for _ in 1..n {
        let farthest = min_dists.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        eps.push(farthest as u32);

        let ep_vec = &centroids[farthest];
        min_dists.par_iter_mut().zip(centroids.par_iter()).for_each(|(md, c)| {
            let d = l2(c, ep_vec);
            if d < *md { *md = d; }
        });
    }

    eps
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

// ivf.bin: [n:u32][slots:u32][dim:u32] then per cluster: [count:u32][slots×dim×u8][slots×u8]
// stride = 4 + SLOTS_PER_CLUSTER × (DIM + 1); offset(i) = 12 + i × stride
fn write_ivf_bin(
    n_centroids: usize,
    float_vecs: &[[f32; DIM]],
    raw_labels: &[u8],
    cluster_ids: &[usize],
    path: &str,
) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(n_centroids as u32).to_le_bytes())?;
    w.write_all(&(SLOTS_PER_CLUSTER as u32).to_le_bytes())?;
    w.write_all(&(DIM as u32).to_le_bytes())?;

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_centroids];
    for (i, &ci) in cluster_ids.iter().enumerate() {
        if clusters[ci].len() < SLOTS_PER_CLUSTER {
            clusters[ci].push(i);
        }
    }

    for cluster in &clusters {
        w.write_all(&(cluster.len() as u32).to_le_bytes())?;
        for slot in 0..SLOTS_PER_CLUSTER {
            if slot < cluster.len() {
                let vi = cluster[slot];
                let q: [u8; DIM] = std::array::from_fn(|j| quantize(float_vecs[vi][j]));
                w.write_all(&q)?;
            } else {
                w.write_all(&[0u8; DIM])?;
            }
        }
        for slot in 0..SLOTS_PER_CLUSTER {
            let label = if slot < cluster.len() { raw_labels[cluster[slot]] } else { 0 };
            w.write_all(&[label])?;
        }
    }

    w.flush()
}

// graph.bin: [n:u32][m:u32][n_ep:u32][n_ep×u32] then per node: [m × (neighbor_idx:u32, dist:f32)]
fn write_graph_bin(graph: &[[(u32, f32); M]], entry_points: &[u32], path: &str) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(graph.len() as u32).to_le_bytes())?;
    w.write_all(&(M as u32).to_le_bytes())?;
    w.write_all(&(entry_points.len() as u32).to_le_bytes())?;
    for &ep in entry_points { w.write_all(&ep.to_le_bytes())?; }
    for node in graph {
        for &(nb_idx, edge_dist) in node {
            w.write_all(&nb_idx.to_le_bytes())?;
            w.write_all(&edge_dist.to_le_bytes())?;
        }
    }
    w.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new().num_threads(6).build_global().unwrap();

    let input = File::open("src/dataset/references.json")?;
    let records: Vec<Record> = serde_json::from_reader(std::io::BufReader::new(input))?;
    let n = records.len();
    println!("loaded {n} records");

    let nlist = (n / SLOTS_PER_CLUSTER).max(1);
    println!("nlist={nlist} ({SLOTS_PER_CLUSTER} slots/cluster)");

    std::fs::create_dir_all("src/data")?;

    let float_vecs: Vec<[f32; DIM]> = records.iter().map(|r| r.vector).collect();
    let raw_labels: Vec<u8> = records.iter().map(|r| label_to_u8(&r.label)).collect();

    println!("building centroids (nlist={nlist}, max_iter={MAX_ITER})...");
    let mut rng = SmallRng::seed_from_u64(42);
    let centroids = kmeans(&float_vecs, nlist, MAX_ITER, &mut rng);

    println!("assigning vectors to clusters...");
    let cluster_ids: Vec<usize> = float_vecs.par_iter()
        .map(|v| nearest_idx(v, &centroids))
        .collect();

    let sizes: Vec<usize> = {
        let mut c = vec![0usize; nlist];
        for &ci in &cluster_ids { c[ci] += 1; }
        c
    };
    println!(
        "cluster sizes: min={} max={} avg={:.1}",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap(),
        sizes.iter().sum::<usize>() as f64 / nlist as f64,
    );

    println!("writing centroids.bin...");
    write_centroids_bin(&centroids, "src/data/centroids.bin")?;
    println!("  {:.2} MB", (nlist * DIM * 4) as f64 / 1_048_576.0);

    println!("writing ivf.bin...");
    write_ivf_bin(nlist, &float_vecs, &raw_labels, &cluster_ids, "src/data/ivf.bin")?;
    let stride = 4 + SLOTS_PER_CLUSTER * (DIM + 1);
    println!("  {:.2} MB (stride={stride} bytes/cluster)", (12 + nlist * stride) as f64 / 1_048_576.0);

    println!("building k-NN graph (M={M})...");
    let graph = build_knn_graph(&centroids, &mut rng);

    println!("farthest-point sampling {N_ENTRY_POINTS} entry points...");
    let entry_points = farthest_point_sampling(&centroids, N_ENTRY_POINTS, &mut rng);

    println!("writing graph.bin...");
    write_graph_bin(&graph, &entry_points, "src/data/graph.bin")?;
    println!(
        "  {:.2} MB",
        (12 + N_ENTRY_POINTS * 4 + nlist * M * 8) as f64 / 1_048_576.0,
    );

    println!("done — {n} vectors → {nlist} clusters");
    Ok(())
}
