use std::{
    cell::RefCell,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

pub const DIM: usize = 14;
const M: usize = 2;
const SLOTS_PER_CLUSTER: usize = 10;
// stride per cluster in ivf.bin: count:u32 + slots×dim×u8 + slots×u8
const IVF_STRIDE: u64 = (4 + SLOTS_PER_CLUSTER * (DIM + 1)) as u64;
const IVF_HEADER: u64 = 12; // n_centroids:u32 + slots:u32 + dim:u32

const IVF_PATH: &str = "src/data/ivf.bin";

pub struct GraphIndex {
    centroids: Vec<[f32; DIM]>,
    // per node: M × (neighbor_idx, real-L2-distance)
    graph: Vec<[(u32, f32); M]>,
    entry_points: Vec<u32>,
}

impl GraphIndex {
    pub fn load(centroids_path: &str, graph_path: &str) -> Self {
        let centroids = load_centroids(centroids_path);
        let (graph, entry_points) = load_graph(graph_path);
        println!(
            "GraphIndex: {} centroids, {} entry points",
            centroids.len(),
            entry_points.len()
        );
        GraphIndex { centroids, graph, entry_points }
    }

    pub fn search(&self, query_f32: &[f32; DIM], query_u8: &[u8; DIM], k: usize, nprobe: usize) -> usize {
        // 1. pick closest entry point
        let (mut current, cur_dist_sq) = self.entry_points.iter()
            .map(|&ep| (ep as usize, l2_sq(query_f32, &self.centroids[ep as usize])))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let mut cur_dist = cur_dist_sq.sqrt();

        // 2. greedy walk — collect all evaluated nodes for nprobe selection
        //    invariant: cur_dist = l2(q, centroids[current])
        let mut candidates: Vec<(usize, f32)> = vec![(current, cur_dist)];
        loop {
            let mut improved = false;
            for &(nb_idx, edge_dist) in &self.graph[current] {
                if nb_idx == u32::MAX { continue; }
                // lower bound: dist(q, v) >= |dist(q, u) - dist(u, v)|
                let lower = (cur_dist - edge_dist).abs();
                if lower >= cur_dist { continue; }

                let d = l2(query_f32, &self.centroids[nb_idx as usize]);
                candidates.push((nb_idx as usize, d));
                if d < cur_dist {
                    current = nb_idx as usize;
                    cur_dist = d;
                    improved = true;
                }
            }
            if !improved { break; }
        }

        // 3. pick top-nprobe clusters by distance
        candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.dedup_by_key(|c| c.0);
        let n_probe = nprobe.min(candidates.len());

        // 4. read and aggregate vectors from top-nprobe clusters
        thread_local! {
            static IVF_FILE: RefCell<File> = RefCell::new(
                File::open(IVF_PATH).expect("failed to open ivf.bin")
            );
        }

        IVF_FILE.with(|cell| {
            let mut f = cell.borrow_mut();
            let mut all_vecs: Vec<u8> = Vec::with_capacity(n_probe * SLOTS_PER_CLUSTER * DIM);
            let mut all_labels: Vec<u8> = Vec::with_capacity(n_probe * SLOTS_PER_CLUSTER);
            let mut total = 0usize;

            let mut vecs_buf = [0u8; SLOTS_PER_CLUSTER * DIM];
            let mut labels_buf = [0u8; SLOTS_PER_CLUSTER];

            for &(cluster_idx, _) in &candidates[..n_probe] {
                let offset = IVF_HEADER + cluster_idx as u64 * IVF_STRIDE;
                f.seek(SeekFrom::Start(offset)).unwrap();

                let mut count_buf = [0u8; 4];
                f.read_exact(&mut count_buf).unwrap();
                let count = (u32::from_le_bytes(count_buf) as usize).min(SLOTS_PER_CLUSTER);

                f.read_exact(&mut vecs_buf).unwrap();
                f.read_exact(&mut labels_buf).unwrap();

                all_vecs.extend_from_slice(&vecs_buf[..count * DIM]);
                all_labels.extend_from_slice(&labels_buf[..count]);
                total += count;
            }

            // 5. KNN on combined candidates
            knn_count(query_u8, &all_vecs, &all_labels, total, k)
        })
    }
}

// Returns number of fraud labels in the K nearest vectors.
// Uses a max-heap emulated with a fixed array (k is always small).
fn knn_count(query: &[u8; DIM], vecs: &[u8], labels: &[u8], count: usize, k: usize) -> usize {
    let mut heap: [(u32, u8); 10] = [(u32::MAX, 0); 10]; // k ≤ 10
    let k = k.min(count).min(10);
    let mut filled = 0usize;
    let mut max_pos = 0usize;
    let mut fraud = 0usize;

    for slot in 0..count {
        let d = l2_sq_u8(query, &vecs[slot * DIM..(slot + 1) * DIM]);
        let label = labels[slot];

        if filled < k {
            heap[filled] = (d, label);
            if label == 1 { fraud += 1; }
            filled += 1;
            if filled == k {
                max_pos = (0..k).max_by_key(|&i| heap[i].0).unwrap();
            }
        } else if d < heap[max_pos].0 {
            if heap[max_pos].1 == 1 { fraud -= 1; }
            heap[max_pos] = (d, label);
            if label == 1 { fraud += 1; }
            max_pos = (0..k).max_by_key(|&i| heap[i].0).unwrap();
        }
    }

    fraud
}

fn load_centroids(path: &str) -> Vec<[f32; DIM]> {
    let bytes = std::fs::read(path).expect("failed to read centroids.bin");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(dim, DIM, "centroids.bin: expected dim={DIM}, got {dim}");

    (0..n)
        .map(|i| {
            let base = 8 + i * DIM * 4;
            std::array::from_fn(|j| {
                f32::from_le_bytes(bytes[base + j * 4..base + j * 4 + 4].try_into().unwrap())
            })
        })
        .collect()
}

fn load_graph(path: &str) -> (Vec<[(u32, f32); M]>, Vec<u32>) {
    let bytes = std::fs::read(path).expect("failed to read graph.bin");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let m_stored = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(m_stored, M, "graph.bin: expected M={M}, got {m_stored}");
    let n_ep = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let ep_start = 12;
    let entry_points: Vec<u32> = (0..n_ep)
        .map(|i| u32::from_le_bytes(bytes[ep_start + i * 4..ep_start + i * 4 + 4].try_into().unwrap()))
        .collect();

    let graph_start = ep_start + n_ep * 4;
    let graph: Vec<[(u32, f32); M]> = (0..n)
        .map(|i| {
            std::array::from_fn(|slot| {
                let base = graph_start + (i * M + slot) * 8;
                let idx = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
                let dist = f32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
                (idx, dist)
            })
        })
        .collect();

    (graph, entry_points)
}

fn l2_sq(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| { let d = x - y; d * d }).sum()
}

fn l2(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    l2_sq(a, b).sqrt()
}

fn l2_sq_u8(a: &[u8; DIM], b: &[u8]) -> u32 {
    let mut sum = 0u32;
    for i in 0..DIM {
        let d = a[i] as i32 - b[i] as i32;
        sum += (d * d) as u32;
    }
    sum
}
