use serde::{Deserialize, Serialize};

pub const DIM: usize = 14;

#[derive(Serialize, Deserialize)]
pub struct IvfIndex {
    pub centroids: Vec<[f32; DIM]>,
    pub lists: Vec<Vec<u32>>,
    pub labels: Vec<u8>,
}

impl IvfIndex {
    pub fn search(
        &self,
        query_f32: &[f32; DIM],
        query_u8: &[u8; DIM],
        vectors: &[u8],
        k: usize,
        nprobe: usize,
    ) -> Vec<(u32, u8)> {
        let nprobe = nprobe.min(self.centroids.len());

        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq_f32(query_f32, c)))
            .collect();

        if nprobe < centroid_dists.len() {
            centroid_dists.select_nth_unstable_by(nprobe - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap()
            });
            centroid_dists.truncate(nprobe);
        }

        let mut candidates: Vec<(u32, u8)> = Vec::new();
        for (ci, _) in &centroid_dists {
            for &id in &self.lists[*ci] {
                let start = id as usize * DIM;
                let dist = l2_sq_u8(query_u8, &vectors[start..start + DIM]);
                candidates.push((dist, self.labels[id as usize]));
            }
        }

        if candidates.len() > k {
            candidates.select_nth_unstable_by_key(k - 1, |&(d, _)| d);
            candidates.truncate(k);
        }

        candidates
    }
}

fn l2_sq_f32(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn l2_sq_u8(a: &[u8; DIM], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as i32 - y as i32;
            (d * d) as u32
        })
        .sum()
}
