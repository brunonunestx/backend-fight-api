use std::cell::RefCell;
use std::collections::BinaryHeap;
use serde::{Deserialize, Serialize};

pub const DIM: usize = 14;

#[derive(Serialize, Deserialize)]
pub struct IvfIndex {
    pub coarse_centroids: Vec<[f32; DIM]>,
    pub coarse_to_fine: Vec<Vec<u32>>,
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
        nprobe_coarse: usize,
    ) -> usize {
        let nprobe_coarse = nprobe_coarse.min(self.coarse_centroids.len());
        let nprobe = nprobe.min(self.centroids.len());

        thread_local! {
            static COARSE_BUF: RefCell<Vec<(usize, f32)>> = RefCell::new(Vec::new());
            static FINE_BUF: RefCell<Vec<(usize, f32)>> = RefCell::new(Vec::new());
        }

        COARSE_BUF.with(|coarse_cell| {
            FINE_BUF.with(|fine_cell| {
                let mut coarse_dists = coarse_cell.borrow_mut();
                coarse_dists.clear();
                coarse_dists.extend(
                    self.coarse_centroids.iter().enumerate().map(|(i, c)| (i, l2_sq_f32(query_f32, c)))
                );

                if nprobe_coarse < coarse_dists.len() {
                    coarse_dists.select_nth_unstable_by(nprobe_coarse - 1, |a, b| {
                        a.1.partial_cmp(&b.1).unwrap()
                    });
                    coarse_dists.truncate(nprobe_coarse);
                }

                let mut fine_dists = fine_cell.borrow_mut();
                fine_dists.clear();
                for (ci, _) in coarse_dists.iter() {
                    for &fi in &self.coarse_to_fine[*ci] {
                        let fi = fi as usize;
                        fine_dists.push((fi, l2_sq_f32(query_f32, &self.centroids[fi])));
                    }
                }

                if nprobe < fine_dists.len() {
                    fine_dists.select_nth_unstable_by(nprobe - 1, |a, b| {
                        a.1.partial_cmp(&b.1).unwrap()
                    });
                    fine_dists.truncate(nprobe);
                }

                let mut heap: BinaryHeap<(u32, u8)> = BinaryHeap::with_capacity(k + 1);
                let mut fraud_in_heap: usize = 0;

                for (fi, _) in fine_dists.iter() {
                    for &id in &self.lists[*fi] {
                        let start = id as usize * DIM;
                        let dist = l2_sq_u8(query_u8, &vectors[start..start + DIM]);
                        let label = self.labels[id as usize];

                        if heap.len() < k {
                            heap.push((dist, label));
                            if label == 1 { fraud_in_heap += 1; }
                        } else if let Some(&(max_dist, _)) = heap.peek() {
                            if dist < max_dist {
                                let (_, evicted) = heap.pop().unwrap();
                                if evicted == 1 { fraud_in_heap -= 1; }
                                heap.push((dist, label));
                                if label == 1 { fraud_in_heap += 1; }
                            }
                        }
                    }
                }

                fraud_in_heap
            })
        })
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

#[inline(always)]
fn l2_sq_u8(a: &[u8; DIM], b: &[u8]) -> u32 {
    let mut sum = 0u32;
    for i in 0..DIM {
        let d = a[i] as i32 - b[i] as i32;
        sum += (d * d) as u32;
    }
    sum
}
