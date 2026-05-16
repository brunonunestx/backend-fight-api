use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

pub const DIM: usize = 14;
const MAX_CANDIDATES: usize = 100;

#[derive(Serialize, Deserialize)]
pub struct LshIndex {
    inv_w: f32,
    projections: Vec<Vec<[f32; DIM]>>,
    offsets: Vec<Vec<f32>>,
    #[serde(with = "fx_map_serde")]
    tables: Vec<FxHashMap<u64, Vec<u32>>>,
    pub labels: Vec<u8>,
}

impl LshIndex {
    fn bucket_key(&self, table: usize, v: &[f32; DIM]) -> u64 {
        self.projections[table]
            .iter()
            .zip(self.offsets[table].iter())
            .fold(0u64, |acc, (proj, offset)| {
                let dot: f32 = proj.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                let bucket = ((dot + offset) * self.inv_w).floor() as i32;
                acc.wrapping_mul(2654435761).wrapping_add(bucket as u64)
            })
    }

    pub fn search(
        &self,
        query_f32: &[f32; DIM],
        query_u8: &[u8; DIM],
        vectors: &[u8],
        seen: &mut Vec<u32>,
        k: usize,
    ) -> Vec<(u32, u8)> {
        let mut dirty: Vec<u32> = Vec::with_capacity(MAX_CANDIDATES);
        let mut candidates: Vec<(u32, u8)> = Vec::with_capacity(MAX_CANDIDATES);

        'outer: for t in 0..self.tables.len() {
            let key = self.bucket_key(t, query_f32);
            if let Some(ids) = self.tables[t].get(&key) {
                for &id in ids {
                    let slot = &mut seen[id as usize];
                    if *slot == 0 {
                        *slot = 1;
                        dirty.push(id);
                        let start = id as usize * DIM;
                        let dist = l2_sq_u8(query_u8, &vectors[start..start + DIM]);
                        candidates.push((dist, self.labels[id as usize]));
                        if candidates.len() == MAX_CANDIDATES {
                            break 'outer;
                        }
                    }
                }
            }
        }

        for id in dirty {
            seen[id as usize] = 0;
        }

        if candidates.len() > k {
            candidates.select_nth_unstable_by_key(k - 1, |(d, _)| *d);
            candidates.truncate(k);
        }

        candidates
    }
}

fn l2_sq_u8(a: &[u8; DIM], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = x as i32 - y as i32;
        (d * d) as u32
    }).sum()
}

mod fx_map_serde {
    use rustc_hash::FxHashMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S>(maps: &Vec<FxHashMap<u64, Vec<u32>>>, s: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let std_maps: Vec<HashMap<u64, Vec<u32>>> = maps
            .iter()
            .map(|m| m.iter().map(|(&k, v)| (k, v.clone())).collect())
            .collect();
        std_maps.serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Vec<FxHashMap<u64, Vec<u32>>>, D::Error>
    where D: Deserializer<'de> {
        let std_maps: Vec<HashMap<u64, Vec<u32>>> = Vec::deserialize(d)?;
        Ok(std_maps.into_iter()
            .map(|m| m.into_iter().collect())
            .collect())
    }
}
