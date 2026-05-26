pub const VECTOR_DIMENSION: usize = 14;
pub const VECTOR_STRIDE: usize = 16; // padded to 16 for safe 128-bit SIMD loads

pub struct Partition {
    pub vectors: Vec<[i8; VECTOR_STRIDE]>,
    pub labels: Vec<u8>,
}

pub struct IvfIndex {
    pub centroids: Vec<[i8; VECTOR_STRIDE]>,
    pub lists: Vec<Vec<u32>>,
}

impl IvfIndex {
    pub fn load(name: &str) -> Option<Self> {
        let path = format!("src/data/{}.ivf", name);
        let bytes = std::fs::read(&path).ok()?;
        let mut cur = 0;

        let num_centroids = u16::from_le_bytes([bytes[cur], bytes[cur + 1]]) as usize;
        cur += 2;

        let mut centroids = Vec::with_capacity(num_centroids);
        for _ in 0..num_centroids {
            let mut c = [0i8; VECTOR_STRIDE];
            for d in 0..VECTOR_STRIDE {
                c[d] = bytes[cur + d] as i8;
            }
            centroids.push(c);
            cur += VECTOR_STRIDE;
        }

        let mut list_lengths = Vec::with_capacity(num_centroids);
        for _ in 0..num_centroids {
            let len = u32::from_le_bytes([bytes[cur], bytes[cur+1], bytes[cur+2], bytes[cur+3]]) as usize;
            list_lengths.push(len);
            cur += 4;
        }

        let mut lists = Vec::with_capacity(num_centroids);
        for &len in &list_lengths {
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                let idx = u32::from_le_bytes([bytes[cur], bytes[cur+1], bytes[cur+2], bytes[cur+3]]);
                list.push(idx);
                cur += 4;
            }
            lists.push(list);
        }

        Some(IvfIndex { centroids, lists })
    }
}

pub fn load_partition(name: &str) -> Option<Partition> {
    let vec_path = format!("src/data/{}.vec", name);
    let lbl_path = format!("src/data/{}.lbl", name);

    let vec_bytes = std::fs::read(&vec_path).ok()?;
    let labels = std::fs::read(&lbl_path).ok()?;

    let vectors = vec_bytes
        .chunks_exact(VECTOR_DIMENSION)
        .map(|chunk| {
            let mut v = [0i8; VECTOR_STRIDE];
            for (i, &b) in chunk.iter().enumerate() {
                v[i] = b as i8;
            }
            v
        })
        .collect();

    Some(Partition { vectors, labels })
}

pub fn normalize(vector: &mut [f32; VECTOR_DIMENSION]) -> [i8; VECTOR_DIMENSION] {
    let mut result = [0i8; VECTOR_DIMENSION];
    for i in 0..VECTOR_DIMENSION {
        result[i] = quantize(vector[i]);
    }
    result
}

pub fn quantize(value: f32) -> i8 {
    (value * 127.0).round() as i8
}

pub fn load_vectors_from_file(file_path: &str) -> Vec<[i8; VECTOR_DIMENSION]> {
    let file_content = std::fs::read(file_path).expect("Failed to read vector file");
    file_content
        .chunks_exact(VECTOR_DIMENSION)
        .map(|chunk| {
            let mut v = [0i8; VECTOR_DIMENSION];
            for (i, &b) in chunk.iter().enumerate() {
                v[i] = b as i8;
            }
            v
        })
        .collect()
}

// Kept for scripts (define_near_ranges).
pub fn calculate_distance(vec1: &[i8; VECTOR_DIMENSION], vec2: &[i8; VECTOR_DIMENSION]) -> f32 {
    let mut sum = 0i32;
    for i in 0..VECTOR_DIMENSION {
        let diff = (vec1[i] as i32) - (vec2[i] as i32);
        sum += diff * diff;
    }
    (sum as f32).sqrt()
}

/// IVF query: find nearest `nprobe` clusters via AVX2, then scan only those
/// vectors with AVX2 2-at-a-time. Falls back to SSE4.1/scalar when unavailable.
pub fn ivf_knn(
    index: &IvfIndex,
    partition: &Partition,
    query: &[i8; VECTOR_STRIDE],
    threshold_sq: i32,
    nprobe: usize,
) -> ([usize; 5], usize) {
    let probed = nearest_centroid_indices(&index.centroids, query, nprobe);

    let mut offsets = [0usize; 5];
    let mut count = 0;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            for ci in &probed {
                if unsafe {
                    scan_list_avx2(&partition.vectors, &index.lists[*ci], query, threshold_sq, &mut offsets, &mut count)
                } {
                    break;
                }
            }
            return (offsets, count);
        }
    }

    // SSE4.1 / scalar fallback
    'outer: for ci in &probed {
        for &vec_idx in &index.lists[*ci] {
            let vec = &partition.vectors[vec_idx as usize];
            if l2_sq_dispatch(query, vec) <= threshold_sq {
                offsets[count] = vec_idx as usize;
                count += 1;
                if count >= 5 {
                    break 'outer;
                }
            }
        }
    }

    (offsets, count)
}

/// Brute-force scan returning up to 5 vector indices whose squared L2
/// distance to `query` is ≤ `threshold_sq`. Dispatches to SIMD when
/// available; falls back to scalar otherwise.
///
/// `query` must be a 14-dim vector zero-padded to 16 bytes.
pub fn brute_force_knn(
    partition: &Partition,
    query: &[i8; VECTOR_STRIDE],
    threshold_sq: i32,
) -> ([usize; 5], usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { brute_force_avx2(partition, query, threshold_sq) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { brute_force_sse41(partition, query, threshold_sq) };
        }
    }
    brute_force_scalar(partition, query, threshold_sq)
}

// ── centroid search ───────────────────────────────────────────────────────────

fn nearest_centroid_indices(
    centroids: &[[i8; VECTOR_STRIDE]],
    query: &[i8; VECTOR_STRIDE],
    nprobe: usize,
) -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { nearest_centroid_indices_avx2(centroids, query, nprobe) };
        }
    }
    nearest_centroid_indices_scalar(centroids, query, nprobe)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nearest_centroid_indices_avx2(
    centroids: &[[i8; VECTOR_STRIDE]],
    query: &[i8; VECTOR_STRIDE],
    nprobe: usize,
) -> Vec<usize> {
    let q = query.as_ptr();
    let n = centroids.len();
    let mut dists: Vec<(i32, usize)> = Vec::with_capacity(n);
    let mut i = 0;

    while i + 1 < n {
        let (d0, d1) = unsafe {
            l2_sq_x2_avx2(q, centroids[i].as_ptr(), centroids[i + 1].as_ptr())
        };
        dists.push((d0, i));
        dists.push((d1, i + 1));
        i += 2;
    }

    if i < n {
        dists.push((unsafe { l2_sq_sse41(q, centroids[i].as_ptr()) }, i));
    }

    dists.sort_unstable_by_key(|&(d, _)| d);
    dists.iter().take(nprobe).map(|&(_, i)| i).collect()
}

fn nearest_centroid_indices_scalar(
    centroids: &[[i8; VECTOR_STRIDE]],
    query: &[i8; VECTOR_STRIDE],
    nprobe: usize,
) -> Vec<usize> {
    let mut dists: Vec<(i32, usize)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (l2_sq_dispatch(query, c), i))
        .collect();
    dists.sort_unstable_by_key(|&(d, _)| d);
    dists.iter().take(nprobe).map(|&(_, i)| i).collect()
}

// ── IVF cluster scan (AVX2: 2 vectors per iteration) ─────────────────────────

/// Scans `list` indices 2-at-a-time using AVX2. Returns true when `offsets` is full.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_list_avx2(
    vectors: &[[i8; VECTOR_STRIDE]],
    list: &[u32],
    query: &[i8; VECTOR_STRIDE],
    threshold_sq: i32,
    offsets: &mut [usize; 5],
    count: &mut usize,
) -> bool {
    let q = query.as_ptr();
    let n = list.len();
    let mut i = 0;

    while i + 1 < n && *count < 5 {
        let idx0 = list[i]     as usize;
        let idx1 = list[i + 1] as usize;
        let (d0, d1) = unsafe {
            l2_sq_x2_avx2(q, vectors[idx0].as_ptr(), vectors[idx1].as_ptr())
        };
        if d0 <= threshold_sq {
            offsets[*count] = idx0;
            *count += 1;
        }
        if *count < 5 && d1 <= threshold_sq {
            offsets[*count] = idx1;
            *count += 1;
        }
        i += 2;
    }

    // tail: at most 1 remaining
    if i < n && *count < 5 {
        let idx = list[i] as usize;
        if unsafe { l2_sq_sse41(q, vectors[idx].as_ptr()) } <= threshold_sq {
            offsets[*count] = idx;
            *count += 1;
        }
    }

    *count >= 5
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn l2_sq_dispatch(a: &[i8; VECTOR_STRIDE], b: &[i8; VECTOR_STRIDE]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { l2_sq_sse41(a.as_ptr(), b.as_ptr()) };
        }
    }
    l2_sq_scalar(a, b)
}

fn l2_sq_scalar(a: &[i8; VECTOR_STRIDE], b: &[i8; VECTOR_STRIDE]) -> i32 {
    let mut sum = 0i32;
    for d in 0..VECTOR_DIMENSION {
        let diff = (a[d] as i32) - (b[d] as i32);
        sum += diff * diff;
    }
    sum
}

// ── scalar fallback ───────────────────────────────────────────────────────────

fn brute_force_scalar(
    partition: &Partition,
    query: &[i8; VECTOR_STRIDE],
    threshold_sq: i32,
) -> ([usize; 5], usize) {
    let mut offsets = [0usize; 5];
    let mut count = 0;
    for (i, vec) in partition.vectors.iter().enumerate() {
        if l2_sq_scalar(query, vec) <= threshold_sq {
            offsets[count] = i;
            count += 1;
            if count >= 5 {
                break;
            }
        }
    }
    (offsets, count)
}

// ── SSE4.1 ───────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn l2_sq_sse41(q: *const i8, d: *const i8) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let vq = _mm_loadu_si128(q as *const __m128i);
        let vd = _mm_loadu_si128(d as *const __m128i);

        let diff_lo = _mm_sub_epi16(_mm_cvtepi8_epi16(vq), _mm_cvtepi8_epi16(vd));
        let diff_hi = _mm_sub_epi16(
            _mm_cvtepi8_epi16(_mm_srli_si128(vq, 8)),
            _mm_cvtepi8_epi16(_mm_srli_si128(vd, 8)),
        );

        let sum = _mm_add_epi32(
            _mm_madd_epi16(diff_lo, diff_lo),
            _mm_madd_epi16(diff_hi, diff_hi),
        );

        let shuf = _mm_shuffle_epi32(sum, 0b_10_11_00_01);
        let s2 = _mm_add_epi32(sum, shuf);
        let shuf2 = _mm_shuffle_epi32(s2, 0b_01_00_11_10);
        _mm_cvtsi128_si32(_mm_add_epi32(s2, shuf2))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn brute_force_sse41(
    partition: &Partition,
    query: &[i8; VECTOR_STRIDE],
    threshold_sq: i32,
) -> ([usize; 5], usize) {
    let mut offsets = [0usize; 5];
    let mut count = 0;
    let q = query.as_ptr();
    for (i, vec) in partition.vectors.iter().enumerate() {
        if unsafe { l2_sq_sse41(q, vec.as_ptr()) } <= threshold_sq {
            offsets[count] = i;
            count += 1;
            if count >= 5 {
                break;
            }
        }
    }
    (offsets, count)
}

// ── AVX2 (2 vectors per iteration) ───────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_sq_x2_avx2(q: *const i8, d0: *const i8, d1: *const i8) -> (i32, i32) {
    use std::arch::x86_64::*;
    unsafe {
        let vq  = _mm_loadu_si128(q  as *const __m128i);
        let vd0 = _mm_loadu_si128(d0 as *const __m128i);
        let vd1 = _mm_loadu_si128(d1 as *const __m128i);
        let qhi = _mm_srli_si128(vq, 8);

        let diff0_lo = _mm_sub_epi16(_mm_cvtepi8_epi16(vq),  _mm_cvtepi8_epi16(vd0));
        let diff0_hi = _mm_sub_epi16(_mm_cvtepi8_epi16(qhi), _mm_cvtepi8_epi16(_mm_srli_si128(vd0, 8)));
        let diff1_lo = _mm_sub_epi16(_mm_cvtepi8_epi16(vq),  _mm_cvtepi8_epi16(vd1));
        let diff1_hi = _mm_sub_epi16(_mm_cvtepi8_epi16(qhi), _mm_cvtepi8_epi16(_mm_srli_si128(vd1, 8)));

        let s0 = _mm_add_epi32(_mm_madd_epi16(diff0_lo, diff0_lo), _mm_madd_epi16(diff0_hi, diff0_hi));
        let s1 = _mm_add_epi32(_mm_madd_epi16(diff1_lo, diff1_lo), _mm_madd_epi16(diff1_hi, diff1_hi));

        let s256 = _mm256_set_m128i(s1, s0);
        let h1 = _mm256_hadd_epi32(s256, s256);
        let h2 = _mm256_hadd_epi32(h1, h1);

        (
            _mm_cvtsi128_si32(_mm256_castsi256_si128(h2)),
            _mm_cvtsi128_si32(_mm256_extracti128_si256(h2, 1)),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn brute_force_avx2(
    partition: &Partition,
    query: &[i8; VECTOR_STRIDE],
    threshold_sq: i32,
) -> ([usize; 5], usize) {
    let mut offsets = [0usize; 5];
    let mut count = 0;
    let q = query.as_ptr();
    let vecs = &partition.vectors;
    let n = vecs.len();
    let mut i = 0;

    while i + 1 < n && count < 5 {
        let (d0, d1) = unsafe { l2_sq_x2_avx2(q, vecs[i].as_ptr(), vecs[i + 1].as_ptr()) };
        if d0 <= threshold_sq {
            offsets[count] = i;
            count += 1;
        }
        if count < 5 && d1 <= threshold_sq {
            offsets[count] = i + 1;
            count += 1;
        }
        i += 2;
    }

    if i < n && count < 5 && unsafe { l2_sq_sse41(q, vecs[i].as_ptr()) } <= threshold_sq {
        offsets[count] = i;
        count += 1;
    }

    (offsets, count)
}
