pub const VECTOR_DIMENSION: usize = 14;

pub struct Partition {
    pub vectors: Vec<[i8; VECTOR_DIMENSION]>,
    pub labels: Vec<u8>,
}

pub fn load_partition(name: &str) -> Option<Partition> {
    let vec_path = format!("src/data/{}.vec", name);
    let lbl_path = format!("src/data/{}.lbl", name);

    let vec_bytes = std::fs::read(&vec_path).ok()?;
    let labels = std::fs::read(&lbl_path).ok()?;

    let vectors = vec_bytes
        .chunks_exact(VECTOR_DIMENSION)
        .map(|chunk| {
            let mut v = [0i8; VECTOR_DIMENSION];
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

pub fn calculate_distance(vec1: &[i8; VECTOR_DIMENSION], vec2: &[i8; VECTOR_DIMENSION]) -> f32 {
    let mut sum = 0.0;
    for i in 0..VECTOR_DIMENSION {
        let diff = (vec1[i] as f32) - (vec2[i] as f32);
        sum += diff * diff;
    }
    sum.sqrt()
}

