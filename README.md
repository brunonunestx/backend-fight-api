# Backend Fight API — Rinha de Backend

Serviço de detecção de fraude construído para a Rinha de Backend, rodando com restrições rígidas de CPU e memória.

---

## O que é

Dado uma transação financeira, o serviço extrai 14 features numéricas, monta um vetor `f32` e classifica como fraude ou não usando KNN (K=5) sobre um dataset de 3 milhões de registros pré-indexados.

```
Transaction → vectorize → [f32; 14] → LSH search → top-5 vizinhos → voto por maioria
```

---

## Stack

- **Rust** — runtime single-binary, zero GC pause
- **Axum** — HTTP server async
- **LSH** (Locality Sensitive Hashing) — busca aproximada de vizinhos (ANN)
- **Nginx** — load balancer
- **Docker Compose** — orquestração local

---

## Como rodar

**Pré-requisitos:** Docker e Docker Compose instalados.

```bash
docker compose up
```

O serviço sobe em `http://localhost:9999`.

**Endpoints:**
```
GET  /ready            # health check
POST /fraud-detection  # predição de fraude (application/json)
```

---

## Arquitetura

```
                    ┌─────────────────┐
                    │   nginx :9999   │  0.10 CPU / 20MB
                    └────────┬────────┘
                   ┌─────────┴─────────┐
                   ▼                   ▼
            ┌──────────┐        ┌──────────┐
            │  api1    │        │  api2    │  0.45 CPU / 165MB cada
            │  :3000   │        │  :3000   │
            └──────────┘        └──────────┘
```

Cada réplica carrega o índice LSH e os vetores quantizados em memória no startup. O nginx faz round-robin entre as duas instâncias.

---

## Como o projeto foi construído

O ponto de partida foi um KNN brute force em Rust: para cada request, percorria todos os 3 milhões de registros, calculava a distância L2 e ordenava. Funciona, mas é O(N log N) por request — inviável sob carga real.

A partir daí, cada otimização foi aplicada com base em profiling e nas restrições do ambiente (1 CPU total, 350MB de RAM para todo o stack).

---

## Otimizações de Performance

### 1. LSH em vez de KNN Brute Force

A implementação original:
```rust
// O(n) — itera todos os 3M vetores
self.vectors.chunks_exact(VECTOR_SIZE)
    .zip(self.labels.iter())
    .map(|(chunk, &label)| (l2_distance(chunk, query), label))
    .collect()
// + sort O(n log n)
```

**LSH** usa `L=7` tabelas de hash com `K=8` funções por tabela. Cada função projeta o vetor em um eixo aleatório e divide em buckets de largura `W`:

```
h(x) = floor((a · x + b) / w)
```

Vetores próximos caem no mesmo bucket com alta probabilidade. Por request: 7 lookups de HashMap em vez de 3M comparações.

**Parâmetros:** `L=7`, `K_HASH=8`, `W=1.0`

---

### 2. Vetores Quantizados u8 (−126MB por réplica)

A primeira versão guardava os vetores f32 dentro do índice serializado:
```
3M × 14 features × 4 bytes = 168MB por réplica
```

Com 2 réplicas + lb dentro de 350MB, inviável.

**Solução:** quantizar os vetores f32 para u8:
```rust
fn quantize(value: f32) -> u8 {
    ((value + 1.0) * 127.5) as u8
}
```

4 bytes → 1 byte por dimensão. O ranqueamento relativo de distâncias é preservado, sem impacto na qualidade do KNN.

Os dados ficam em dois arquivos:

| Arquivo | Conteúdo | Tamanho |
|---|---|---|
| `lsh.bin` | tabelas hash + projeções + labels | ~100MB |
| `vectors.bin` | vetores u8 | ~42MB |

Memória total por réplica: ~157MB — dentro do limite de 165MB.

---

### 3. FxHashMap em vez de std::HashMap

`std::HashMap` usa SipHash (resistente a DoS por design). Para chaves `u64` internas sem input externo, isso é desnecessário.

`FxHashMap` usa hash multiplicativo simples (`x * 0x517CC1B727220A95`) — ~3x mais rápido. Com 7 lookups por request, o ganho se acumula.

Como `FxHashMap` não implementa `serde` nativamente, foi necessário um adapter de serialização:
```rust
#[serde(with = "fx_map_serde")]
tables: Vec<FxHashMap<u64, Vec<u32>>>,
```

---

### 4. `inv_w` — Divisão por Multiplicação

O cálculo do bucket key ocorre `K × L = 56` vezes por request:
```rust
// antes
let bucket = ((dot + offset) / self.w).floor() as i32;

// depois
let bucket = ((dot + offset) * self.inv_w).floor() as i32;
```

Divisão FP custa ~3x mais ciclos que multiplicação. `inv_w = 1.0 / w` é precomputado no build do índice.

---

### 5. Bitmap + Dirty List para Deduplicação

Para evitar processar o mesmo candidato em múltiplas tabelas, a primeira versão usava `HashSet<u32>` — overhead de hashing por inserção e cache misses por acessos espalhados.

**Bitmap:** buffer indexado diretamente pelo ID do ponto, acesso O(1) real sem hashing.

**Dirty list:** em vez de resetar o buffer inteiro (memset de 3MB por request):
```rust
let mut dirty: Vec<u32> = Vec::with_capacity(MAX_CANDIDATES);

seen[id as usize] = 1;
dirty.push(id);

// reset apenas dos ~100 tocados, não dos 3M
for id in dirty {
    seen[id as usize] = 0;
}
```

Reset passa de O(3M) para O(candidatos ≈ 100).

---

### 6. Pool de Buffers `seen`

Compartilhar um único buffer via `Mutex` serializa todos os requests concorrentes — 10 requests simultâneos, 9 bloqueados.

**Pool:** o Mutex é mantido apenas no `pop`/`push` (microssegundos), não durante a busca:
```rust
let mut seen = pool.lock().unwrap().pop()
    .unwrap_or_else(|| vec![0u32; n_points]);

let result = index.search(&mut seen, query, k);  // sem lock

pool.lock().unwrap().push(seen);
```

O pool cresce até o pico de concorrência e para de alocar.

---

### 7. Cap de Candidatos + `select_nth_unstable`

**Cap em 100 candidatos:** para K=5 vizinhos finais, 100 candidatos são mais que suficientes. Evita processar buckets populosos inteiros.

**`select_nth_unstable` em vez de sort:**
```rust
// antes: O(n log n)
candidates.sort_unstable_by_key(|(d, _)| *d);

// depois: O(n) esperado
candidates.select_nth_unstable_by_key(k - 1, |(d, _)| *d);
candidates.truncate(k);
```

Encontra os K menores sem ordenar o restante.

---

### 8. `quantize_vector` com `array::from_fn`

```rust
pub fn quantize_vector(vector: &[f32; 14]) -> [u8; 14] {
    std::array::from_fn(|i| quantize(vector[i]))
}
```

Tamanho fixo em compile time permite ao LLVM vectorizar com SSE2/AVX — processa 4-8 floats por instrução.

---

### 9. `HashSet` para `known_merchants`

Lookup de merchant em `Vec<String>` era O(N). Trocar para `HashSet<String>` torna o `contains` O(1). Serde desserializa JSON array direto em `HashSet` sem mudança no payload.

---

## Resumo do Impacto

| Otimização | Impacto |
|---|---|
| LSH em vez de brute force | O(N log N) → O(log N) por request |
| Vetores u8 separados | −126MB por réplica |
| FxHashMap | ~3x no lookup de hash |
| `inv_w` | divisão FP → multiplicação |
| Bitmap + dirty list | elimina HashSet e memset O(N) |
| Pool de buffers | elimina serialização de requests concorrentes |
| Cap + `select_nth_unstable` | O(N log N) → O(N) no sort final |
| `HashSet` known_merchants | O(N) → O(1) por lookup |

---

## Referências

- [LSH — Andoni & Indyk, 2008](https://people.csail.mit.edu/indyk/p117-andoni.pdf)
- [rustc-hash — FxHashMap](https://github.com/rust-lang/rustc-hash)
- [select_nth_unstable — Rust std](https://doc.rust-lang.org/std/primitive.slice.html#method.select_nth_unstable)
