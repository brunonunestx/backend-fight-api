# Otimizações de Performance — Fraud Detection Service

Documentação de todas as otimizações aplicadas ao serviço de detecção de fraude, com justificativa técnica e impacto de cada decisão.

---

## Contexto

O serviço recebe uma transação, extrai 14 features numéricas (vetor f32), e classifica como fraude ou não usando KNN (K=5 vizinhos mais próximos) sobre um dataset de 3 milhões de registros.

Pipeline por request:
```
Transaction → vectorize → [f32; 14] → LSH search → top-5 vizinhos → voto por maioria
```

---

## 1. Substituição do KNN Brute Force por LSH

### Problema
A implementação original fazia varredura linear sobre todos os N registros:
```rust
// O(n) — itera todos os 3M vetores
self.vectors.chunks_exact(VECTOR_SIZE)
    .zip(self.labels.iter())
    .map(|(chunk, &label)| (l2_distance(chunk, query), label))
    .collect()
// + sort O(n log n)
```

Para 3M registros isso é inviável em produção.

### Solução: Locality Sensitive Hashing (LSH)

LSH é um método de busca aproximada de vizinhos (ANN) baseado em hashing que preserva localidade — vetores próximos no espaço tendem a colidir no mesmo bucket.

**Como funciona:**

Para cada uma das `L` tabelas, gera `K` funções de hash da forma:
```
h(x) = floor((a · x + b) / w)
```
Onde:
- `a` é um vetor aleatório de distribuição Normal(0,1) com mesma dimensão do dado
- `b` é um offset aleatório de Uniform(0, w)
- `w` é a largura do bucket

Vetores próximos têm produto interno `a·x` próximo, então `floor(...)` cai na mesma fatia. Os `K` inteiros resultantes são combinados em uma única chave `u64` via fold multiplicativo.

**Por que funciona:**

Se `x` e `y` são próximos, a projeção `a·x ≈ a·y`, então caem no mesmo bucket com alta probabilidade. Com `L` tabelas e projeções diferentes, a chance de ao menos uma delas colidir para vizinhos reais é alta.

**Parâmetros finais:**
```
L = 7       → número de tabelas (mais tabelas = melhor recall)
K_HASH = 8  → funções de hash por tabela (mais K = buckets menores = menos candidatos)
W = 1.0     → largura do bucket
```

**Complexidade:** O(L) lookups de HashMap por query, versus O(N) do brute force.

---

## 2. Separação de Vetores e Índice (Gestão de Memória)

### Problema
A primeira versão do `LshIndex` armazenava os vetores f32 originais dentro do próprio índice serializado. Para 3M registros isso gerava:
```
vetores f32: 3M × 14 × 4 bytes = 168MB por réplica
```

Com 2 réplicas + loadbalancer dentro de um orçamento de 350MB total, isso era inviável.

### Solução

Separar os dados em dois artefatos com tipos distintos:

| Arquivo | Conteúdo | Tamanho (3M registros) |
|---|---|---|
| `lsh.bin` | tabelas hash + labels + projeções | ~100MB |
| `vectors.bin` | vetores quantizados u8 | ~42MB |

**Vetores quantizados para u8:**

Os vetores f32 normalizados em [-1, 1] são quantizados para u8:
```rust
fn quantize(value: f32) -> u8 {
    ((value + 1.0) * 127.5) as u8
}
```

Isso reduz 4x o uso de memória (4 bytes → 1 byte por dimensão) sem impacto relevante na qualidade do KNN, pois o ranqueamento relativo de distâncias é preservado.

**Memória final por réplica:**
```
lsh.bin:      ~100MB
vectors.bin:   ~42MB
runtime:       ~15MB
─────────────
total:        ~157MB × 2 réplicas + ~20MB LB ≈ 334MB ✓
```

---

## 3. FxHashMap em vez de std::HashMap

### Problema
`std::HashMap` usa SipHash por padrão — um hash criptograficamente seguro projetado para resistir ataques DoS. Para chaves `u64` internas (sem input externo), essa segurança é desnecessária e cara.

### Solução
`rustc_hash::FxHashMap` usa um hash multiplicativo simples:
```
hash(x) = x * 0x517CC1B727220A95
```

Para chaves u64 é ~3x mais rápido que SipHash. Como fazemos `L=7` lookups de HashMap por request, esse ganho se multiplica.

`FxHashMap` não implementa `serde` nativamente, então foi necessário um adapter de serialização que converte para/de `std::HashMap` apenas no momento de salvar/carregar o índice:
```rust
#[serde(with = "fx_map_serde")]
tables: Vec<FxHashMap<u64, Vec<u32>>>,
```

---

## 4. Precomputação de `inv_w` (Divisão → Multiplicação)

### Problema
No cálculo do `bucket_key`, a divisão por `w` ocorre para cada uma das `K × L` projeções:
```rust
let bucket = ((dot + offset) / self.w).floor() as i32;
```

Divisão FP custa ~3x mais ciclos de CPU que multiplicação.

### Solução
Armazenar o inverso de `w` no índice e multiplicar:
```rust
// build
let index = LshIndex { inv_w: 1.0 / W, ... };

// search
let bucket = ((dot + offset) * self.inv_w).floor() as i32;
```

Mesma semântica matemática, sem divisão em runtime.

---

## 5. Bitmap para Deduplicação de Candidatos

### Problema
A primeira versão usava `HashSet<u32>` para evitar processar o mesmo candidato em múltiplas tabelas:
```rust
let mut seen = HashSet::new();
if seen.insert(id) { /* processa */ }
```

`HashSet` tem overhead por inserção: computa hash do valor, resolve colisões, acessa memória em endereços potencialmente não contíguos (cache miss).

### Solução
Buffer booleano indexado diretamente pelo ID do ponto:
```rust
let mut seen = vec![0u32; self.labels.len()];
if seen[id as usize] == 0 {
    seen[id as usize] = 1;
    /* processa */
}
```

Acesso O(1) real sem hashing — vai direto no índice do array. Para IDs sequenciais, o prefetch do CPU funciona melhor do que acessos espalhados do HashMap.

---

## 6. Dirty List para Reset do Buffer

### Problema
Após cada request, o buffer de 3M posições precisava ser resetado. A abordagem ingênua:
```rust
let mut seen = vec![false; 3_000_000]; // aloca 3MB + memset de 3MB
```
Isso representava 3MB de alocação e 3MB de zeragem a cada request.

### Solução
Rastrear apenas os índices modificados e resetar somente eles:
```rust
let mut dirty: Vec<u32> = Vec::with_capacity(MAX_CANDIDATES);

// ao marcar como visto:
seen[id as usize] = 1;
dirty.push(id);

// ao final do search, em vez de limpar tudo:
for id in dirty {
    seen[id as usize] = 0; // reseta apenas os ~100 tocados
}
```

O reset passa de O(N=3M) para O(candidatos=100).

---

## 7. Pool de Buffers `seen` (Eliminação de Contention)

### Problema
Compartilhar um único buffer `seen` via `Mutex<Vec<u32>>` serializa todos os requests:
```rust
// 10 requests simultâneos → 9 ficam bloqueados esperando
let mut seen = self.seen.lock().unwrap();
let result = self.index.search(..., &mut seen, K); // lock mantido durante toda busca
```

### Solução
Pool de buffers — o Mutex é mantido apenas durante `pop`/`push` (microssegundos), não durante a busca:
```rust
// pega do pool (lock brevíssimo)
let mut seen = self.seen_pool.lock().unwrap()
    .pop()
    .unwrap_or_else(|| vec![0u32; self.n_points]); // aloca se pool vazio

// busca sem nenhum lock
let neighbours = self.index.search(..., &mut seen, K);

// devolve ao pool (lock brevíssimo)
self.seen_pool.lock().unwrap().push(seen);
```

O pool cresce organicamente até o pico de concorrência e para de alocar. Requests concorrentes não se bloqueiam.

---

## 8. Cap de Candidatos + `select_nth_unstable`

### Problema
Sem limite, buckets populosos podiam gerar centenas de candidatos por request, todos passando pelo `l2_sq_u8`. O sort final era O(n log n) sobre todos eles.

### Solução

**Cap de candidatos:**
```rust
const MAX_CANDIDATES: usize = 100;

if candidates.len() == MAX_CANDIDATES {
    break 'outer; // para de coletar, ignora tabelas restantes
}
```

Para K=5, processar 100 candidatos é 20x mais do que o necessário. Com K=8 hash functions, os verdadeiros vizinhos aparecem nas primeiras tabelas com alta probabilidade.

**`select_nth_unstable` em vez de sort:**
```rust
// antes: O(n log n)
candidates.sort_unstable_by_key(|(d, _)| *d);

// depois: O(n) esperado
candidates.select_nth_unstable_by_key(k - 1, |(d, _)| *d);
candidates.truncate(k);
```

`select_nth_unstable` usa introselect para particionar o array em O(n) — encontra os K menores sem ordenar o restante.

---

## 9. `quantize` Simplificada

### Antes
```rust
fn quantize(value: f32) -> u8 {
    (((value + 1.0) / 2.0) * 255.0).clamp(0.0, 255.0) as u8
}
```

### Depois
```rust
pub fn quantize(value: f32) -> u8 {
    ((value + 1.0) * 127.5) as u8
}
```

A divisão por 2 foi incorporada ao multiplicador (255/2 = 127.5), eliminando uma operação FP. O `.clamp()` é redundante porque:
- Todos os valores saem de `vectorize` no range [-1.0, 1.0]
- `(-1.0 + 1.0) * 127.5 = 0.0` → `0u8`
- `(1.0 + 1.0) * 127.5 = 255.0` → `255u8`
- O cast `as u8` em Rust já satura para valores fora do range

---

## 10. `quantize_vector` com `array::from_fn`

```rust
pub fn quantize_vector(vector: &[f32; 14]) -> [u8; 14] {
    std::array::from_fn(|i| quantize(vector[i]))
}
```

`array::from_fn` com tamanho fixo em compile time permite ao LLVM vectorizar a operação com SSE2/AVX — processa 4-8 floats por instrução em vez de um por vez.

---

## 11. `HashSet` para `known_merchants`

### Problema
```rust
// O(n) — percorre todos os merchants conhecidos
tx.customer.known_merchants.contains(&tx.merchant.id)
```

### Solução
```rust
// types.rs
pub struct Customer {
    pub known_merchants: std::collections::HashSet<String>,
}
```

`HashSet::contains` computa o hash da string e vai direto no bucket — O(1) independente do tamanho da coleção. Serde desserializa JSON array direto em `HashSet` sem mudança no payload.

---

## Resumo do Impacto

| Otimização | Tipo | Impacto |
|---|---|---|
| LSH em vez de brute force | Algorítmico | O(n) → O(log n) |
| Vetores u8 separados | Memória | -126MB por réplica |
| FxHashMap | CPU | ~3x no lookup |
| `inv_w` | CPU | divisão → multiplicação |
| Bitmap para deduplicação | CPU | elimina overhead de hashing |
| Dirty list | CPU + Memória | reset O(N) → O(candidatos) |
| Pool de buffers | Concorrência | elimina serialização de requests |
| Cap + `select_nth_unstable` | CPU | O(n log n) → O(n) no sort |
| `quantize` simplificada | CPU | -1 operação FP por elemento |
| `HashSet` known_merchants | CPU | O(n) → O(1) por lookup |

---

## Referências

- [LSH — Locality Sensitive Hashing (Andoni & Indyk, 2008)](https://people.csail.mit.edu/indyk/p117-andoni.pdf)
- [HNSW — Hierarchical Navigable Small World (Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320)
- [rustc-hash — FxHashMap](https://github.com/rust-lang/rustc-hash)
- [select_nth_unstable — Rust std](https://doc.rust-lang.org/std/primitive.slice.html#method.select_nth_unstable)
