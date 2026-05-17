# Test Plan — Fraud Detection API

## Contexto

A API expõe `POST /fraud-score` na porta 9999 (load balancer via nginx).

**Request:** payload `Transaction` (formato dos exemplos em `payloads/example-payloads.json`)

**Response:**
```json
{ "approved": bool, "fraud_score": float }
```

`approved` é `true` quando menos de 3 dos 5 vizinhos mais próximos (KNN com k=5) são fraude.
`fraud_score` é a proporção: `fraud_neighbors / 5`.

---

## Como o modelo decide

O `FraudService` usa LSH + KNN (k=5). Vetoriza 14 features por transação:

| # | Feature | Normalização |
|---|---------|-------------|
| 0 | amount | / 10.000 |
| 1 | installments | / 12 |
| 2 | amount / customer.avg_amount | / 10 |
| 3 | hora do dia | / 23 |
| 4 | dia da semana | / 6 |
| 5 | minutos desde última tx | / 1.440 (−1 se nenhuma) |
| 6 | km da última tx | / 1.000 (−1 se nenhuma) |
| 7 | km_from_home | / 1.000 |
| 8 | tx_count_24h | / 20 |
| 9 | is_online | 0 / 1 |
| 10 | card_present | 0 / 1 |
| 11 | merchant desconhecido | 0 / 1 |
| 12 | mcc_risk | conforme mcc_risk.json |
| 13 | merchant.avg_amount | / 10.000 |

---

## Sinais de fraude usados para rotular os payloads

1. **Razão de valor** (`amount / customer.avg_amount`): ratio > 5x é suspeito, > 10x é forte sinal.
2. **Volume de transações**: `tx_count_24h` alto (> 10) indica comportamento anômalo.
3. **Merchant desconhecido**: `merchant.id` não está em `customer.known_merchants`.
4. **Viagem impossível**: intervalo curto desde a última tx mas grande distância (`km_from_current`). Ex.: 700 km em 2 min.
5. **Distância de casa**: `km_from_home` > 300 km combinado com outros sinais.
6. **Canal de alto risco**: `is_online=true` + `card_present=false` + merchant desconhecido.
7. **MCC de risco**: `7801`, `7802`, `7995` (jogos/apostas) têm `mcc_risk` alto.

---

## Classificação dos payloads

### Fraude esperada (`approved: false`)

| id | Razão principal |
|----|----------------|
| tx-1788243118 | ratio 63x, tx_count 18, merchant desconhecido, 881km, viagem impossível (661km em 6min) |
| tx-2870794086 | ratio 21x, tx_count 13, merchant desconhecido, 898km, viagem impossível (428km em 9min) |
| tx-3943816664 | ratio 59x, tx_count 20 (máx), merchant desconhecido, viagem impossível (550km em 1min) |
| tx-3795737979 | ratio 15x, tx_count 17, merchant desconhecido, viagem impossível (920km em 4min) |
| tx-3330991687 | ratio 117x, tx_count 20 (máx), merchant desconhecido, 952km de casa |
| tx-317390006 | ratio 22x, tx_count 10, merchant desconhecido, viagem impossível (791km em 2min) |
| tx-669205318 | ratio 63x, tx_count 19, merchant desconhecido, viagem impossível (788km em 7min) |
| tx-1239962632 | ratio 26x, tx_count 19, merchant desconhecido, viagem impossível (799km em 3min) |
| tx-3943856389 | ratio 55x, tx_count 18, merchant desconhecido, MCC 7995, viagem impossível |
| tx-2053987499 | ratio 13.6x, tx_count 11, merchant desconhecido, viagem impossível (966km em 3min) |
| tx-643415378 | ratio 20x, tx_count 18, merchant desconhecido, viagem impossível (737km em 5min) |
| tx-2201636350 | ratio 13.4x, tx_count 13, merchant desconhecido, viagem impossível (604km em 7min) |
| tx-3894616122 | ratio 28x, merchant desconhecido, viagem impossível (689km em 2min), MCC 7995 |
| tx-232785218 | ratio 55x, tx_count 15, merchant desconhecido, viagem impossível (859km em 6min) |
| tx-3724573454 | ratio 14.8x, merchant desconhecido, viagem impossível (411km em 9min) |
| tx-1628318673 | ratio 17.8x, tx_count 11, merchant desconhecido, 351km |
| tx-1471757780 | ratio 38.7x, merchant desconhecido, viagem impossível (699km em 5min), MCC 7801 |

### Legítimas esperadas (`approved: true`)

| id | Motivo |
|----|--------|
| tx-1329056812 | ratio normal, merchant conhecido, perto de casa, sem histórico suspeito |
| tx-3576980410 | ratio normal, merchant conhecido, perto de casa |
| tx-1841834722 | ratio normal, merchant conhecido, última tx 0.04km |
| tx-2735832589 | ratio normal, merchant conhecido, card present |
| tx-693951394 | ratio normal, merchant conhecido, perto de casa |
| tx-3777393415 | ratio normal, merchant conhecido, card present |
| tx-2686770673 | ratio normal, merchant conhecido, card present |
| tx-3375683392 | ratio normal, merchant conhecido, card present |
| tx-2910680682 | ratio normal, merchant conhecido, perto de casa |
| tx-2410742111 | ratio normal, merchant conhecido, card present |
| tx-1218477082 | ratio normal, merchant conhecido, card present |
| tx-3902972416 | ratio normal, merchant conhecido, perto de casa |
| tx-1608576718 | ratio normal, merchant conhecido, card present |
| tx-977505054 | ratio normal, merchant conhecido, card present |
| tx-1241204672 | ratio normal, merchant conhecido, card present |
| tx-66134907 | ratio normal, merchant conhecido, card present |
| tx-1823654689 | ratio normal, merchant conhecido, muito perto de casa |
| tx-2931216357 | ratio normal, merchant conhecido, card present |
| tx-3726908500 | ratio normal, merchant conhecido, perto de casa |
| tx-3101938525 | ratio normal, merchant conhecido, 0.84km de casa |
| tx-233303908 | ratio normal, merchant conhecido, card present |
| tx-3458921981 | ratio normal, merchant conhecido, card present |
| tx-862114234 | ratio normal, merchant conhecido, card present |
| tx-2221285182 | ratio normal, merchant conhecido (online mas distância normal) |

### Borderline (incerto — modelo decide)

Esses casos têm sinais mistos. Não vamos assertar `approved`, apenas logar o `fraud_score`.

| id | Por quê é incerto |
|----|-------------------|
| tx-4112059057 | ratio 8.4x, merchant desconhecido, online — pode ser rejeitado |
| tx-2174907811 | ratio 3.6x, merchant desconhecido, online, 132km |
| tx-48575952 | ratio normal, merchant conhecido, online mas sem cartão |
| tx-4189182839 | ratio normal, merchant conhecido, online mas sem cartão |
| tx-386145042 | valor pequeno, merchant conhecido, online |
| tx-2383971911 | valor pequeno, merchant conhecido, card_present=false |
| tx-2032115259 | ratio normal, merchant conhecido, muito perto de casa, online |
| tx-1217566662 | ratio normal, merchant conhecido, offline mas card_present=false |
| tx-2686770673 | ratio normal, merchant conhecido, card present |

---

## Estrutura de testes proposta

```
tests/
├── payloads/
│   └── example-payloads.json    (já existe)
├── fraud.test.ts                (testes com assertions)
├── vitest.config.ts
├── package.json
└── test-plan.md                 (este arquivo)
```

### `fraud.test.ts`

Cada caso de teste faz um `POST localhost:9999/fraud-score` e:
- Para **fraudes**: `expect(approved).toBe(false)`
- Para **legítimas**: `expect(approved).toBe(true)`
- Para **borderline**: sem assertion — apenas loga `id`, `approved` e `fraud_score`

### Output esperado

```
FRAUD cases    → todos approved=false
LEGIT cases    → todos approved=true
BORDERLINE     → tabela com fraud_score para análise manual
```

---

## Como rodar

```bash
cd tests
npm install
npm test
```

> Requer a stack de pé (`docker compose up`). O LB deve estar respondendo em `localhost:9999`.
