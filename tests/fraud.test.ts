import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));

const BASE_URL = "http://localhost:9999";

type Transaction = {
  id: string;
  transaction: { amount: number; installments: number; requested_at: string };
  customer: { avg_amount: number; tx_count_24h: number; known_merchants: string[] };
  merchant: { id: string; mcc: string; avg_amount: number };
  terminal: { is_online: boolean; card_present: boolean; km_from_home: number };
  last_transaction: { timestamp: string; km_from_current: number } | null;
};

type FraudResult = { approved: boolean; fraud_score: number };

const allPayloads: Transaction[] = JSON.parse(
  readFileSync(join(__dirname, "payloads/example-payloads.json"), "utf-8")
);

function byId(id: string): Transaction {
  const tx = allPayloads.find((p) => p.id === id);
  if (!tx) throw new Error(`payload ${id} not found`);
  return tx;
}

async function score(tx: Transaction): Promise<FraudResult> {
  const res = await fetch(`${BASE_URL}/fraud-score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(tx),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${tx.id}`);
  return res.json() as Promise<FraudResult>;
}

// ---------------------------------------------------------------------------
// Fraudes esperadas — approved: false
// Critérios: ratio de valor extremo, viagem impossível, merchant desconhecido,
//            tx_count alto, MCC de apostas
// ---------------------------------------------------------------------------
const FRAUD_IDS = [
  // --- heurística dos payloads originais ---
  "tx-1788243118", // ratio 63x, 881km, viagem impossível (661km em 6min)
  "tx-2870794086", // ratio 21x, 898km, viagem impossível (428km em 9min)
  "tx-3943816664", // ratio 59x, tx_count 20, viagem impossível (550km em 1min)
  "tx-3795737979", // ratio 15x, tx_count 17, viagem impossível (920km em 4min)
  "tx-3330991687", // ratio 117x, tx_count 20, merchant desconhecido, 952km
  "tx-317390006",  // ratio 22x, merchant desconhecido, viagem impossível (791km em 2min)
  "tx-669205318",  // ratio 63x, tx_count 19, viagem impossível (788km em 7min)
  "tx-1239962632", // ratio 26x, tx_count 19, viagem impossível (799km em 3min)
  "tx-3943856389", // ratio 55x, tx_count 18, MCC 7995, viagem impossível
  "tx-2053987499", // ratio 13.6x, tx_count 11, viagem impossível (966km em 3min)
  "tx-643415378",  // ratio 20x, tx_count 18, viagem impossível (737km em 5min)
  "tx-2201636350", // ratio 13.4x, tx_count 13, viagem impossível (604km em 7min)
  "tx-3894616122", // ratio 28x, MCC 7995, viagem impossível (689km em 2min)
  "tx-232785218",  // ratio 55x, tx_count 15, viagem impossível (859km em 6min)
  "tx-3724573454", // ratio 14.8x, merchant desconhecido, viagem impossível (411km em 9min)
  "tx-1628318673", // ratio 17.8x, tx_count 11, merchant desconhecido, 351km
  "tx-1471757780", // ratio 38.7x, MCC 7801, viagem impossível (699km em 5min)
  // --- ground truth dos vetores de referência ---
  "tx-ref-006", "tx-ref-008", "tx-ref-010", "tx-ref-013", "tx-ref-015",
  "tx-ref-021", "tx-ref-025", "tx-ref-027", "tx-ref-031", "tx-ref-037",
  "tx-ref-042", "tx-ref-052", "tx-ref-053", "tx-ref-056", "tx-ref-058",
  "tx-ref-064", "tx-ref-068", "tx-ref-069", "tx-ref-071", "tx-ref-072",
  "tx-ref-080", "tx-ref-090", "tx-ref-091", "tx-ref-092", "tx-ref-098",
];

// ---------------------------------------------------------------------------
// Legítimas esperadas — approved: true
// Critérios: ratio ~0.5x, merchant conhecido, perto de casa, card present
// ---------------------------------------------------------------------------
const LEGIT_IDS = [
  // --- heurística dos payloads originais ---
  "tx-1329056812", // ratio 0.5x, merchant conhecido, 29km, card present
  "tx-3576980410", // ratio 0.5x, merchant conhecido, 13.7km, card present
  "tx-1841834722", // ratio 0.5x, merchant conhecido, última tx 0.04km
  "tx-2735832589", // ratio 0.5x, merchant conhecido, card present
  "tx-693951394",  // ratio 0.5x, merchant conhecido, 17.4km, card present
  "tx-3777393415", // ratio 0.5x, merchant conhecido, card present
  "tx-2686770673", // ratio 0.5x, merchant conhecido, card present
  "tx-3375683392", // ratio 0.5x, merchant conhecido, card present
  "tx-2910680682", // ratio 0.5x, merchant conhecido, 7.2km, card present
  "tx-2410742111", // ratio 0.5x, merchant conhecido, card present
  "tx-1218477082", // ratio 0.5x, merchant conhecido, card present
  "tx-3902972416", // ratio 0.5x, merchant conhecido, 4.6km, card present
  "tx-1608576718", // ratio 0.5x, merchant conhecido, card present
  "tx-977505054",  // ratio 0.5x, merchant conhecido, card present
  "tx-1241204672", // ratio 0.5x, merchant conhecido, card present
  "tx-66134907",   // ratio 0.5x, merchant conhecido, card present
  "tx-1823654689", // ratio 0.5x, merchant conhecido, 6.4km de casa
  "tx-2931216357", // ratio 0.5x, merchant conhecido, card present
  "tx-3726908500", // ratio 0.5x, merchant conhecido, card present
  "tx-3101938525", // ratio 0.5x, merchant conhecido, 0.84km de casa
  "tx-233303908",  // ratio 0.5x, merchant conhecido, card present
  "tx-3458921981", // ratio 0.5x, merchant conhecido, card present
  "tx-862114234",  // ratio 0.5x, merchant conhecido, card present
  "tx-2221285182", // ratio 0.5x, merchant conhecido (online mas distância normal)
  // --- ground truth dos vetores de referência ---
  "tx-ref-001", "tx-ref-002", "tx-ref-003", "tx-ref-004", "tx-ref-005",
  "tx-ref-007", "tx-ref-009", "tx-ref-011", "tx-ref-012", "tx-ref-014",
  "tx-ref-016", "tx-ref-017", "tx-ref-018", "tx-ref-019", "tx-ref-020",
  "tx-ref-022", "tx-ref-023", "tx-ref-024", "tx-ref-026", "tx-ref-028",
  "tx-ref-029", "tx-ref-030", "tx-ref-032", "tx-ref-033", "tx-ref-034",
  "tx-ref-035", "tx-ref-036", "tx-ref-038", "tx-ref-039", "tx-ref-040",
  "tx-ref-041", "tx-ref-043", "tx-ref-044", "tx-ref-045", "tx-ref-046",
  "tx-ref-047", "tx-ref-048", "tx-ref-049", "tx-ref-050", "tx-ref-051",
  "tx-ref-054", "tx-ref-055", "tx-ref-057", "tx-ref-059", "tx-ref-060",
  "tx-ref-061", "tx-ref-062", "tx-ref-063", "tx-ref-065", "tx-ref-066",
  "tx-ref-067", "tx-ref-070", "tx-ref-073", "tx-ref-074", "tx-ref-075",
  "tx-ref-076", "tx-ref-077", "tx-ref-078", "tx-ref-079", "tx-ref-081",
  "tx-ref-082", "tx-ref-083", "tx-ref-084", "tx-ref-085", "tx-ref-086",
  "tx-ref-087", "tx-ref-088", "tx-ref-089", "tx-ref-093", "tx-ref-094",
  "tx-ref-095", "tx-ref-096", "tx-ref-097", "tx-ref-099", "tx-ref-100",
];

// ---------------------------------------------------------------------------
// Borderline — sinais mistos, apenas loga o fraud_score sem assertion
// ---------------------------------------------------------------------------
const BORDERLINE_IDS = [
  "tx-4112059057", // ratio 8.4x, merchant desconhecido, online
  "tx-2174907811", // ratio 3.6x, merchant desconhecido, online, 132km
  "tx-48575952",   // ratio 0.5x, merchant conhecido, online sem cartão
  "tx-4189182839", // ratio 0.5x, merchant conhecido, online sem cartão
  "tx-386145042",  // valor pequeno, merchant conhecido, online
  "tx-2383971911", // valor pequeno, merchant conhecido, card_present=false
  "tx-2032115259", // ratio 0.5x, merchant conhecido, 2.6km, online
  "tx-1217566662", // ratio 0.5x, merchant conhecido, offline mas card_present=false
];

describe("Fraud Detection — /fraud-score", () => {
  describe("Fraudes esperadas (approved: false)", () => {
    for (const id of FRAUD_IDS) {
      it(id, async () => {
        const result = await score(byId(id));
        expect(
          result.approved,
          `esperado approved=false, fraud_score=${result.fraud_score.toFixed(2)}`
        ).toBe(false);
      });
    }
  });

  describe("Legítimas esperadas (approved: true)", () => {
    for (const id of LEGIT_IDS) {
      it(id, async () => {
        const result = await score(byId(id));
        expect(
          result.approved,
          `esperado approved=true, fraud_score=${result.fraud_score.toFixed(2)}`
        ).toBe(true);
      });
    }
  });

  describe("Borderline — apenas observação (sem assertion)", () => {
    for (const id of BORDERLINE_IDS) {
      it(id, async () => {
        const result = await score(byId(id));
        console.log(
          `  ${id} → approved=${result.approved}, fraud_score=${result.fraud_score.toFixed(2)}`
        );
      });
    }
  });
});
