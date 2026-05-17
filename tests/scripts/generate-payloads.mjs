import { readFileSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

// --- Constants matching features.rs ---
const MAX_AMOUNT = 10_000;
const MAX_INSTALLMENTS = 12;
const AMOUNT_VS_AVG_RATIO = 10;
const MAX_MINUTES = 1_440;
const MAX_KM = 1_000;
const MAX_TX_COUNT_24H = 20;
const MAX_MERCHANT_AVG_AMOUNT = 10_000;

// Reverse map: risk value → MCC (from mcc_risk.json)
const mccByRisk = [
  [0.15, "5411"],
  [0.20, "5912"],
  [0.25, "5311"],
  [0.30, "5812"],
  [0.35, "4511"],
  [0.45, "5944"],
  [0.50, "5999"],
  [0.75, "7802"],
  [0.80, "7801"],
  [0.85, "7995"],
];

function findMcc(risk) {
  return mccByRisk.reduce((best, cur) =>
    Math.abs(cur[0] - risk) < Math.abs(best[0] - risk) ? cur : best
  )[1];
}

// Monday=0 … Sunday=6 (Rust's num_days_from_monday)
// Anchor to the week of 2026-03-09 (Mon) – 2026-03-15 (Sun)
const dayToDate = {
  0: "2026-03-09",
  1: "2026-03-10",
  2: "2026-03-11",
  3: "2026-03-12",
  4: "2026-03-13",
  5: "2026-03-14",
  6: "2026-03-15",
};

// Pools of merchant IDs
const KNOWN_POOL = Array.from({ length: 20 }, (_, i) =>
  `MERC-${String(i + 1).padStart(3, "0")}`
);
const UNKNOWN_POOL = ["MERC-061","MERC-062","MERC-063","MERC-064","MERC-065",
                      "MERC-066","MERC-067","MERC-068","MERC-069","MERC-070",
                      "MERC-071","MERC-072","MERC-073","MERC-074","MERC-075"];

function r(n, dec = 2) {
  return Math.round(n * 10 ** dec) / 10 ** dec;
}

function vectorToPayload(vector, idx) {
  const amount = r(vector[0] * MAX_AMOUNT);
  const installments = Math.max(1, Math.round(vector[1] * MAX_INSTALLMENTS));

  // vector[2] = clamp((amount/avg_amount)/10, 0, 1)
  // → avg_amount = amount / (vector[2] * 10)
  const avgAmount = vector[2] > 0
    ? r(amount / (vector[2] * AMOUNT_VS_AVG_RATIO))
    : r(amount * 2);

  const hour = Math.round(vector[3] * 23);
  const dayOfWeek = Math.round(vector[4] * 6);
  const date = dayToDate[dayOfWeek] ?? "2026-03-10";
  const requestedAt = `${date}T${String(hour).padStart(2, "0")}:00:00Z`;

  let lastTransaction = null;
  if (vector[5] !== -1) {
    const minutesAgo = Math.round(vector[5] * MAX_MINUTES);
    const kmFromCurrent = r(vector[6] * MAX_KM);
    const lastDate = new Date(new Date(requestedAt).getTime() - minutesAgo * 60_000);
    lastTransaction = {
      timestamp: lastDate.toISOString().replace(".000", ""),
      km_from_current: kmFromCurrent,
    };
  }

  const kmFromHome = r(vector[7] * MAX_KM);
  const txCount24h = Math.round(vector[8] * MAX_TX_COUNT_24H);
  const isOnline = vector[9] === 1;
  const cardPresent = vector[10] === 1;
  const unknownMerchant = vector[11] === 1;

  const mcc = findMcc(vector[12]);
  const merchantAvgAmount = r(vector[13] * MAX_MERCHANT_AVG_AMOUNT);

  const merchantId = unknownMerchant
    ? UNKNOWN_POOL[idx % UNKNOWN_POOL.length]
    : KNOWN_POOL[idx % KNOWN_POOL.length];

  const knownMerchants = unknownMerchant
    ? [KNOWN_POOL[idx % KNOWN_POOL.length], KNOWN_POOL[(idx + 3) % KNOWN_POOL.length]]
    : [merchantId, KNOWN_POOL[(idx + 5) % KNOWN_POOL.length], KNOWN_POOL[(idx + 11) % KNOWN_POOL.length]];

  return {
    id: `tx-ref-${String(idx + 1).padStart(3, "0")}`,
    transaction: { amount, installments, requested_at: requestedAt },
    customer: { avg_amount: avgAmount, tx_count_24h: txCount24h, known_merchants: knownMerchants },
    merchant: { id: merchantId, mcc, avg_amount: merchantAvgAmount },
    terminal: { is_online: isOnline, card_present: cardPresent, km_from_home: kmFromHome },
    last_transaction: lastTransaction,
  };
}

// --- Load and process ---
const refs = JSON.parse(
  readFileSync(join(root, "..", "docs", "example-references (1).json"), "utf-8")
);

const fraudIds = [];
const legitIds = [];
const newPayloads = [];

refs.forEach((ref, idx) => {
  const payload = vectorToPayload(ref.vector, idx);
  newPayloads.push(payload);
  (ref.label === "fraud" ? fraudIds : legitIds).push(payload.id);
});

// Append to existing payloads
const existing = JSON.parse(
  readFileSync(join(root, "payloads", "example-payloads.json"), "utf-8")
);
const existingIds = new Set(existing.map((p) => p.id));
const toAdd = newPayloads.filter((p) => !existingIds.has(p.id));

writeFileSync(
  join(root, "payloads", "example-payloads.json"),
  JSON.stringify([...existing, ...toAdd], null, 2)
);

console.log(`\nAdded ${toAdd.length} payloads to example-payloads.json\n`);
console.log("FRAUD_IDS from references:");
console.log(JSON.stringify(fraudIds, null, 2));
console.log("\nLEGIT_IDS from references:");
console.log(JSON.stringify(legitIds, null, 2));
