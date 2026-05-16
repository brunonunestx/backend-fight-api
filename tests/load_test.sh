#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

RATE="${RATE:-100}"
DURATION="${DURATION:-30s}"
HOST="${HOST:-http://localhost:9999}"
OUTPUT_DIR="$SCRIPT_DIR/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_BIN="$OUTPUT_DIR/attack_${TIMESTAMP}.bin"
REPORT_TXT="$OUTPUT_DIR/report_${TIMESTAMP}.txt"
PLOT_HTML="$OUTPUT_DIR/plot_${TIMESTAMP}.html"

if ! command -v vegeta &> /dev/null; then
  echo "vegeta não encontrado. Instale com: go install github.com/tsenart/vegeta@latest"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Substitui o host no targets.txt se HOST for diferente do padrão
TARGETS_FILE="$SCRIPT_DIR/targets.txt"
if [[ "$HOST" != "http://localhost:3000" ]]; then
  TARGETS_FILE="$OUTPUT_DIR/targets_tmp.txt"
  sed "s|http://localhost:3000|$HOST|g" "$SCRIPT_DIR/targets.txt" > "$TARGETS_FILE"
fi

echo "Iniciando load test..."
echo "  Host:     $HOST"
echo "  Rate:     $RATE req/s"
echo "  Duration: $DURATION"
echo ""

# Roda o ataque a partir do root para que os paths @tests/payloads/... resolvam corretamente
cd "$ROOT_DIR"

vegeta attack \
  -targets="$TARGETS_FILE" \
  -rate="$RATE" \
  -duration="$DURATION" \
  > "$RESULTS_BIN"

echo "--- Relatório ---"
vegeta report "$RESULTS_BIN" | tee "$REPORT_TXT"

echo ""
echo "--- Latências (percentis) ---"
vegeta report -type=hdrplot "$RESULTS_BIN" | awk -F',' 'NR>1 { printf "p%-6s %s ms\n", $2, $1 }' | head -20

vegeta plot "$RESULTS_BIN" > "$PLOT_HTML"

echo ""
echo "Resultados salvos em:"
echo "  Relatório: $REPORT_TXT"
echo "  Plot HTML: $PLOT_HTML"
echo "  Binário:   $RESULTS_BIN"
