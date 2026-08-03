#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission script for running ELLA on Visium HD data (full analysis)
#
# Submit one job per gene × sample.  Adjust paths, partition, memory, and
# Python environment to match your HPC cluster.
#
# Usage (from the login node, after activating your Python environment):
#   bash run_ella_slurm.sh B408
#   bash run_ella_slurm.sh B573
# ─────────────────────────────────────────────────────────────────────────────

set -e

SAMPLE="${1:?Usage: $0 <SAMPLE_ID>}"

BASE_DIR="/path/to/SubcellularExpressionPatern"   # ← update this
PYTHON_BIN="/path/to/python"                        # ← update this

SDIR="${BASE_DIR}/${SAMPLE}"
PREPARED="${SDIR}/prepared_data/training_data.jsonl"
LOG_DIR="${SDIR}/slurm_logs"

mkdir -p "$LOG_DIR"

N_GENES=$(wc -l < "$PREPARED")
echo "Submitting ${N_GENES} jobs for ${SAMPLE}..."

for GENE_IDX in $(seq 0 $((N_GENES - 1))); do
    sbatch \
        --job-name="ella_${SAMPLE}_g${GENE_IDX}" \
        --output="${LOG_DIR}/g${GENE_IDX}_%j.out" \
        --error="${LOG_DIR}/g${GENE_IDX}_%j.err" \
        --time=2:00:00 \
        --mem=8G \
        --cpus-per-task=1 \
        --partition=cpu \
        --wrap "${PYTHON_BIN} -c \"
import subprocess, sys
cmd = [
    'ella-train',
    '--config-path=${SDIR}',
    '--config-name=ella_config',
    'data.gene_idx=${GENE_IDX}',
]
r = subprocess.run(cmd, cwd='${SDIR}')
sys.exit(r.returncode)
\""
done

echo "All ${N_GENES} jobs submitted."
echo "Monitor with: squeue -u \$USER | grep ella_${SAMPLE}"
