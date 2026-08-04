#!/usr/bin/env bash
# Nightly backup of the data volume (corpus, ChromaDB index, BM25 cache).
# Usage: scripts/backup_data.sh [BACKUP_DIR] [KEEP]
# Cron example: 0 3 * * * /app/scripts/backup_data.sh /backups 7
set -euo pipefail
BACKUP_DIR="${1:-./backups}"
KEEP="${2:-7}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/gurbani-data-$STAMP.tar.gz" data/
# prune old backups beyond KEEP
ls -1t "$BACKUP_DIR"/gurbani-data-*.tar.gz 2>/dev/null | tail -n "+$((KEEP + 1))" | xargs -r rm -f
echo "Backup written: $BACKUP_DIR/gurbani-data-$STAMP.tar.gz (keeping last $KEEP)"
