#!/bin/bash
# Run full inventory pipeline on selected participant(s)
#
# Pipeline Steps:
#   01. Export narrations (filtered for food-related actions)
#   02. Inventory discovery (with intersection deduplication)
#   03. Lifecycle tracking (events for each item)
#   04. Dispensal classification (difficulty rating per item)
#   05. Filter for annotation (items with known quantities)
#
# Usage:
#   ./run_all_participants.sh P01              # Run full pipeline for P01
#   ./run_all_participants.sh P01 --step 02    # Run only step 02 for P01
#   ./run_all_participants.sh all              # Run all participants (P01-P09)

set -e
cd "$(dirname "$0")"

# First argument is participant (required)
if [[ -z "$1" ]]; then
    echo "Usage: $0 <participant|all> [--step <02|03|04|05>]"
    echo "  participant: P01, P02, ... P09, or 'all'"
    echo "  --step: Run only specific step (default: all steps)"
    exit 1
fi

if [[ "$1" == "all" ]]; then
    PARTICIPANTS="P01 P02 P03 P04 P05 P06 P07 P08 P09"
else
    PARTICIPANTS="$1"
fi

# Parse step argument
STEP="all"
if [[ "$2" == "--step" ]]; then
    STEP="$3"
fi

echo "========================================"
echo "Inventory Pipeline - $(date)"
echo "Participants: $PARTICIPANTS"
echo "Step: $STEP"
echo "========================================"

for P in $PARTICIPANTS; do
    echo ""
    echo "=== Processing $P ==="
    echo ""

    if [[ "$STEP" == "all" || "$STEP" == "02" ]]; then
        echo "--- Step 02: Inventory Discovery ---"
        python 02_inventory_discovery.py --participant $P
    fi

    if [[ "$STEP" == "all" || "$STEP" == "03" ]]; then
        echo "--- Step 03: Lifecycle Tracking ---"
        python 03_lifecycle_tracking.py --participant $P
    fi

    if [[ "$STEP" == "all" || "$STEP" == "04" ]]; then
        echo "--- Step 04: Dispensal Classification ---"
        python 04_dispensal_classification.py --participant $P
    fi

    if [[ "$STEP" == "all" || "$STEP" == "05" ]]; then
        echo "--- Step 05: Filter for Annotation ---"
        python 05_filter_for_annotation.py --participant $P
    fi

    echo ""
    echo "=== $P Complete ==="
done

echo ""
echo "========================================"
echo "Pipeline completed at $(date)"
echo "========================================"
