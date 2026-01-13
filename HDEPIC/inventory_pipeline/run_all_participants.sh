#!/bin/bash
# Run full inventory pipeline on selected participant(s)
#
# Pipeline Steps:
#   01. Export narrations (optional - filtered for food-related actions)
#   02. Inventory discovery (with intersection deduplication)
#   03. Lifecycle tracking (events for each item)
#   04. Dispensal classification (difficulty rating per item)
#   05. Filter for annotation (items with known quantities)
#
# Usage:
#   ./run_all_participants.sh P01              # Run steps 02-05 for P01
#   ./run_all_participants.sh P01 --step 02    # Run only step 02 for P01
#   ./run_all_participants.sh P01 --all        # Run all steps including 01
#   ./run_all_participants.sh all              # Run steps 02-05 for all (P01-P09)

set -e
cd "$(dirname "$0")"

# First argument is participant (required)
if [[ -z "$1" ]]; then
    echo "Usage: $0 <participant|all> [--step <01|02|03|04|05>] [--all]"
    echo "  participant: P01, P02, ... P09, or 'all'"
    echo "  --step: Run only specific step"
    echo "  --all: Include step 01 (export narrations)"
    echo ""
    echo "Default: runs steps 02-05 (skips 01)"
    exit 1
fi

if [[ "$1" == "all" ]]; then
    PARTICIPANTS="P01 P02 P03 P04 P05 P06 P07 P08 P09"
else
    PARTICIPANTS="$1"
fi

# Parse arguments
STEP=""
INCLUDE_01=false

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            STEP="$2"
            shift 2
            ;;
        --all)
            INCLUDE_01=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "Inventory Pipeline - $(date)"
echo "Participants: $PARTICIPANTS"
if [[ -n "$STEP" ]]; then
    echo "Step: $STEP"
else
    if $INCLUDE_01; then
        echo "Steps: 01, 02, 03, 04, 05"
    else
        echo "Steps: 02, 03, 04, 05 (skipping 01)"
    fi
fi
echo "========================================"

for P in $PARTICIPANTS; do
    echo ""
    echo "=== Processing $P ==="
    echo ""

    if [[ "$STEP" == "01" ]] || { [[ -z "$STEP" ]] && $INCLUDE_01; }; then
        echo "--- Step 01: Export Narrations ---"
        python 01_export_narrations.py filter --participant $P
    fi

    if [[ "$STEP" == "02" ]] || [[ -z "$STEP" ]]; then
        echo "--- Step 02: Inventory Discovery ---"
        python 02_inventory_discovery.py --participant $P
    fi

    if [[ "$STEP" == "03" ]] || [[ -z "$STEP" ]]; then
        echo "--- Step 03: Lifecycle Tracking ---"
        python 03_lifecycle_tracking.py --participant $P
    fi

    if [[ "$STEP" == "04" ]] || [[ -z "$STEP" ]]; then
        echo "--- Step 04: Dispensal Classification ---"
        python 04_dispensal_classification.py --participant $P
    fi

    if [[ "$STEP" == "05" ]] || [[ -z "$STEP" ]]; then
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
