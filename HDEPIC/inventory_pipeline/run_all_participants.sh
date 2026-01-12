#!/bin/bash
# Run full inventory pipeline on selected participant(s)
# Recipe-based processing: all videos for a recipe processed together
#
# Phases:
#   1. Process recipes (inventory discovery + ingredient mapping + lifecycle tracking)
#   2. Export lifecycle edits and classify dispensal difficulty
#
# Usage:
#   ./run_all_participants.sh P03                    # Run all phases for P03
#   ./run_all_participants.sh P01 --phase1           # Only process recipes for P01
#   ./run_all_participants.sh P02 --phase2           # Only export & classify for P02
#   ./run_all_participants.sh all                    # Run all participants (P01-P09)
#   ./run_all_participants.sh all --phase1           # Process recipes for all

set -e
cd /home/kailaic/NeuroTrace/kitchen/HDEPIC/inventory_pipeline
LOG="../outputs/batch_processing.log"

# First argument is participant (required)
if [[ -z "$1" ]]; then
    echo "Usage: $0 <participant|all> [--phase1|--phase2]"
    echo "  participant: P01, P02, ... P09, or 'all'"
    echo "  --phase1: Only process recipes (inventory + mapping + lifecycle)"
    echo "  --phase2: Only export & classify"
    exit 1
fi

if [[ "$1" == "all" ]]; then
    PARTICIPANTS="P01 P02 P03 P04 P05 P06 P07 P08 P09"
else
    PARTICIPANTS="$1"
fi

# Parse phase argument
PHASE1=true
PHASE2=true

if [[ "$2" == "--phase1" ]]; then
    PHASE2=false
elif [[ "$2" == "--phase2" ]]; then
    PHASE1=false
fi

echo "========================================" | tee $LOG
echo "Starting batch processing at $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG

# Phase 1: Process recipes for participant(s)
# Uses recipe-based mode: all videos concatenated, cross-video lifecycle tracking
if $PHASE1; then
    echo "" | tee -a $LOG
    echo "=== PHASE 1: PROCESS RECIPES FOR $PARTICIPANTS ===" | tee -a $LOG
    echo "" | tee -a $LOG

    for P in $PARTICIPANTS; do
        echo "--- Processing recipes for $P ---" | tee -a $LOG
        python 02_inventory_transactions.py --process-recipes --participant $P --reasoning 2>&1 | tee -a $LOG
    done

    echo "" | tee -a $LOG
    echo "Phase 1 completed at $(date)" | tee -a $LOG
fi

# Phase 2: Export lifecycle edits and classify
if $PHASE2; then
    echo "" | tee -a $LOG
    echo "=== PHASE 2: EXPORT & CLASSIFY ===" | tee -a $LOG
    echo "" | tee -a $LOG

    for P in $PARTICIPANTS; do
        echo "--- Exporting $P ---" | tee -a $LOG
        python 03_export_lifecycle_edits.py --participant $P 2>&1 | tee -a $LOG
        python 04_dispensal_classification.py --participant $P 2>&1 | tee -a $LOG
    done

    echo "" | tee -a $LOG
    echo "Phase 2 completed at $(date)" | tee -a $LOG
fi

echo "" | tee -a $LOG
echo "========================================" | tee -a $LOG
echo "Batch processing completed at $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG
