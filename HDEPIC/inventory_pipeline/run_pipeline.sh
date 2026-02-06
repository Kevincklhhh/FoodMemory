#!/bin/bash
#
# run_pipeline.sh - Run inventory pipeline steps 3-6 for a participant
#
# Usage: ./run_pipeline.sh P04
#
# Prerequisites:
#   - Step 2 completed: {participant}_discovery_edit.json must exist
#
# Steps executed:
#   3. Lifecycle Tracking
#   4. Dispensal Classification
#   5. Filter for Annotation
#   6. Timeline Aggregation

set -e  # Exit on error

# Check argument
if [ -z "$1" ]; then
    echo "Usage: $0 <participant_id>"
    echo "Example: $0 P04"
    exit 1
fi

PARTICIPANT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/02_inventory/${PARTICIPANT}"

echo "========================================"
echo "Running pipeline for: ${PARTICIPANT}"
echo "========================================"

# Check prerequisite
if [ ! -f "${OUTPUT_DIR}/${PARTICIPANT}_discovery_edit.json" ]; then
    echo "ERROR: ${PARTICIPANT}_discovery_edit.json not found"
    echo "Please complete Step 2 (discovery) and create the _edit file first."
    exit 1
fi

# Step 3: Lifecycle Tracking
echo ""
echo "========================================"
echo "Step 3: Lifecycle Tracking"
echo "========================================"
python "${SCRIPT_DIR}/03_lifecycle_tracking.py" \
    --participant "${PARTICIPANT}" \
    --model gpt-5.2 \
    --reasoning \
    --reasoning-effort high \
    --verbose

# Step 4: Dispensal Classification
echo ""
echo "========================================"
echo "Step 4: Dispensal Classification"
echo "========================================"
python "${SCRIPT_DIR}/04_dispensal_classification.py" \
    --participant "${PARTICIPANT}" \
    --model gpt-5.2 \
    --verbose

# Step 5: Filter for Annotation
echo ""
echo "========================================"
echo "Step 5: Filter for Annotation"
echo "========================================"
python "${SCRIPT_DIR}/05_filter_for_annotation.py" \
    --participant "${PARTICIPANT}" \
    --verbose

# Step 6: Timeline Aggregation
echo ""
echo "========================================"
echo "Step 6: Timeline Aggregation"
echo "========================================"
python "${SCRIPT_DIR}/06_timeline_aggregation.py" \
    --participant "${PARTICIPANT}" \
    --model gpt-5.2 \
    --reasoning-effort high \
    --verbose

echo ""
echo "========================================"
echo "Pipeline complete for ${PARTICIPANT}"
echo "========================================"
echo ""
echo "Output files:"
echo "  - ${PARTICIPANT}_lifecycle.json"
echo "  - ${PARTICIPANT}_dispensal_classified.json"
echo "  - ${PARTICIPANT}_known_quantities.json"
echo "  - ${PARTICIPANT}_timeline_aggregated.json"
echo ""
echo "Next steps:"
echo "  1. Copy and verify timeline:"
echo "     cp ${OUTPUT_DIR}/${PARTICIPANT}_timeline_aggregated.json \\"
echo "        ${OUTPUT_DIR}/${PARTICIPANT}_timeline_annotated.json"
echo ""
echo "  2. Run VLM Q&A:"
echo "     python 07_vlm_QA.py --participant ${PARTICIPANT} --tag qwen_v1"
