#!/bin/bash
#
# Inventory Pipeline Runner
# Usage: ./run_pipeline.sh <participant> [step]
#
# Examples:
#   ./run_pipeline.sh P04          # Run all steps (with pauses for manual review)
#   ./run_pipeline.sh P04 2        # Run only step 2 (discovery)
#   ./run_pipeline.sh P04 3-6      # Run steps 3 through 6
#   ./run_pipeline.sh P04 auto     # Run all automated steps (skip manual review)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/outputs/02_inventory"

# Default model settings
MODEL="gpt-5.2"
REASONING_EFFORT="high"

# Parse arguments
PARTICIPANT="${1:-}"
STEP="${2:-all}"

if [ -z "$PARTICIPANT" ]; then
    echo -e "${RED}Error: Participant ID required${NC}"
    echo "Usage: $0 <participant> [step]"
    echo ""
    echo "Available participants: P01 P02 P03 P04 P05 P06 P07 P08 P09"
    echo ""
    echo "Steps:"
    echo "  2     - Inventory Discovery"
    echo "  3     - Lifecycle Tracking"
    echo "  4     - Dispensal Classification"
    echo "  5     - Filter for Annotation"
    echo "  6     - Timeline Aggregation"
    echo "  all   - Run all steps (default, pauses for manual review)"
    echo "  auto  - Run all automated steps (no pause)"
    echo "  2-4   - Run steps 2 through 4"
    exit 1
fi

# Validate participant
if [[ ! "$PARTICIPANT" =~ ^P0[1-9]$ ]]; then
    echo -e "${RED}Error: Invalid participant ID '$PARTICIPANT'${NC}"
    echo "Valid IDs: P01, P02, ..., P09"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR/$PARTICIPANT"

# Helper functions
log_step() {
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}STEP $1: $2${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

wait_for_manual_review() {
    local file="$1"
    local dest="$2"

    echo ""
    echo -e "${YELLOW}======================================================================${NC}"
    echo -e "${YELLOW}MANUAL REVIEW REQUIRED${NC}"
    echo -e "${YELLOW}======================================================================${NC}"
    echo ""
    echo "Please review and edit: $file"
    echo "Save as: $dest"
    echo ""

    if [ "$STEP" == "auto" ]; then
        log_warning "Auto mode: copying file without manual review"
        cp "$file" "$dest"
        return
    fi

    read -p "Press Enter when done (or 'skip' to copy as-is): " response

    if [ "$response" == "skip" ] || [ ! -f "$dest" ]; then
        log_warning "Copying file as-is (no manual edits)"
        cp "$file" "$dest"
    else
        log_success "Using manually edited file"
    fi
}

should_run_step() {
    local step_num=$1

    if [ "$STEP" == "all" ] || [ "$STEP" == "auto" ]; then
        return 0
    fi

    if [ "$STEP" == "$step_num" ]; then
        return 0
    fi

    # Handle range like "2-4"
    if [[ "$STEP" =~ ^([0-9])-([0-9])$ ]]; then
        local start="${BASH_REMATCH[1]}"
        local end="${BASH_REMATCH[2]}"
        if [ "$step_num" -ge "$start" ] && [ "$step_num" -le "$end" ]; then
            return 0
        fi
    fi

    return 1
}

# ============================================================
# STEP 2: Inventory Discovery
# ============================================================
run_step_2() {
    log_step 2 "Inventory Discovery"

    local output_file="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_discovery.json"

    python "$SCRIPT_DIR/02_inventory_discovery.py" \
        --participant "$PARTICIPANT" \
        --model "$MODEL" \
        --reasoning \
        --reasoning-effort "$REASONING_EFFORT" \
        --verbose

    if [ -f "$output_file" ]; then
        log_success "Created: ${PARTICIPANT}_discovery.json"
    else
        log_error "Failed to create discovery file"
        exit 1
    fi
}

# ============================================================
# STEP 2.5: Manual Review of Discovery
# ============================================================
run_step_2_5() {
    local src="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_discovery.json"
    local dest="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_discovery_edit.json"

    if [ -f "$dest" ]; then
        log_warning "discovery_edit.json already exists, skipping manual review"
        return
    fi

    wait_for_manual_review "$src" "$dest"
}

# ============================================================
# STEP 3: Lifecycle Tracking
# ============================================================
run_step_3() {
    log_step 3 "Lifecycle Tracking"

    local input_file="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_discovery_edit.json"

    if [ ! -f "$input_file" ]; then
        log_error "Missing input: ${PARTICIPANT}_discovery_edit.json"
        log_error "Run step 2 and manual review first"
        exit 1
    fi

    python "$SCRIPT_DIR/03_lifecycle_tracking.py" \
        --participant "$PARTICIPANT" \
        --model "$MODEL" \
        --reasoning \
        --reasoning-effort "$REASONING_EFFORT" \
        --verbose

    log_success "Created: ${PARTICIPANT}_lifecycle.json"
}

# ============================================================
# STEP 4: Dispensal Classification
# ============================================================
run_step_4() {
    log_step 4 "Dispensal Classification"

    local input_file="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_lifecycle.json"

    if [ ! -f "$input_file" ]; then
        log_error "Missing input: ${PARTICIPANT}_lifecycle.json"
        log_error "Run step 3 first"
        exit 1
    fi

    python "$SCRIPT_DIR/04_dispensal_classification.py" \
        --participant "$PARTICIPANT" \
        --model "$MODEL" \
        --verbose

    log_success "Created: ${PARTICIPANT}_dispensal_classified.json"
}

# ============================================================
# STEP 5: Filter for Annotation
# ============================================================
run_step_5() {
    log_step 5 "Filter for Annotation"

    local input_file="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_dispensal_classified.json"

    if [ ! -f "$input_file" ]; then
        log_error "Missing input: ${PARTICIPANT}_dispensal_classified.json"
        log_error "Run step 4 first"
        exit 1
    fi

    python "$SCRIPT_DIR/05_filter_for_annotation.py" \
        --participant "$PARTICIPANT" \
        --verbose

    log_success "Created: ${PARTICIPANT}_known_quantities.json"
    log_success "Created: ${PARTICIPANT}_known_quantities.txt"
}

# ============================================================
# STEP 6: Timeline Aggregation
# ============================================================
run_step_6() {
    log_step 6 "Timeline Aggregation"

    local input_file="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_known_quantities.json"

    if [ ! -f "$input_file" ]; then
        log_error "Missing input: ${PARTICIPANT}_known_quantities.json"
        log_error "Run step 5 first"
        exit 1
    fi

    python "$SCRIPT_DIR/06_timeline_aggregation.py" \
        --participant "$PARTICIPANT" \
        --model "$MODEL" \
        --reasoning-effort "$REASONING_EFFORT" \
        --verbose

    log_success "Created: ${PARTICIPANT}_timeline_aggregated.json"
}

# ============================================================
# STEP 6.5: Manual Review of Timeline
# ============================================================
run_step_6_5() {
    local src="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_timeline_aggregated.json"
    local dest="$OUTPUT_DIR/$PARTICIPANT/${PARTICIPANT}_timeline_annotated.json"

    if [ -f "$dest" ]; then
        log_warning "timeline_annotated.json already exists, skipping manual review"
        return
    fi

    wait_for_manual_review "$src" "$dest"
}

# ============================================================
# MAIN EXECUTION
# ============================================================

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}INVENTORY PIPELINE - $PARTICIPANT${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Model: $MODEL"
echo "Reasoning: $REASONING_EFFORT"
echo "Steps: $STEP"
echo ""

# Run requested steps
if should_run_step 2; then
    run_step_2
    if [ "$STEP" == "all" ]; then
        run_step_2_5
    fi
fi

if should_run_step 3; then
    run_step_3
fi

if should_run_step 4; then
    run_step_4
fi

if should_run_step 5; then
    run_step_5
fi

if should_run_step 6; then
    run_step_6
    if [ "$STEP" == "all" ]; then
        run_step_6_5
    fi
fi

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}PIPELINE COMPLETE${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Output files in: $OUTPUT_DIR/$PARTICIPANT/"
ls -la "$OUTPUT_DIR/$PARTICIPANT/"
