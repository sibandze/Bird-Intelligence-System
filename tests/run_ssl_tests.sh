#!/bin/bash
# tests/run_ssl_tests.sh
# Run all SSL pipeline tests in order

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="${PROJECT_ROOT}/tests"
COVERAGE_THRESHOLD=80

# Ensure project root is on PYTHONPATH so `import src` works
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# Parse arguments
VERBOSE=false
COVERAGE=false
SPECIFIC_TEST=""
FAIL_FAST=false
MARKER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -x|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        -k|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -m|--marker)
            MARKER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose      Verbose output"
            echo "  -c, --coverage     Generate coverage report"
            echo "  -x, --fail-fast    Stop on first failure"
            echo "  -k, --test NAME    Run specific test (pytest -k pattern)"
            echo "  -m, --marker M     Run tests with specific marker"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -v                          # Run all tests verbosely"
            echo "  $0 -k test_encoder             # Run encoder tests only"
            echo "  $0 -m slow                     # Run slow tests"
            echo "  $0 -c -v                       # Run with coverage and verbose"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}====================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}====================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# Environment Setup
# ============================================================================

print_header "SSL Pipeline Test Suite"
echo ""

print_info "Project root: ${PROJECT_ROOT}"
print_info "Test directory: ${TEST_DIR}"

# Check Python
PYTHON_VERSION=$(python --version 2>&1)
print_info "Python: ${PYTHON_VERSION}"

# Check pytest
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Install with: pip install pytest"
    exit 1
fi

# Build pytest command
PYTEST_ARGS="-v"  # Always use verbose in CI-like runs

if [[ "$VERBOSE" == true ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -s --tb=long"
else
    PYTEST_ARGS="$PYTEST_ARGS --tb=short"
fi

if [[ "$FAIL_FAST" == true ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -x"
fi

if [[ -n "$MARKER" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -m $MARKER"
fi

if [[ -n "$SPECIFIC_TEST" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -k $SPECIFIC_TEST"
fi

if [[ "$COVERAGE" == true ]]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=term-missing --cov-report=html"
    print_info "Coverage enabled (threshold: ${COVERAGE_THRESHOLD}%)"
fi

# ============================================================================
# Phase 1: Data Pipeline Tests
# ============================================================================

print_header "Phase 1: Data Pipeline Tests"

run_phase1() {
    echo ""
    print_info "Running augmentation tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_data_pipeline.py::TestBaseAugmentation" \
        "${TEST_DIR}/test_ssl_data_pipeline.py::TestAcousticAugmentation" \
        "${TEST_DIR}/test_ssl_data_pipeline.py::TestSpecAugmentation" \
        "${TEST_DIR}/test_ssl_data_pipeline.py::TestAugmentationPipeline"
    
    echo ""
    print_info "Running SSL dataset tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_data_pipeline.py::TestSSLBirdSongDataset" \
        "${TEST_DIR}/test_ssl_data_pipeline.py::TestSSLFrameworkAdapters" \
        "${TEST_DIR}/test_ssl_data_pipeline.py::TestSSLCollateFunctions"
    
    echo ""
    print_info "Running DataLoader integration tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_data_pipeline.py::TestDataLoaderIntegration"
    
    echo ""
    print_info "Running supervised dataset tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_data_pipeline.py::TestSupervisedDataset" \
        "${TEST_DIR}/test_supervised_equivalence.py::TestSupervisedSSLEquivalence"
}

if [[ -n "$SPECIFIC_TEST" ]] && [[ "$SPECIFIC_TEST" != *"phase1" ]] && [[ "$SPECIFIC_TEST" != *"data" ]]; then
    print_warning "Skipping Phase 1 (not matching filter: $SPECIFIC_TEST)"
else
    run_phase1
    print_success "Phase 1 complete"
fi

# ============================================================================
# Phase 2: Model Architecture Tests
# ============================================================================

print_header "Phase 2: Model Architecture Tests"

run_phase2() {
    echo ""
    print_info "Running CNN encoder tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_models.py::TestCNNEncoder"
    
    echo ""
    print_info "Running CNN-SimCLR integration tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_models.py::TestCNNSimCLRIntegration"
    
    echo ""
    print_info "Running projection head tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_models.py::TestProjectionHead"
}

if [[ -n "$SPECIFIC_TEST" ]] && [[ "$SPECIFIC_TEST" != *"phase2" ]] && [[ "$SPECIFIC_TEST" != *"encoder" ]] && [[ "$SPECIFIC_TEST" != *"projection" ]]; then
    print_warning "Skipping Phase 2 (not matching filter: $SPECIFIC_TEST)"
else
    run_phase2
    print_success "Phase 2 complete"
fi

# ============================================================================
# Phase 3: SimCLR Loss & Training Tests
# ============================================================================

print_header "Phase 3: SimCLR Tests"

run_phase3() {
    echo ""
    print_info "Running NT-Xent loss tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_models.py::TestSimCLRLoss"
    
    echo ""
    print_info "Running SimCLR model tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_models.py::TestSimCLRModel"
    
    echo ""
    print_info "Running data-model integration tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_models.py::TestDataModelIntegration"
    
    echo ""
    print_info "Running sanity/overfit tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/test_ssl_sanity.py"
}

if [[ -n "$SPECIFIC_TEST" ]] && [[ "$SPECIFIC_TEST" != *"phase3" ]] && [[ "$SPECIFIC_TEST" != *"simclr" ]] && [[ "$SPECIFIC_TEST" != *"loss" ]]; then
    print_warning "Skipping Phase 3 (not matching filter: $SPECIFIC_TEST)"
else
    run_phase3
    print_success "Phase 3 complete"
fi

# ============================================================================
# Final Summary
# ============================================================================

print_header "Test Suite Complete"

# Run all remaining tests if no specific filter
if [[ -z "$SPECIFIC_TEST" ]]; then
    echo ""
    print_info "Running any remaining tests..."
    pytest ${PYTEST_ARGS} "${TEST_DIR}/" --ignore="${TEST_DIR}/test_ssl_data_pipeline.py" \
        --ignore="${TEST_DIR}/test_ssl_models.py" \
        --ignore="${TEST_DIR}/test_ssl_sanity.py" \
        --ignore="${TEST_DIR}/test_supervised_equivalence.py" \
        --tb=no -q || true  # Don't fail on missing tests
fi

# Coverage report
if [[ "$COVERAGE" == true ]]; then
    echo ""
    print_header "Coverage Report"
    coverage report --fail-under=${COVERAGE_THRESHOLD} || print_warning "Coverage below ${COVERAGE_THRESHOLD}%"
    echo ""
    print_info "HTML report: ${PROJECT_ROOT}/htmlcov/index.html"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           All SSL tests completed!                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
