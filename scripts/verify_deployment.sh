#!/bin/bash
#
# Deployment Verification Script for Kosmos AI Scientist
#
# This script verifies that a Kosmos deployment is healthy and functional.
# Run this after deployment to ensure all components are working correctly.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
KOSMOS_URL="${KOSMOS_URL:-http://localhost:8000}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-kosmos}"
POSTGRES_DB="${POSTGRES_DB:-kosmos}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_PORT="${NEO4J_PORT:-7687}"

ERRORS=0
WARNINGS=0
CHECKS_PASSED=0
TOTAL_CHECKS=0

# Helper functions
print_header() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
}

print_check() {
    echo -n "  Checking $1... "
    ((TOTAL_CHECKS++))
}

print_pass() {
    echo -e "${GREEN}✓ PASS${NC}"
    ((CHECKS_PASSED++))
}

print_fail() {
    echo -e "${RED}✗ FAIL${NC}"
    echo -e "    ${RED}$1${NC}"
    ((ERRORS++))
}

print_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}"
    echo -e "    ${YELLOW}$1${NC}"
    ((WARNINGS++))
}

# Check functions
check_kosmos_health() {
    print_header "Kosmos Application Health"

    print_check "Kosmos API is accessible"
    if curl -sf "${KOSMOS_URL}/health" > /dev/null 2>&1; then
        print_pass
    else
        print_fail "Cannot reach ${KOSMOS_URL}/health"
        return
    fi

    print_check "Kosmos API responds with 'healthy' status"
    HEALTH_STATUS=$(curl -sf "${KOSMOS_URL}/health" | grep -o '"status":"healthy"' || true)
    if [ -n "$HEALTH_STATUS" ]; then
        print_pass
    else
        print_fail "Health check did not return 'healthy' status"
    fi

    print_check "Kosmos readiness check"
    READY_STATUS=$(curl -sf "${KOSMOS_URL}/health/ready" | grep -o '"status":"ready"' || true)
    if [ -n "$READY_STATUS" ]; then
        print_pass
    else
        print_warn "Readiness check did not return 'ready' status (some dependencies may be unavailable)"
    fi

    print_check "Kosmos metrics endpoint"
    if curl -sf "${KOSMOS_URL}/health/metrics" > /dev/null 2>&1; then
        print_pass
    else
        print_warn "Metrics endpoint not accessible"
    fi
}

check_database() {
    print_header "Database Health"

    print_check "PostgreSQL is accessible"
    if command -v pg_isready > /dev/null 2>&1; then
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" > /dev/null 2>&1; then
            print_pass
        else
            print_fail "Cannot connect to PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}"
        fi
    elif command -v psql > /dev/null 2>&1; then
        if PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1" > /dev/null 2>&1; then
            print_pass
        else
            print_fail "Cannot connect to PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}"
        fi
    else
        print_warn "PostgreSQL client tools not available, skipping database check"
    fi
}

check_redis() {
    print_header "Redis Cache Health"

    print_check "Redis is accessible"
    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
            print_pass
        else
            print_warn "Cannot connect to Redis at ${REDIS_HOST}:${REDIS_PORT}"
        fi
    else
        print_warn "redis-cli not available, skipping Redis check"
    fi
}

check_neo4j() {
    print_header "Neo4j Knowledge Graph Health"

    print_check "Neo4j is accessible"
    if command -v wget > /dev/null 2>&1; then
        if wget -q --spider "http://${NEO4J_HOST}:7474" 2>&1; then
            print_pass
        else
            print_warn "Cannot reach Neo4j at ${NEO4J_HOST}:7474 (optional component)"
        fi
    elif command -v curl > /dev/null 2>&1; then
        if curl -sf "http://${NEO4J_HOST}:7474" > /dev/null 2>&1; then
            print_pass
        else
            print_warn "Cannot reach Neo4j at ${NEO4J_HOST}:7474 (optional component)"
        fi
    else
        print_warn "Cannot check Neo4j (wget/curl not available)"
    fi
}

check_environment() {
    print_header "Environment Configuration"

    print_check "ANTHROPIC_API_KEY is set"
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        print_pass
    else
        print_warn "ANTHROPIC_API_KEY not set in environment"
    fi

    print_check "Database URL is configured"
    if [ -n "$DATABASE_URL" ]; then
        print_pass
    else
        print_warn "DATABASE_URL not set in environment"
    fi
}

check_docker() {
    print_header "Docker Deployment Health"

    if command -v docker > /dev/null 2>&1; then
        print_check "Docker containers are running"
        RUNNING_CONTAINERS=$(docker ps --filter "name=kosmos" --format "{{.Names}}" | wc -l)
        if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
            print_pass
            echo "    Found $RUNNING_CONTAINERS Kosmos container(s) running"
        else
            print_warn "No Kosmos containers found running"
        fi

        print_check "No containers in unhealthy state"
        UNHEALTHY=$(docker ps --filter "health=unhealthy" --filter "name=kosmos" --format "{{.Names}}" | wc -l)
        if [ "$UNHEALTHY" -eq 0 ]; then
            print_pass
        else
            print_fail "Found $UNHEALTHY unhealthy container(s)"
        fi
    else
        print_warn "Docker not available, skipping container checks"
    fi
}

check_kubernetes() {
    print_header "Kubernetes Deployment Health"

    if command -v kubectl > /dev/null 2>&1; then
        print_check "Kosmos pods are running"
        RUNNING_PODS=$(kubectl get pods -n kosmos --field-selector=status.phase=Running 2>/dev/null | grep -v NAME | wc -l || echo 0)
        if [ "$RUNNING_PODS" -gt 0 ]; then
            print_pass
            echo "    Found $RUNNING_PODS running pod(s)"
        else
            print_warn "No running Kosmos pods found in namespace 'kosmos'"
        fi

        print_check "No pods in error state"
        ERROR_PODS=$(kubectl get pods -n kosmos --field-selector=status.phase=Failed 2>/dev/null | grep -v NAME | wc -l || echo 0)
        if [ "$ERROR_PODS" -eq 0 ]; then
            print_pass
        else
            print_fail "Found $ERROR_PODS pod(s) in error state"
        fi
    else
        print_warn "kubectl not available, skipping Kubernetes checks"
    fi
}

# Main verification
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║   Kosmos AI Scientist - Deployment Verification Script    ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Verifying deployment at: $KOSMOS_URL"
    echo ""

    # Run all checks
    check_environment
    check_kosmos_health
    check_database
    check_redis
    check_neo4j
    check_docker
    check_kubernetes

    # Print summary
    print_header "Verification Summary"
    echo ""
    echo "  Total checks: $TOTAL_CHECKS"
    echo -e "  ${GREEN}Passed: $CHECKS_PASSED${NC}"
    echo -e "  ${YELLOW}Warnings: $WARNINGS${NC}"
    echo -e "  ${RED}Errors: $ERRORS${NC}"
    echo ""

    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓ Deployment verification PASSED${NC}"
        echo ""
        if [ $WARNINGS -gt 0 ]; then
            echo -e "${YELLOW}Note: $WARNINGS warning(s) found. Review warnings above.${NC}"
        fi
        exit 0
    else
        echo -e "${RED}✗ Deployment verification FAILED${NC}"
        echo -e "${RED}Fix the $ERRORS error(s) above before proceeding.${NC}"
        echo ""
        exit 1
    fi
}

# Run main function
main
