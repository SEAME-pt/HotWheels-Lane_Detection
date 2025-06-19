#!/bin/bash
# Clean old data
rm -f coverage.info
rm -rf coverage-report

# Capture coverage
lcov --capture --directory . --output-file coverage.info --no-external

# Exclude test and system files
lcov --remove coverage.info "*/tests/*" "*/usr/*" "*/test_*" "*/mocks/*" --output-file coverage.info

# Generate HTML
genhtml coverage.info --output-directory coverage-report
