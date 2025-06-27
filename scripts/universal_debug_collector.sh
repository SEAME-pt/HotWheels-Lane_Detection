#!/bin/bash

# Universal Remote Debug Collector
# Configurable script for collecting logs from any remote embedded project
# Supports both tmux legacy and modern debug systems

# =============================================================================
# CONFIGURATION - Edit these variables for your project
# =============================================================================

# Default values (can be overridden by environment variables or command line)
REMOTE_USER="${REMOTE_USER:-jetson}"
REMOTE_HOST="${REMOTE_HOST:-hotwheels-car.netbird.cloud}"
REMOTE_PATH="${REMOTE_PATH:-/home/jetson/Documents/MPC}"
LOCAL_PATH="${LOCAL_PATH:-$(pwd)}"
TMUX_SESSION="${TMUX_SESSION:-MPC}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

show_usage() {
    echo "Usage: $0 [OPTIONS] [MODE]"
    echo ""
    echo "Universal Remote Debug Collector"
    echo "Collects logs from remote embedded systems"
    echo ""
    echo "MODES:"
    echo "  debug     - Collect modern debug logs (default)"
    echo "  tmux      - Collect legacy tmux output"
    echo "  both      - Collect both debug logs and tmux"
    echo "  monitor   - Real-time monitoring"
    echo ""
    echo "OPTIONS:"
    echo "  -u USER   - Remote username (default: $REMOTE_USER)"
    echo "  -h HOST   - Remote hostname (default: $REMOTE_HOST)"
    echo "  -r PATH   - Remote project path (default: $REMOTE_PATH)"
    echo "  -l PATH   - Local output path (default: $LOCAL_PATH)"
    echo "  -s NAME   - Tmux session name (default: $TMUX_SESSION)"
    echo "  -q        - Quiet mode (less output)"
    echo "  -v        - Verbose mode (more output)"
    echo "  --help    - Show this help"
    echo ""
    echo "ENVIRONMENT VARIABLES:"
    echo "  REMOTE_USER, REMOTE_HOST, REMOTE_PATH, LOCAL_PATH, TMUX_SESSION"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Collect debug logs"
    echo "  $0 -u pi -h raspberrypi.local debug  # Raspberry Pi"
    echo "  $0 -r /opt/myproject tmux            # Different path"
    echo "  $0 monitor                           # Real-time monitoring"
}

check_connection() {
    print_info "Testing connection to $REMOTE_USER@$REMOTE_HOST..."
    if ssh -o ConnectTimeout=5 -o BatchMode=yes $REMOTE_USER@$REMOTE_HOST "exit" 2>/dev/null; then
        print_status "Connection successful"
        return 0
    else
        print_error "Cannot connect to remote host"
        print_info "Check: network, SSH keys, hostname, username"
        return 1
    fi
}

collect_debug_logs() {
    print_info "Collecting modern debug logs..."
    
    # Check for debug logs
    AVAILABLE_LOGS=$(ssh $REMOTE_USER@$REMOTE_HOST "find $REMOTE_PATH -name '*.log' 2>/dev/null | wc -l" 2>/dev/null)
    
    if [ "$AVAILABLE_LOGS" = "0" ] || [ -z "$AVAILABLE_LOGS" ]; then
        print_warning "No debug logs found"
        print_info "Expected locations: $REMOTE_PATH/outputs/*.log"
        return 1
    fi
    
    print_status "Found $AVAILABLE_LOGS log files"
    
    # Create local output directory
    mkdir -p "$LOCAL_PATH/outputs"
    
    # Sync all log files
    if rsync -avz --progress $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/outputs/*.log $LOCAL_PATH/outputs/ 2>/dev/null; then
        print_status "Debug logs synchronized"
        return 0
    else
        print_error "Failed to sync debug logs"
        return 1
    fi
}

collect_tmux_output() {
    print_info "Collecting tmux output..."
    
    # Capture tmux content
    if ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && tmux capture-pane -p -S- -t $TMUX_SESSION > ./output.txt" 2>/dev/null; then
        print_status "Tmux output captured"
    else
        print_warning "Failed to capture tmux (session may not exist)"
        return 1
    fi
    
    # Download tmux output
    mkdir -p "$LOCAL_PATH/outputs"
    if rsync -avz $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/output.txt $LOCAL_PATH/outputs/ 2>/dev/null; then
        print_status "Tmux output downloaded"
        return 0
    else
        print_error "Failed to download tmux output"
        return 1
    fi
}

analyze_logs() {
    print_info "Analyzing collected logs..."
    
    # Count files
    LOG_COUNT=$(ls $LOCAL_PATH/outputs/*.log 2>/dev/null | wc -l)
    TXT_COUNT=$(ls $LOCAL_PATH/outputs/*.txt 2>/dev/null | wc -l)
    
    echo ""
    echo "📊 COLLECTION SUMMARY:"
    echo "   Debug logs: $LOG_COUNT files"
    echo "   Text outputs: $TXT_COUNT files"
    echo "   Location: $LOCAL_PATH/outputs/"
    echo ""
    
    # Analyze by type
    if [ $LOG_COUNT -gt 0 ]; then
        echo "🔍 Debug Log Analysis:"
        for log in $LOCAL_PATH/outputs/*.log; do
            if [ -f "$log" ]; then
                LINES=$(wc -l < "$log")
                SIZE=$(ls -lh "$log" | awk '{print $5}')
                echo "   $(basename "$log"): $LINES lines ($SIZE)"
            fi
        done
        echo ""
        
        # Check for errors
        ERRORS=$(grep -i "error\|critical\|failed" $LOCAL_PATH/outputs/*.log 2>/dev/null | wc -l)
        if [ $ERRORS -gt 0 ]; then
            print_warning "$ERRORS errors found in logs"
        else
            print_status "No critical errors found"
        fi
    fi
    
    echo "💡 Useful commands:"
    echo "   grep 'pattern' $LOCAL_PATH/outputs/*.log"
    echo "   tail -f $LOCAL_PATH/outputs/*.log"
    echo "   less $LOCAL_PATH/outputs/*.log"
}

monitor_realtime() {
    print_info "Starting real-time monitoring..."
    print_info "Press Ctrl+C to stop"
    echo ""
    
    # Colorize function
    colorize() {
        while IFS= read -r line; do
            case "$line" in
                *"ERROR"*|*"CRITICAL"*|*"FAILED"*)
                    echo -e "${RED}$line${NC}"
                    ;;
                *"WARNING"*)
                    echo -e "${YELLOW}$line${NC}"
                    ;;
                *"MPC"*|*"CONTROL"*)
                    echo -e "${GREEN}$line${NC}"
                    ;;
                *"VISION"*|*"CAMERA"*)
                    echo -e "${BLUE}$line${NC}"
                    ;;
                *)
                    echo "$line"
                    ;;
            esac
        done
    }
    
    # Monitor logs
    ssh $REMOTE_USER@$REMOTE_HOST "
        cd $REMOTE_PATH/outputs 2>/dev/null || cd $REMOTE_PATH
        echo '📡 Monitoring logs from: \$(pwd)'
        echo '========================'
        tail -f *.log output.txt saida.txt 2>/dev/null
    " | colorize
}

cleanup_old_logs() {
    if [ "$QUIET" != "true" ]; then
        echo ""
        read -p "🧹 Clean old logs on remote (keep 3 newest)? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH/outputs 2>/dev/null && ls -t *.log 2>/dev/null | tail -n +4 | xargs rm -f" 2>/dev/null
            print_status "Old logs cleaned"
        fi
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Parse command line arguments
QUIET=false
VERBOSE=false
MODE="debug"

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -h|--host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -r|--remote-path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        -l|--local-path)
            LOCAL_PATH="$2"
            shift 2
            ;;
        -s|--session)
            TMUX_SESSION="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        debug|tmux|both|monitor)
            MODE="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
if [ "$QUIET" != "true" ]; then
    echo "🔧 Universal Remote Debug Collector"
    echo "=================================="
    echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo "Local:  $LOCAL_PATH/outputs"
    echo "Mode:   $MODE"
    echo ""
fi

# Check connection
if ! check_connection; then
    exit 1
fi

# Execute based on mode
case $MODE in
    debug)
        if collect_debug_logs; then
            analyze_logs
            cleanup_old_logs
        fi
        ;;
    tmux)
        if collect_tmux_output; then
            analyze_logs
            cleanup_old_logs
        fi
        ;;
    both)
        collect_debug_logs
        collect_tmux_output
        analyze_logs
        cleanup_old_logs
        ;;
    monitor)
        monitor_realtime
        ;;
    *)
        print_error "Invalid mode: $MODE"
        show_usage
        exit 1
        ;;
esac

if [ "$QUIET" != "true" ]; then
    print_status "✅ Operation completed!"
fi
