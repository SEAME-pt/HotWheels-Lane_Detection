#!/bin/bash

# Universal Debug Setup & Maintenance
# Configurable script for setting up debug environment on any embedded project
# Handles debug directories, environment setup, and system maintenance

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_NAME="${PROJECT_NAME:-embedded-project}"
DEBUG_DIR="${DEBUG_DIR:-outputs}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-7}"
MAX_LOG_SIZE_MB="${MAX_LOG_SIZE_MB:-100}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() { echo -e "${CYAN}=== $1 ===${NC}"; }
print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Universal Debug Setup & Maintenance Tool"
    echo ""
    echo "COMMANDS:"
    echo "  setup     - Initialize debug environment (default)"
    echo "  clean     - Clean old logs and temporary files"
    echo "  monitor   - Setup real-time monitoring"
    echo "  status    - Show debug system status"
    echo "  config    - Configure debug settings"
    echo ""
    echo "OPTIONS:"
    echo "  -p NAME   - Project name (default: $PROJECT_NAME)"
    echo "  -d DIR    - Debug directory (default: $DEBUG_DIR)"
    echo "  -r DAYS   - Log retention days (default: $LOG_RETENTION_DAYS)"
    echo "  -s MB     - Max log size in MB (default: $MAX_LOG_SIZE_MB)"
    echo "  -q        - Quiet mode"
    echo "  --help    - Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                          # Basic setup"
    echo "  $0 -p MyProject setup       # Setup with custom name"
    echo "  $0 clean                    # Clean old logs"
    echo "  $0 -r 3 clean              # Clean logs older than 3 days"
}

setup_debug_environment() {
    print_header "SETTING UP DEBUG ENVIRONMENT"
    
    # Create debug directories
    print_info "Creating debug directories..."
    mkdir -p "$DEBUG_DIR"
    mkdir -p "$DEBUG_DIR/archive"
    mkdir -p "$DEBUG_DIR/reports"
    
    if [ -d "$DEBUG_DIR" ]; then
        print_status "Debug directories created: $DEBUG_DIR"
    else
        print_error "Failed to create debug directories"
        return 1
    fi
    
    # Create debug configuration
    create_debug_config
    
    # Create monitoring script
    create_monitor_script
    
    # Set permissions
    chmod +x "$DEBUG_DIR"/*.sh 2>/dev/null
    
    print_status "Debug environment setup complete"
}

create_debug_config() {
    print_info "Creating debug configuration..."
    
    cat > "$DEBUG_DIR/debug_config.env" << EOF
# Debug Configuration for $PROJECT_NAME
# Generated on $(date)

# Project settings
PROJECT_NAME="$PROJECT_NAME"
DEBUG_DIR="$DEBUG_DIR"

# Log settings
LOG_RETENTION_DAYS=$LOG_RETENTION_DAYS
MAX_LOG_SIZE_MB=$MAX_LOG_SIZE_MB
LOG_LEVEL="DEBUG"

# File patterns
MPC_LOG_PATTERN="mpc_*.log"
VISION_LOG_PATTERN="vision_*.log"
CONTROL_LOG_PATTERN="control_*.log"
DEBUG_LOG_PATTERN="debug_*.log"

# Monitoring settings
MONITOR_INTERVAL=2
COLORIZE_OUTPUT=true

# Archive settings
AUTO_ARCHIVE=true
ARCHIVE_THRESHOLD_MB=50
EOF
    
    print_status "Configuration created: $DEBUG_DIR/debug_config.env"
}

create_monitor_script() {
    print_info "Creating monitoring script..."
    
    cat > "$DEBUG_DIR/monitor.sh" << 'EOF'
#!/bin/bash
# Auto-generated monitoring script

# Load configuration
if [ -f "debug_config.env" ]; then
    source debug_config.env
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

colorize() {
    if [ "$COLORIZE_OUTPUT" = "true" ]; then
        while IFS= read -r line; do
            case "$line" in
                *"ERROR"*|*"CRITICAL"*)
                    echo -e "${RED}$line${NC}"
                    ;;
                *"WARNING"*)
                    echo -e "${YELLOW}$line${NC}"
                    ;;
                *"MPC"*|*"CONTROL"*)
                    echo -e "${GREEN}$line${NC}"
                    ;;
                *"VISION"*)
                    echo -e "${BLUE}$line${NC}"
                    ;;
                *)
                    echo "$line"
                    ;;
            esac
        done
    else
        cat
    fi
}

echo "🔍 Monitoring logs for $PROJECT_NAME"
echo "Press Ctrl+C to stop"
echo ""

tail -f *.log 2>/dev/null | colorize
EOF
    
    chmod +x "$DEBUG_DIR/monitor.sh"
    print_status "Monitor script created: $DEBUG_DIR/monitor.sh"
}

clean_debug_logs() {
    print_header "CLEANING DEBUG LOGS"
    
    if [ ! -d "$DEBUG_DIR" ]; then
        print_warning "Debug directory not found: $DEBUG_DIR"
        return 1
    fi
    
    cd "$DEBUG_DIR"
    
    # Count current files
    LOG_COUNT=$(ls *.log 2>/dev/null | wc -l)
    print_info "Found $LOG_COUNT log files"
    
    if [ $LOG_COUNT -eq 0 ]; then
        print_info "No log files to clean"
        return 0
    fi
    
    # Archive large logs
    print_info "Archiving large logs (>${MAX_LOG_SIZE_MB}MB)..."
    ARCHIVED=0
    for log in *.log 2>/dev/null; do
        if [ -f "$log" ]; then
            SIZE_MB=$(du -m "$log" | cut -f1)
            if [ $SIZE_MB -gt $MAX_LOG_SIZE_MB ]; then
                gzip "$log"
                mv "$log.gz" archive/ 2>/dev/null
                ARCHIVED=$((ARCHIVED + 1))
            fi
        fi
    done
    
    if [ $ARCHIVED -gt 0 ]; then
        print_status "$ARCHIVED large logs archived"
    fi
    
    # Remove old logs
    print_info "Removing logs older than $LOG_RETENTION_DAYS days..."
    REMOVED=$(find . -name "*.log" -mtime +$LOG_RETENTION_DAYS -delete -print | wc -l)
    
    if [ $REMOVED -gt 0 ]; then
        print_status "$REMOVED old logs removed"
    else
        print_info "No old logs to remove"
    fi
    
    # Clean empty files
    find . -name "*.log" -size 0 -delete 2>/dev/null
    
    cd - >/dev/null
    print_status "Debug cleanup complete"
}

show_debug_status() {
    print_header "DEBUG SYSTEM STATUS"
    
    if [ ! -d "$DEBUG_DIR" ]; then
        print_error "Debug directory not found: $DEBUG_DIR"
        return 1
    fi
    
    cd "$DEBUG_DIR"
    
    # File counts
    LOG_COUNT=$(ls *.log 2>/dev/null | wc -l)
    ARCHIVE_COUNT=$(ls archive/*.gz 2>/dev/null | wc -l)
    
    echo "📊 File Statistics:"
    echo "   Active logs: $LOG_COUNT"
    echo "   Archived logs: $ARCHIVE_COUNT"
    echo "   Directory: $(pwd)"
    echo ""
    
    # Disk usage
    if [ $LOG_COUNT -gt 0 ]; then
        echo "💾 Disk Usage:"
        du -sh *.log 2>/dev/null | head -10
        echo ""
        
        # Recent activity
        echo "🕒 Recent Activity:"
        ls -lt *.log 2>/dev/null | head -5
        echo ""
        
        # Error summary
        ERROR_COUNT=$(grep -i "error\|critical" *.log 2>/dev/null | wc -l)
        WARNING_COUNT=$(grep -i "warning" *.log 2>/dev/null | wc -l)
        
        echo "⚠️  Issue Summary:"
        echo "   Errors: $ERROR_COUNT"
        echo "   Warnings: $WARNING_COUNT"
        
        if [ $ERROR_COUNT -gt 0 ]; then
            echo ""
            echo "🔍 Recent Errors:"
            grep -i "error\|critical" *.log 2>/dev/null | tail -3
        fi
    fi
    
    cd - >/dev/null
}

configure_debug() {
    print_header "DEBUG CONFIGURATION"
    
    if [ ! -f "$DEBUG_DIR/debug_config.env" ]; then
        print_warning "Configuration file not found, creating default..."
        create_debug_config
    fi
    
    echo "📝 Current Configuration:"
    cat "$DEBUG_DIR/debug_config.env" | grep -v "^#" | grep -v "^$"
    echo ""
    
    read -p "Edit configuration? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} "$DEBUG_DIR/debug_config.env"
        print_status "Configuration updated"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Parse arguments
COMMAND="setup"
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -d|--debug-dir)
            DEBUG_DIR="$2"
            shift 2
            ;;
        -r|--retention)
            LOG_RETENTION_DAYS="$2"
            shift 2
            ;;
        -s|--size)
            MAX_LOG_SIZE_MB="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        setup|clean|monitor|status|config)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute command
if [ "$QUIET" != "true" ]; then
    echo "🔧 Universal Debug Setup & Maintenance"
    echo "======================================"
    echo "Project: $PROJECT_NAME"
    echo "Debug Dir: $DEBUG_DIR"
    echo "Command: $COMMAND"
    echo ""
fi

case $COMMAND in
    setup)
        setup_debug_environment
        ;;
    clean)
        clean_debug_logs
        ;;
    monitor)
        if [ -f "$DEBUG_DIR/monitor.sh" ]; then
            exec "$DEBUG_DIR/monitor.sh"
        else
            print_error "Monitor script not found. Run 'setup' first."
            exit 1
        fi
        ;;
    status)
        show_debug_status
        ;;
    config)
        configure_debug
        ;;
    *)
        print_error "Invalid command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

if [ "$QUIET" != "true" ]; then
    print_status "✅ Operation completed!"
fi
