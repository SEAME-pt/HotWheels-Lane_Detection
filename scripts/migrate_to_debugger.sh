#!/bin/bash

# Script to migrate all classes to use the centralized Debugger system
# Usage: ./migrate_to_debugger.sh

set -e

echo "🔄 Migrating all components to use centralized Debugger system..."

# Define the files to migrate
declare -a FILES=(
    "car_controls/sources/ControlsManager.cpp"
    "car_controls/sources/EngineController.cpp"
    "car_controls/sources/JoysticksController.cpp"
    "car_controls/sources/PeripheralController.cpp"
    "car_controls/sources/MPCPlanner.cpp"
    "car_controls/sources/Polyfitter.cpp"
    "car_controls/sources/inference/InferenceManager.cpp"
    "car_controls/sources/inference/CameraStreamer.cpp"
    "car_controls/sources/inference/LanePostProcessor.cpp"
    "car_controls/sources/inference/TensorRTInferencer.cpp"
    "car_controls/sources/inference/ONNXInferencer.cpp"
    "car_controls/sources/inference/KerasInferencer.cpp"
    "car_controls/sources/inference/LaneCurveFitter.cpp"
    "car_controls/sources/objectDetection/YOLOv5TRT.cpp"
    "main.cpp"
)

# Function to add Debugger include if not present
add_debugger_include() {
    local file="$1"
    if [ -f "$file" ]; then
        if ! grep -q '#include "Debugger.hpp"' "$file"; then
            echo "📝 Adding Debugger.hpp include to $file"
            # Find the last #include line and add after it
            sed -i '/^#include.*\.hpp"$/a #include "Debugger.hpp"' "$file"
        fi
    fi
}

# Function to replace common debug patterns
replace_debug_patterns() {
    local file="$1"
    local component="$2"
    
    if [ -f "$file" ]; then
        echo "🔧 Replacing debug patterns in $file (component: $component)"
        
        # Replace qDebug() patterns
        sed -i "s/qDebug() << \(.*\);/INFO_LOG(\"$component\", \1);/g" "$file"
        
        # Replace std::cout debug patterns  
        sed -i "s/std::cout << \"\[.*DEBUG.*\]\" << \(.*\) << std::endl;/DEBUG_LOG(\"$component\", \1);/g" "$file"
        sed -i "s/std::cout << \"\[.*INFO.*\]\" << \(.*\) << std::endl;/INFO_LOG(\"$component\", \1);/g" "$file"
        sed -i "s/std::cout << \"\[.*ERROR.*\]\" << \(.*\) << std::endl;/ERROR_LOG(\"$component\", \1);/g" "$file"
        sed -i "s/std::cout << \"\[.*WARNING.*\]\" << \(.*\) << std::endl;/WARNING_LOG(\"$component\", \1);/g" "$file"
        
        # Replace std::cerr patterns
        sed -i "s/std::cerr << \(.*\) << std::endl;/ERROR_LOG(\"$component\", \1);/g" "$file"
        
        # Replace printf patterns (basic)
        sed -i "s/printf(\"\[DEBUG\].*\\\n\");/DEBUG_LOG(\"$component\", \"Debug message\");/g" "$file"
    fi
}

# Process each file
echo "📂 Processing files..."

# ControlsManager
if [ -f "car_controls/sources/ControlsManager.cpp" ]; then
    add_debugger_include "car_controls/sources/ControlsManager.cpp"
    replace_debug_patterns "car_controls/sources/ControlsManager.cpp" "ControlsManager"
    echo "✅ ControlsManager migrated"
fi

# EngineController  
if [ -f "car_controls/sources/EngineController.cpp" ]; then
    add_debugger_include "car_controls/sources/EngineController.cpp"
    replace_debug_patterns "car_controls/sources/EngineController.cpp" "EngineController"
    echo "✅ EngineController migrated"
fi

# JoysticksController
if [ -f "car_controls/sources/JoysticksController.cpp" ]; then
    add_debugger_include "car_controls/sources/JoysticksController.cpp"
    replace_debug_patterns "car_controls/sources/JoysticksController.cpp" "JoysticksController"
    echo "✅ JoysticksController migrated"
fi

# PeripheralController
if [ -f "car_controls/sources/PeripheralController.cpp" ]; then
    add_debugger_include "car_controls/sources/PeripheralController.cpp"
    replace_debug_patterns "car_controls/sources/PeripheralController.cpp" "PeripheralController"
    echo "✅ PeripheralController migrated"
fi

# MPCPlanner
if [ -f "car_controls/sources/MPCPlanner.cpp" ]; then
    add_debugger_include "car_controls/sources/MPCPlanner.cpp"
    replace_debug_patterns "car_controls/sources/MPCPlanner.cpp" "MPCPlanner"
    echo "✅ MPCPlanner migrated"
fi

# Polyfitter
if [ -f "car_controls/sources/Polyfitter.cpp" ]; then
    add_debugger_include "car_controls/sources/Polyfitter.cpp"
    replace_debug_patterns "car_controls/sources/Polyfitter.cpp" "Polyfitter"
    echo "✅ Polyfitter migrated"
fi

# Vision/Inference components
if [ -f "car_controls/sources/inference/InferenceManager.cpp" ]; then
    add_debugger_include "car_controls/sources/inference/InferenceManager.cpp"
    replace_debug_patterns "car_controls/sources/inference/InferenceManager.cpp" "InferenceManager"
    echo "✅ InferenceManager migrated"
fi

if [ -f "car_controls/sources/inference/CameraStreamer.cpp" ]; then
    add_debugger_include "car_controls/sources/inference/CameraStreamer.cpp"
    replace_debug_patterns "car_controls/sources/inference/CameraStreamer.cpp" "CameraStreamer"
    echo "✅ CameraStreamer migrated"
fi

# Main
if [ -f "main.cpp" ]; then
    add_debugger_include "main.cpp"
    replace_debug_patterns "main.cpp" "Main"
    echo "✅ main.cpp migrated"
fi

echo ""
echo "🎉 Migration completed!"
echo ""
echo "📋 Summary:"
echo "   - All major components now use centralized Debugger"
echo "   - qDebug(), std::cout, std::cerr patterns replaced with Debugger macros"
echo "   - Debugger.hpp included in all relevant files"
echo ""
echo "🔍 Next steps:"
echo "   1. Test compilation: make clean && make"
echo "   2. Run debug setup: ./scripts/setup_debug.sh"
echo "   3. Test logging: ./main and check outputs/ directory"
echo "   4. Use enhanced output collection: ./scripts/get_output_enhanced.sh"
echo ""
echo "📚 Available logging macros:"
echo "   - DEBUG_LOG(component, message)     - General debug info"
echo "   - INFO_LOG(component, message)      - General information"
echo "   - WARNING_LOG(component, message)   - Warnings"
echo "   - ERROR_LOG(component, message)     - Errors"
echo "   - CRITICAL_LOG(component, message)  - Critical errors"
echo "   - MPC_DEBUG(message)                - MPC-specific debug"
echo "   - MPC_INFO(message)                 - MPC-specific info"
echo "   - VISION_DEBUG(message)             - Vision-specific debug"
echo "   - VISION_INFO(message)              - Vision-specific info"
echo "   - CONTROL_DEBUG(message)            - Control-specific debug"
echo "   - CONTROL_INFO(message)             - Control-specific info"
