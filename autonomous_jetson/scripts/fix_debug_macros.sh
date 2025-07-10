#!/bin/bash

# Quick Fix Script for Debug Macro Compilation Errors
# Converts stream-style logging to string concatenation

echo "🔧 Fixing debug macro compilation errors..."

# Function to fix a file
fix_file() {
    local file="$1"
    echo "Fixing $file..."
    
    # Create backup
    cp "$file" "$file.backup"
    
    # Fix ERROR_LOG with concatenation
    sed -i 's/ERROR_LOG(\([^,]*\), \([^"]*\)"[^"]*" << \([^)]*\))/ERROR_LOG(\1, std::string(\2) + \3)/g' "$file"
    
    # Fix INFO_LOG with concatenation  
    sed -i 's/INFO_LOG(\([^,]*\), \([^"]*\)"[^"]*" << \([^)]*\))/INFO_LOG(\1, std::string(\2) + \3)/g' "$file"
    
    # Fix WARNING_LOG with concatenation
    sed -i 's/WARNING_LOG(\([^,]*\), \([^"]*\)"[^"]*" << \([^)]*\))/WARNING_LOG(\1, std::string(\2) + \3)/g' "$file"
    
    # Fix DEBUG_LOG with concatenation
    sed -i 's/DEBUG_LOG(\([^,]*\), \([^"]*\)"[^"]*" << \([^)]*\))/DEBUG_LOG(\1, std::string(\2) + \3)/g' "$file"
}

# Specific fixes for the current errors
echo "Applying specific fixes..."

# Fix JoysticksController.cpp
if [ -f "car_controls/sources/JoysticksController.cpp" ]; then
    sed -i 's/INFO_LOG("JoysticksController", "Failed to initialize SDL:" << SDL_GetError());/INFO_LOG("JoysticksController", std::string("Failed to initialize SDL: ") + SDL_GetError());/' car_controls/sources/JoysticksController.cpp
fi

# Fix CameraStreamer.cpp  
if [ -f "car_controls/sources/inference/CameraStreamer.cpp" ]; then
    sed -i 's/ERROR_LOG("CameraStreamer", "CUDA sync error in stop(): " << e.what());/ERROR_LOG("CameraStreamer", std::string("CUDA sync error in stop(): ") + e.what());/' car_controls/sources/inference/CameraStreamer.cpp
fi

# Fix ControlsManager.cpp - multiple fixes
if [ -f "car_controls/sources/ControlsManager.cpp" ]; then
    echo "Fixing ControlsManager.cpp..."
    
    # Fix all the ERROR_LOG calls
    sed -i 's/ERROR_LOG("ControlsManager", "Error: " << e.what());/ERROR_LOG("ControlsManager", std::string("Error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[Subscriber\] ZMQ error: " << e.what());/ERROR_LOG("ControlsManager", std::string("[Subscriber] ZMQ error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[~ControlsManager\] Error stopping motors: " << e.what());/ERROR_LOG("ControlsManager", std::string("[~ControlsManager] Error stopping motors: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "Autonomous control loop #" << control_counter << "- Using cached data");/INFO_LOG("ControlsManager", std::string("Autonomous control loop #") + std::to_string(control_counter) + "- Using cached data");/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "Autonomous control error: " << e.what());/ERROR_LOG("ControlsManager", std::string("Autonomous control error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "Autonomous control error:" << e.what());/INFO_LOG("ControlsManager", std::string("Autonomous control error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[getWaypointsFromVision\] ZMQ error: " << e.what());/ERROR_LOG("ControlsManager", std::string("[getWaypointsFromVision] ZMQ error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[getLaneInfoFromVision\] ZMQ error: " << e.what());/ERROR_LOG("ControlsManager", std::string("[getLaneInfoFromVision] ZMQ error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[checkEmergencyObstacles\] ZMQ error: " << e.what());/ERROR_LOG("ControlsManager", std::string("[checkEmergencyObstacles] ZMQ error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[showVisionDebug\] ZMQ error: " << e.what());/ERROR_LOG("ControlsManager", std::string("[showVisionDebug] ZMQ error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "Vision data update error: " << e.what());/ERROR_LOG("ControlsManager", std::string("Vision data update error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "Obstacle data update error: " << e.what());/ERROR_LOG("ControlsManager", std::string("Obstacle data update error: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "Real sensors" << (enable ? "enabled" : "disabled"));/INFO_LOG("ControlsManager", std::string("Real sensors ") + (enable ? "enabled" : "disabled"));/' car_controls/sources/ControlsManager.cpp
    
    # Fix complex log statements with multiple << operators
    sed -i 's/INFO_LOG("ControlsManager", "  Position: (" << state.x << ", " << state.y << ")");/INFO_LOG("ControlsManager", std::string("  Position: (") + std::to_string(state.x) + ", " + std::to_string(state.y) + ")");/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "  Velocity: " << state.velocity << " m\/s");/INFO_LOG("ControlsManager", std::string("  Velocity: ") + std::to_string(state.velocity) + " m\/s");/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "  Yaw: " << state.yaw \* 180.0 \/ M_PI << " degrees");/INFO_LOG("ControlsManager", std::string("  Yaw: ") + std::to_string(state.yaw * 180.0 \/ M_PI) + " degrees");/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "  Real sensors: " << (m_stateEstimator.m_useRealSensors.load() ? "ON" : "OFF"));/INFO_LOG("ControlsManager", std::string("  Real sensors: ") + (m_stateEstimator.m_useRealSensors.load() ? "ON" : "OFF"));/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "  Applied throttle: " << m_lastThrottle.load());/INFO_LOG("ControlsManager", std::string("  Applied throttle: ") + std::to_string(m_lastThrottle.load()));/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/INFO_LOG("ControlsManager", "  Applied steering: " << m_lastSteering.load() \* 180.0 \/ M_PI << " degrees");/INFO_LOG("ControlsManager", std::string("  Applied steering: ") + std::to_string(m_lastSteering.load() * 180.0 \/ M_PI) + " degrees");/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[EMERGENCY\] Error in primary stop: " << e.what());/ERROR_LOG("ControlsManager", std::string("[EMERGENCY] Error in primary stop: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[EMERGENCY\] Error in primary emergency stop: " << e.what());/ERROR_LOG("ControlsManager", std::string("[EMERGENCY] Error in primary emergency stop: ") + e.what());/' car_controls/sources/ControlsManager.cpp
    
    sed -i 's/ERROR_LOG("ControlsManager", "\[EMERGENCY\] Error in forced stop: " << e2.what());/ERROR_LOG("ControlsManager", std::string("[EMERGENCY] Error in forced stop: ") + e2.what());/' car_controls/sources/ControlsManager.cpp
fi

# Fix Polyfitter.cpp
if [ -f "car_controls/sources/Polyfitter.cpp" ]; then
    sed -i 's/ERROR_LOG("Polyfitter", "Folder does not exist: " << folderPath);/ERROR_LOG("Polyfitter", std::string("Folder does not exist: ") + folderPath);/' car_controls/sources/Polyfitter.cpp
fi

echo "✅ Quick fixes applied!"
echo ""
echo "💡 To prevent future issues, use these patterns:"
echo '   ERROR_LOG("Component", std::string("Message: ") + variable);'
echo '   INFO_LOG("Component", std::string("Value: ") + std::to_string(number));'
echo ""
echo "🔨 Try compiling again: make clean && make"
