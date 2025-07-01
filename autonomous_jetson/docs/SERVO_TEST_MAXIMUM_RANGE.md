# Servo Test with Maximum Range Implementation

## Problem Identified

The servo initialization test was using only a small portion of the servo's physical range due to the calculation method:
- Test angle: ±45° 
- Max angle constant: 180°
- Resulting usage: 45/180 = 25% of available range
- This made the movement barely visible during testing

## Solution Implemented

Enhanced the `testServoInitialization()` function with a **two-phase test approach**:

### Phase 1: Calculated PWM Values (Original Logic)
- Tests the normal angle-to-PWM calculation logic
- Uses ±45° angles with the existing formula
- Shows percentage of servo range being used
- Demonstrates the calculation accuracy

### Phase 2: Direct Extreme PWM Values (Maximum Movement)
- Uses the direct hardware PWM limits for maximum visible movement
- SERVO_LEFT_PWM = 240 (direct value)
- SERVO_RIGHT_PWM = 470 (direct value)
- Guarantees the servo moves to its absolute physical limits

## PWM Constants Used

```cpp
const int SERVO_CENTER_PWM = 340;
const int SERVO_LEFT_PWM = 240;    // 340 - 100
const int SERVO_RIGHT_PWM = 470;   // 340 + 130
const int MAX_ANGLE = 180;
```

## Test Sequence

### Phase 1 (Calculated):
1. Move to LEFT (-45°): PWM ≈ 315 (25% of left range)
2. Move to RIGHT (+45°): PWM ≈ 373 (25% of right range)
3. Return to CENTER: PWM = 340

### Phase 2 (Direct Limits):
1. Move to ABSOLUTE LEFT: PWM = 240 (100% left range)
2. Move to ABSOLUTE RIGHT: PWM = 470 (100% right range)  
3. Return to CENTER: PWM = 340

## Timing Improvements

- Increased wait times for better visibility:
  - Phase 1: 2000ms per position
  - Phase 2: 2500ms per position (maximum visibility)
  - Center return: 1000ms

## Expected Results

### Phase 1 Output:
```
[SERVO TEST] Phase 1 servo range usage - Left: 25%, Right: 25%
```

### Phase 2 Output:
- **Maximum visible servo movement** 
- Full left-to-right sweep using entire hardware range
- Clear visual confirmation that commands are being transmitted

## Benefits

1. **Comprehensive Testing**: Tests both calculation logic and hardware limits
2. **Maximum Visibility**: Phase 2 ensures maximum movement for visual confirmation
3. **Diagnostic Information**: Shows actual PWM values and range usage percentages
4. **Hardware Verification**: Confirms the servo can reach its physical limits
5. **MPC Calibration**: Helps verify that MPC commands will be properly transmitted

## Next Steps

If Phase 1 shows correct calculations but Phase 2 doesn't show maximum movement, this indicates:
- Hardware/wiring issues with the servo
- PWM driver problems
- Physical mechanical constraints

The two-phase approach helps isolate whether issues are:
- **Software**: Calculation problems (Phase 1)
- **Hardware**: Physical servo/driver problems (Phase 2)
