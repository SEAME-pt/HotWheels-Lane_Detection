# Android Automotive Operating System (AAOS)
## Introduction
### Base concepts
***Infotainment*** systems are integrated multimedia displays and control units that provide information, entertainment, and connectivity. They typically include features like radio, media playback, navigation, Bluetooth connectivity, and smartphone integration, all accessible through a touchscreen interface.  
  
***In-Vehicle Infotainment (IVI)*** system apps are software applications designed to run within the car's infotainment system. They enhance the driving experience by providing entertainment, information, and connectivity and often focus on features like navigation, media streaming, and smartphone integration.

### What is AAOS?
Android Automotive Operating System is a base Android platform that runs pre-installed IVI system Android applications as well as optional second and third-party Android Applications offering openness, customization, and scale to automotive infotainment systems and head units. Android Automotive is not a fork or parallel development of Android. It is the same codebase and lives in the same repository as the Android shipped on phones, tablets, etc. It builds on a robust platform and feature set developed over 10+ years, enabling it to leverage the existing security model, compatibility program, developer tools, and infrastructure while continuing to be highly customizable and portable, completely free, and open source.  
  
***!!! All apps must follow Google’s guidelines for safety and driver distraction !!!***
  
## System architecture
Android Automotive OS is a Linux-based embedded operating system with a structure like this:
  
![image](https://github.com/user-attachments/assets/009f32f8-9063-4a7e-bb13-d735360237af)  
  
## Automotive HALs (Hardware Abstraction Layers)
- **vehicle_hal:** Interface for vehicle properties like speed, RPM, gear, fuel, etc.  
  
- **hvac_hal:** Controls for AC, heating, and ventilation  
  
- **audio_hal:** Car-specific audio routing  
  
- **power_hal:** Manage sleep/wake, ignition state  
  
## Android Automotive vs Android Auto
The nomenclature can be confusing but they're different concepts.  
  
***Android Auto*** is a platform running on the user’s phone, projecting the Android Auto user experience to a compatible in-vehicle infotainment system over a USB connection, supporting apps designed for in-vehicle use.  
  
***Android Automotive*** is an operating system and platform running directly on the in-vehicle hardware. It is a full-stack, open source, highly customizable platform powering the infotainment experience that supports apps built for Android as well as those built for Android Auto.


___