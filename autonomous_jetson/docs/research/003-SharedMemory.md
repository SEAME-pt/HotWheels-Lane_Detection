# Shared Memory
## Introduction
Shared memory is a method of communication between multiple processes or threads in a computer system, where a specific region of memory is accessible by all participating entities. It allows for fast data exchange since the data does not need to be copied between processes. This approach is commonly used in parallel computing and multiprocessing environments to enhance performance and efficiency.  

Methods that can be used for this purpose:
- POSIX
- System V
- Boost.Interprocess

## Methods
### POSIX
POSIX shared memory is a standardized method for inter-process communication (IPC) on Unix-like systems. It uses file-like objects mapped into memory via shm_open() and mmap(), offering a simple and portable interface with modern permission and cleanup mechanisms.

### System V
System V shared memory is an older IPC mechanism that provides shared memory segments using shmget(), shmat(), and related functions. It offers fine-grained control but involves more complex management and lacks some of the conveniences of POSIX.

### Boost.Interprocess
Boost.Interprocess is a C++ library that abstracts shared memory and other IPC mechanisms using an object-oriented approach. It supports both POSIX and System V under the hood and provides high-level features like allocators and containers, making it ideal for C++ applications.

## Comparison
**Ease of use:** Boost.Interprocess > POSIX > System V  
  
**Portability:** POSIX and Boost are more portable than System V.  
  
**Control and flexibility:** System V offers more low-level control, while Boost provides high-level abstractions.  
  
**Modern support:** POSIX is more modern and widely recommended over System V in contemporary Unix systems.  

___