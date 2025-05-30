# ZeroC Ice middleware
## Introduction
### Base concepts
***Middleware*** is software that acts as a bridge between different applications, services, or systems, enabling them to communicate and exchange data. It provides common services such as messaging, authentication, data management, and communication protocols, simplifying the development of distributed systems. Middleware is essential in modern computing environments, helping to integrate diverse components in areas like cloud computing, enterprise applications, and real-time systems.  
  
  
### What is ZeroC?
ZeroC is a software company best known for developing Ice (Internet Communications Engine), a high-performance RPC (Remote Procedure Call) framework for building distributed applications. Ice supports multiple programming languages (like C++, Java, Python) and provides features such as object-oriented remote communication, automatic code generation, security, and efficient serialization. It is designed as a modern alternative to older technologies like CORBA, offering a flexible and scalable solution for complex networked systems.  

## Key features
- **Cross-platform**  
Ice supports many platforms, including Windows, Linux, macOS, and various embedded systems. It provides a consistent programming model regardless of the underlying operating system.

- **Object-Oriented Communication**  
Ice is designed around the concept of remote objects, and it uses an object-oriented model for communication. You define interfaces (services or operations) in the Ice IDL (Interface Definition Language), and Ice automatically handles the details of remote communication.

- **Efficient and Scalable**  
Ice is designed to be both fast and scalable. It uses highly optimized protocols to minimize latency and maximize throughput, and it's capable of handling high volumes of requests.

- **Supports Multiple Languages**  
Ice supports various programming languages, such as C++, Java, Python, and more. This makes it easy to integrate systems built with different technologies.

- **Transparency**  
Ice abstracts away the complexities of network communication. Developers can call remote objects in the same way they would call local objects, without needing to worry about the underlying network protocols or data serialization.

- **Security**  
Ice offers security features like authentication, encryption, and integrity checking to ensure that communication between components is secure.

- **Object Activation and Persistence**  
Ice supports object activation (ensuring objects are created on demand) and persistence (allowing objects to be saved and restored across different runs of the application).

## Application
**Single Ice server, Multiple clients**  
- The two applications (clients) connect to the same Ice server  
- The server maintains an object in RAM with a shared variable  
- When one app updates the variable, the change is visible to the other app  

___