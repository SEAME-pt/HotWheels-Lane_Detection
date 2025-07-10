# ZeroMQ middleware
## Introduction
ZeroMQ (ØMQ) is a high-performance asynchronous messaging library aimed at use in scalable, distributed, or concurrent applications. It provides a message queue, but unlike traditional message brokers, ZeroMQ runs without a dedicated message server, making it lightweight and fast.  
  
## How it works
ZeroMQ acts as a concurrency framework, sitting between raw sockets and higher-level messaging protocols. It abstracts away the complexity of socket programming and supports multiple messaging patterns like:
  
- Request/Reply
  
- Publish/Subscribe
  
- Push/Pull (Pipeline)
  
- Dealer/Router (Advanced Async Messaging)
  
It's a very good option for a middleware implementation because it communicates using messages (while handling message queuing), connection handling, and automatic reconnection behind the scenes.  
  
## Key Features
    
***Asynchronous Messaging:*** Enables non-blocking communication.
  
***Multiple Messaging Patterns:*** Built-in support for common communication paradigms.
  
***No Broker Needed:*** Peer-to-peer communication reduces latency and single points of failure.
  
***Cross-Platform:*** Works on major operating systems with bindings for many languages (C++, Python, Java, etc.).
  
***High Throughput and Low Latency:*** Designed for performance-critical applications.
  
***Message Queuing:*** Buffers messages during temporary disconnections.

___