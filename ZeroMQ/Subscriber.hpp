#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <zmq.hpp>

class Subscriber {
private:
  zmq::context_t context;
  zmq::socket_t subscriber;
  bool running;

public:
  Subscriber();
  ~Subscriber();

  zmq::socket_t &getSocket();

  void subscribe(const std::string &topic);
  // void listen();
  // void listenFrames();
  void connect(const std::string &address);
  // void reconnect(const std::string& address);
  void stop();
};

#endif // SUBSCRIBER_HPP
