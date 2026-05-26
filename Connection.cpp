#include "Connection.h"

Connection::Connection()
  : from(nullptr), to(nullptr),
  weight(0.0), gradient(0.0),
  trainable(true),
  filter(nullptr),
  filterSlot(-1)
{}
