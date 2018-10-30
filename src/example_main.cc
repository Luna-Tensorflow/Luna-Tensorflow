#include <iostream>
#include <c_api.h>

int main() {
  std::cout << "Tensorflow version = " << TF_Version() << std::endl;
}
