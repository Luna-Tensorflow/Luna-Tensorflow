#include "tf/version.h"
#include <tensorflow/c/c_api.h>

const char* tf_version() {
  return TF_Version();
}
