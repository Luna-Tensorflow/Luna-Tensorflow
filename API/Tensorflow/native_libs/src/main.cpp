#include <cstdint>
#include <memory>
#include <API/Tensorflow/native_libs/src/helpers/logging.h>
#include <API/Tensorflow/native_libs/src/c_wrappers/common.h>

#include "helpers/LifeTimeManager.h"
#include "tensorflow/c/c_api.h"

using namespace std;
// TODO these are for debugging only, they should be removed
extern "C" {
    TFL_API int32_t add1(int32_t x) {
        LOG(x);
        return x + 1;
    }

    TFL_API int32_t *getPtr(int32_t n) {
        shared_ptr<int32_t> ptr = make_shared<int32_t>(n);
        return LifetimeManager::instance().addOwnership(std::move(ptr));
    }

    TFL_API int32_t getValue(int32_t *ptr) {
        return *LifetimeManager::instance().accessOwned(ptr);
    }
}