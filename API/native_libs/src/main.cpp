#include <cstdint>
#include <memory>

#include "LifeTimeManager.h"
#include "tensorflow/c/c_api.h"
#include "TypeLabel.h"
#include "Tensor.h"

using namespace std;

extern "C" {
    int32_t add1(int32_t x) {
        return x + 1;
    }

    int32_t *getPtr(int32_t n) {
        shared_ptr<int32_t> ptr = make_shared<int32_t>(n);
        return LifetimeManager::instance().addOwnership(std::move(ptr));
    }

    int32_t getValue(int32_t *ptr) {
        return *LifetimeManager::instance().accessOwned(ptr);
    }

    Tensor<TF_FLOAT> *tensorFloatFromArray(float *values, int64_t len) {
        shared_ptr<Tensor<TF_FLOAT>> ptr = make_shared<Tensor<TF_FLOAT>>(vector<typename Type<TF_FLOAT>::type>(values, values + len));
        return LifetimeManager::instance().addOwnership(std::move(ptr));
    }

    float getFloat1d(Tensor<TF_FLOAT> *tensor, int64_t index) {
        return LifetimeManager::instance().accessOwned(tensor)->at({index});
    }
}