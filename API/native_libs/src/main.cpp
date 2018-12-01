#include <cstdint>
#include <memory>

#include "LifeTimeManager.h"

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


    // RESOURCE MANAGEMENT
    void release(void *handle) noexcept
    {
        LifetimeManager::instance().releaseOwnership(handle);
    }
}