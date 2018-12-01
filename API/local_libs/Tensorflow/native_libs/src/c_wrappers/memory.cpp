#include <API/native_libs/src/LifeTimeManager.h>
#include "memory.h"

void release(void *handle) noexcept
{
    LifetimeManager::instance().releaseOwnership(handle);
}