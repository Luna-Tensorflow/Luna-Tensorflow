#include "../LifeTimeManager.h"
#include "memory.h"

void release(void *handle) noexcept
{
    LifetimeManager::instance().releaseOwnership(handle);
}