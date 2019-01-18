#include "../helpers/LifeTimeManager.h"
#include "memory.h"

void release(void *handle) noexcept
{
    // not logged as releaseOwnership logs
    //LifetimeManager::instance().releaseOwnership(handle);
}

void free_pointer(void *pointer) {
    free(pointer);
}