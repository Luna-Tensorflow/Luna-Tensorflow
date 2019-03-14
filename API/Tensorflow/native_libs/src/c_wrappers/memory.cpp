#include "../helpers/LifeTimeManager.h"
#include "memory.h"
#include "../helpers/logging.h"

void release(void *handle) noexcept
{
    // not logged as releaseOwnership logs
    LifetimeManager::instance().releaseOwnership(handle);
}

void free_pointer(void *pointer) {
    FFILOG(pointer);
    free(pointer);
}