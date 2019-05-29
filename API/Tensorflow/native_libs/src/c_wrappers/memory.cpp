#include "../helpers/LifeTimeManager.h"
#include "memory.h"
#include "../helpers/logging.h"
#include "../helpers/error.h"

void release(void *handle) noexcept
{
    TRANSLATE_EXCEPTION(nullptr) {
        (void)_func_;
        // not logged as releaseOwnership logs
        LifetimeManager::instance().releaseOwnership(handle);
    };
}

void free_pointer(void *pointer, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(pointer);
        free(pointer);
    };
}