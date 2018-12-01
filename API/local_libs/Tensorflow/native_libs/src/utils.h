#ifndef FFITESTHELPER_UTILS_H
#define FFITESTHELPER_UTILS_H

#include <cstddef>
#include <stdexcept>

/*
 * A placeholder type used to mark places where other classes are not yet available.
 */
class TODOType {};

template <typename T> T not_implemented() {
    throw std::logic_error("Not implemented");
}
/*
 * Based on https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
 */
size_t hash_combine(size_t x, size_t y);

#endif //FFITESTHELPER_UTILS_H
