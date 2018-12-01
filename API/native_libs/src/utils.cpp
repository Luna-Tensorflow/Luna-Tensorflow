//
// Created by radeusgd on 01.12.18.
//
#include "utils.h"

size_t hash_combine(size_t x, size_t y) {
    x ^= y + 0x9e3779b9 + (x << 6u) + (x >> 2u);
    return x;
}