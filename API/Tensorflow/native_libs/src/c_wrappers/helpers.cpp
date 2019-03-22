//
// Created by radeusgd on 05.03.19.
//
#include <string>
#include <unordered_map>
#include "../helpers/logging.h"
#include "helpers.h"
#include "../helpers/error.h"

int64_t get_and_increase_layer_counter(const char* name, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(name);
        static std::unordered_map<std::string, int> id_map;
        return id_map[name]++; // we use the fact that non-existent entries are automatically initialized to 0
    };
}
