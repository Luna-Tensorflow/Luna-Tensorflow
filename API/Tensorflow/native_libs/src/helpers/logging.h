//
// Created by radeusgd on 06.12.18.
//

#ifndef TFL_LOGGING_H
#define TFL_LOGGING_H

#include <string>
#include <iostream>

inline void print_args(std::ostream& stream, std::string delimiter) {
    (void)stream;
    (void)delimiter;
}

template<typename T>
void print_args(std::ostream& stream, std::string delimiter, T head) {
    (void)delimiter;
    stream << head;
}

template<typename T, typename ...Args>
void print_args(std::ostream& stream, std::string delimiter, T head, Args... tail) {
    stream << head << delimiter;
    print_args(stream, delimiter, tail...);
}

template<typename ...Args>
void log_function_call(const char* name, Args... args) {
    std::cerr << "C++ " << name << "(";
    print_args(std::cerr, ", ", args...);
    std::cerr << ")\n";
}

template<typename R, typename ...Args>
void log_function_call_with_return(const char* name, R ret, Args... args) {
    std::cerr << "C++ " << name << "(";
    print_args(std::cerr, ", ", args...);
    std::cerr << ") = " << ret << "\n";
}

#ifdef VERBOSE
    #define LOG(...) do { log_function_call(__FUNCTION__, __VA_ARGS__); } while(0)
#else
    #define LOG(...) do {} while(0)
#endif

#ifdef VERBOSE
    #define LOGANDRETURN(ret, ...) do { log_function_call_with_return(__FUNCTION__, ret, __VA_ARGS__); return ret; } while(0)
#else
    #define LOGANDRETURN(ret, ...) do { return ret; } while(0)
#endif

#endif //TFL_LOGGING_H
