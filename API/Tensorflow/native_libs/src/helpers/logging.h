#ifndef TFL_LOGGING_H
#define TFL_LOGGING_H

#include <string>
#include <iostream>
#include <vector>

template<class T>
std::string vec_to_string(const std::vector<T>& vec)
{
    std::string ret = "[";
    for(auto& elem : vec)
        ret += std::to_string(elem) + ",";
    if(!vec.empty()) ret.pop_back();
    ret += "]";
    return ret;
}

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
void log_function_call(const char* prefix, const char* name, Args... args) {
    std::cerr << prefix << " " << name << "(";
    print_args(std::cerr, ", ", args...);
    std::cerr << ")\n";
}

template<typename R, typename ...Args>
void log_function_call_with_return(const char* prefix, const char* name, R ret, Args... args) {
    std::cerr << prefix << " " << name << "(";
    print_args(std::cerr, ", ", args...);
    std::cerr << ") = " << ret << "\n";
}

#ifdef VERBOSE
    #define LOG(...) do { print_args(std::cerr, " ", "C++", __VA_ARGS__, "\n"); } while(0)
#else
    #define LOG(...) do {} while (0)
#endif

#ifdef VERBOSE
    #define LOG_CALL(...) do { log_function_call("C++", __FUNCTION__, __VA_ARGS__); } while(0)
#else
    #define LOG_CALL(...) do {} while(0)
#endif

#ifdef VERBOSE
    #define LOG_PARAMLESS do { log_function_call("C++", __FUNCTION__); } while(0)
#else
    #define LOG_PARAMLESS do {} while(0)
#endif

#ifdef VERBOSE
    #define LOGANDRETURN(ret, ...) do { log_function_call_with_return("C++", __FUNCTION__, ret, __VA_ARGS__); return ret; } while(0)
#else
    #define LOGANDRETURN(ret, ...) do { return ret; } while(0)
#endif

#ifdef VERBOSEFFI
#define FFILOG(...) do { log_function_call("FFI", __FUNCTION__, __VA_ARGS__); } while(0)
#else
#define FFILOG(...) do {} while(0)
#endif

#ifdef VERBOSEFFI
#define FFILOG_PARAMLESS do { log_function_call("FFI", __FUNCTION__); } while(0)
#else
#define FFILOG_PARAMLESS do {} while(0)
#endif

#ifdef VERBOSEFFI
#define FFILOGANDRETURN(ret, ...) do { log_function_call_with_return("FFI", __FUNCTION__, ret, __VA_ARGS__); return ret; } while(0)
#else
#define FFILOGANDRETURN(ret, ...) do { return ret; } while(0)
#endif

#ifdef LOG_GRAPH_STRUCTURE
#define LOG_GRAPH(...) do { log_function_call("C++", __FUNCTION__, __VA_ARGS__); } while(0)
#else
#define LOG_GRAPH(...) do {} while(0)
#endif

#endif //TFL_LOGGING_H
