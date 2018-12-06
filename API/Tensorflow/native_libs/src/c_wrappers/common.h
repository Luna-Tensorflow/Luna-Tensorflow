//
// Created by radeusgd on 06.12.18.
//

#ifndef TFL_COMMON_H
#define TFL_COMMON_H

// intellisense is checked because of MSVC bug: https://developercommunity.visualstudio.com/content/problem/335672/c-intellisense-stops-working-with-given-code.html
#if defined(_MSC_VER) && !defined(__INTELLISENSE__)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT [[gnu::visibility ("default")]]
#endif

#ifdef BUILDING_TFL_HELPER
#define TFL_API EXPORT
#else
#define TFL_API
#endif

#endif //TFL_COMMON_H
