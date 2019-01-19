//Source: https://github.com/luna/Dataframes/blob/787b733241cd833d140e42effeeede0a2e81fa65/native_libs/src/LifetimeManager.h

#pragma once

#include <any>

#include <memory>
#include <mutex>
#include <sstream>
#include <vector>
#include <unordered_map>
#include "logging.h"

// TODO? could get rid of most headers by using pimpl

// Class is meant as a helper for managing std::shared_ptr lifetimes when they are shared
// with a foreign language through C API.
// Each time when shared_ptr is moved to foreign code it should be done through `addOwnership`
// When foreign code is done with the pointer, `releaseOwnership` should be called.
//
// Storage is thread-safe (internally synchronized with a lock).
//
// Technically there's nothing shared_ptr specific in storage (it uses type-erased any),
// if needed it can be adjusted to work with other kinds of types with similar semantics.
class LifetimeManager
{
    mutable std::mutex mx;
    std::unordered_multimap<const void *, std::any> storage; // address => shared_ptr<T>

    // Looks up the pointer and calls the given function with storage iterator (while having the storage lock).
    template<typename Function>
    auto access(const void *ptr, Function &&f) const
    {
    	LOG(ptr);
        std::unique_lock<std::mutex> lock{ mx };
        if(auto itr = storage.find(ptr); itr != storage.end())
        {
            return f(itr);
        }

        std::ostringstream out;
        out << "Cannot find storage for pointer " << ptr << " -- was it previously registered?";
        throw std::runtime_error(out.str());
    }

public:
    LifetimeManager();
    ~LifetimeManager();

    template<typename T>
    T *addOwnership(std::shared_ptr<T> ptr)
    {
        LOG(ptr);
        // we don't bother tracking nullptr - there is no object with a lifetime to manage
        if(!ptr)
            return nullptr;

        auto ret = ptr.get();
        std::unique_lock<std::mutex> lock{ mx };
        storage.emplace(ret, std::move(ptr));
        return ret;
    }
    void releaseOwnership(const void *ptr)
    {
        LOG(ptr);
        access(ptr, [this] (auto itr)
        {
            // TODO should separate retrieving any from storage and deleting it
            // deleting can take time and lock is not needed then
            storage.erase(itr);
        });
    }

    // NOTE: be careful, as this does not handle shared_ptr casting (type should exactly match)
    template<typename T>
    std::shared_ptr<T> accessOwned(const void *ptr) const
    {
        return access(ptr, [&] (auto itr)
        {
        	LOG("Accessing member of type ", itr->second.type().name());
            return std::any_cast<std::shared_ptr<T>>(itr->second);
        });
    }
    template<typename T>
    std::shared_ptr<T> accessOwned(const T *ptr) const
    {
        return accessOwned<T>(static_cast<const void*>(ptr));
    }

    template<typename T>
    std::vector<std::shared_ptr<T>> accessOwnedArray(const T **ptr, int32_t itemCount) const
    {
        std::vector<std::shared_ptr<T>> ret;
        ret.reserve(itemCount);
        for(int32_t i = 0; i < itemCount; i++)
        {
            auto managedItem = accessOwned(ptr[i]);
            ret.emplace_back(std::move(managedItem));
        }

        return ret;
    }


    // TODO reconsider at some stage more explicit global state
    static auto &instance()
    {
        static LifetimeManager manager;
        return manager;
    }
};