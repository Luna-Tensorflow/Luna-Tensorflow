//
// Created by wojtek on 20.12.18.
//

#include <vector>
#include <memory>

#include "gradient_wrapper.h"
#include "../helpers/LifeTimeManager.h"
#include "../ops/Gradient.h"

Output** add_gradients(Output** ys, std::int64_t nys,  Output** xs, std::int64_t nxs, Output** dxs) {
    FFILOG(ys, nys, xs, nxs, dxs);
    std::vector<std::shared_ptr<Output>> ys_v;
    std::vector<std::shared_ptr<Output>> xs_v;
    std::vector<std::shared_ptr<Output>> dxs_v;

    for (int i = 0; i < nys; ++i) {
        ys_v.push_back(LifetimeManager::instance().accessOwned(ys[i]));
    }

    for (int i = 0; i < nxs; ++i) {
        xs_v.push_back(LifetimeManager::instance().accessOwned(xs[i]));
    }

    if (dxs != nullptr) {
        for (int i = 0; i < nys; ++i) {
            dxs_v.push_back(LifetimeManager::instance().accessOwned(dxs[i]));
        }
    }

    std::vector<std::shared_ptr<Output>> outputs = Gradient::add_gradients(ys_v, xs_v, dxs_v);

    auto** output_ptrs = static_cast<Output**>(std::malloc(sizeof(Output*) * outputs.size()));

    for(unsigned i = 0; i < outputs.size(); ++i)
    {
        output_ptrs[i] = LifetimeManager::instance().addOwnership(
            std::dynamic_pointer_cast<Output>(outputs[i]));
    }

    return output_ptrs;
}