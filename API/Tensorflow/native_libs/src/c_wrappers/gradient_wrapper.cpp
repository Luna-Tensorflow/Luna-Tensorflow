//
// Created by wojtek on 20.12.18.
//

#include <vector>
#include <memory>

#include "gradient_wrapper.h"
#include "../helpers/LifeTimeManager.h"
#include "../gradient/Gradient.h"

namespace {
    template <TF_DataType DataTypeLabel>
    Operation<DataTypeLabel>** add_gradients(Operation<DataTypeLabel>** ys, std::int64_t nys,  Operation<DataTypeLabel>** xs, std::int64_t nxs,
            Operation<DataTypeLabel>** dxs) {
        std::vector<std::shared_ptr<Operation<DataTypeLabel>>> ys_v;
        std::vector<std::shared_ptr<Operation<DataTypeLabel>>> xs_v;
        std::vector<std::shared_ptr<Operation<DataTypeLabel>>> dxs_v;

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

        std::vector<std::shared_ptr<Partial<DataTypeLabel>>> partials = Gradient<DataTypeLabel>::add_gradients(ys_v, xs_v, dxs_v);

        Operation<DataTypeLabel>** partial_ptrs = static_cast<Operation<DataTypeLabel>**>(std::malloc(sizeof(Operation<DataTypeLabel>*) * partials.size()));

        for(unsigned i = 0; i < partials.size(); ++i)
        {
            partial_ptrs[i] = LifetimeManager::instance().addOwnership(partials[i]);
        }

        return partial_ptrs;
    }
}

Operation<TF_FLOAT>** add_gradients_float(Operation<TF_FLOAT>** ys, std::int64_t nys,  Operation<TF_FLOAT>** xs, std::int64_t nxs,
                                         Operation<TF_FLOAT>** dxs) {
    return add_gradients(ys, nys, xs, nxs, dxs);
}