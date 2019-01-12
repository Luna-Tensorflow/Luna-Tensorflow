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
        LOG(ys, nys, xs, nxs, dxs);
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

#define DECLARE_GRADIENT(typelabel) \
Operation<typelabel>** add_gradients_##typelabel(Operation<typelabel>** ys, std::int64_t nys,  Operation<typelabel>** xs, \
                                         std::int64_t nxs, Operation<typelabel>** dxs) { \
    return add_gradients(ys, nys, xs, nxs, dxs); \
}

DECLARE_GRADIENT(TF_FLOAT);
DECLARE_GRADIENT(TF_DOUBLE);
DECLARE_GRADIENT(TF_INT8);
DECLARE_GRADIENT(TF_INT16);
DECLARE_GRADIENT(TF_INT32);
DECLARE_GRADIENT(TF_INT64);
DECLARE_GRADIENT(TF_UINT8);
DECLARE_GRADIENT(TF_UINT16);
DECLARE_GRADIENT(TF_UINT32);
DECLARE_GRADIENT(TF_UINT64);
DECLARE_GRADIENT(TF_BOOL);
//DECLARE_GRADIENT(TF_STRING);
//DECLARE_GRADIENT(TF_HALF);