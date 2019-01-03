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
    Operation<DataTypeLabel>** add_gradient(Operation<DataTypeLabel>** ys, int nys,  Operation<DataTypeLabel>** xs, int nxs,
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

        for (int i = 0; i < nys; ++i) {
            dxs_v.push_back(LifetimeManager::instance().accessOwned(dxs[i]));
        }

        Gradient<DataTypeLabel> gradient(ys_v, xs_v, dxs_v);
    }
}