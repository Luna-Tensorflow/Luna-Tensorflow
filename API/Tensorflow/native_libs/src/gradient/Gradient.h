//
// Created by wojtek on 19.12.18.
//

#ifndef TFL_GRADIENT_H
#define TFL_GRADIENT_H

#include <vector>
#include <memory>
#include <tensorflow/c/c_api.h>

#include "../graph/GraphSession.h"
#include "../helpers/utils.h"
#include "../ops/Partial.h"

template <TF_DataType DataTypeLabel>
class Gradient {
private:
    Gradient(std::vector<std::shared_ptr<Operation<DataTypeLabel>>> ys,
            std::vector<std::shared_ptr<Operation<DataTypeLabel>>> xs,
            std::vector<std::shared_ptr<Operation<DataTypeLabel>>> dxs) : ys(ys), xs(xs), dxs(dxs) {
        if (!dxs.empty() && dxs.size() != ys.size()) {
            throw std::invalid_argument("dxs must be empty or of the same size as ys!");
        }
        if (xs.empty() || ys.empty()) {
            throw std::invalid_argument("xs and ys must not be empty!");
        }

        std::shared_ptr<Gradient<DataTypeLabel>> ptr = std::shared_ptr<Gradient<DataTypeLabel>>(this);

        size_t hash_base = ys[0]->hashcode();
        for (auto it = ys.begin() + 1; it != ys.end(); ++it) {
            hash_base = hash_combine(hash_base, (*it)->hashcode());
        }
        for (auto &x: xs) {
            hash_base = hash_combine(hash_base, x->hashcode());
        }
        for (auto &dx: dxs) {
            hash_base = hash_combine(hash_base, dx->hashcode());
        }

        for (auto &x : xs) {
            partials.push_back(new Partial<DataTypeLabel>(x, ptr, hash_base));
            partial_hashes.push_back(partials.back()->hashcode());
        }
    }

public:
    static std::vector<std::shared_ptr<Partial<DataTypeLabel>>> add_gradients(std::vector<std::shared_ptr<Operation<DataTypeLabel>>> ys,
            std::vector<std::shared_ptr<Operation<DataTypeLabel>>> xs, std::vector<std::shared_ptr<Operation<DataTypeLabel>>> dxs) {
        Gradient<DataTypeLabel> *gradient = new Gradient<DataTypeLabel>(ys, xs, dxs);

        std::vector<std::shared_ptr<Partial<DataTypeLabel>>> ret;
        for (auto partial: gradient->partials) {
            ret.push_back(std::shared_ptr<Partial<DataTypeLabel>>(partial));
        }

        return ret;
    }

    void add_to_graph(GraphSession &graph) {
        std::vector<TF_Output> y_outputs;
        for (auto &y : ys) {
            y_outputs.push_back(graph.add_operation(y.get()));
        }
        std::vector<TF_Output> x_outputs;
        for (auto &x : xs) {
            x_outputs.push_back(graph.add_operation(x.get()));
        }
        std::vector<TF_Output> dx_outputs;
        for (auto &dx : dxs) {
            dx_outputs.push_back(graph.add_operation(dx.get()));
        }

        std::vector<TF_Output> results(xs.size());
        run_with_status<void>(std::bind(TF_AddGradients, graph.get_underlying(), y_outputs.data(), y_outputs.size(),
                x_outputs.data(), x_outputs.size(), dx_outputs.data(), std::placeholders::_1, results.data()));

        for (size_t i = 0; i < results.size(); ++i) {
            graph.register_output_hash(partial_hashes[i], results[i]);
        }
    }

private:
    std::vector<Partial<DataTypeLabel>*> partials;
    std::vector<size_t> partial_hashes;
    std::vector<std::shared_ptr<Operation<DataTypeLabel>>> ys;
    std::vector<std::shared_ptr<Operation<DataTypeLabel>>> xs;
    std::vector<std::shared_ptr<Operation<DataTypeLabel>>> dxs;
};

#endif //TFL_GRADIENT_H
