//
// Created by wojtek on 14.02.19.
//

#include "Gradient.h"
#include <utility>

Gradient::Gradient(std::vector<std::shared_ptr<Output>> ys, std::vector<std::shared_ptr<Output>> xs,
        std::vector<std::shared_ptr<Output>> dxs) : ys(ys), xs(xs), dxs(dxs) {
    if (!dxs.empty() && dxs.size() != ys.size()) {
        throw std::invalid_argument("dxs must be empty or of the same size as ys!");
    }
    if (xs.empty() || ys.empty()) {
        throw std::invalid_argument("xs and ys must not be empty!");
    }
    
    hash = 0;

    for(auto &input : ys) {
        hash = hash_combine(hash, input->hashcode());
    }
    for(auto &input : xs) {
        hash = hash_combine(hash, input->hashcode());
    }
    for(auto &input : dxs) {
        hash = hash_combine(hash, input->hashcode());
    }
}

std::vector<std::shared_ptr<Output>> Gradient::add_gradients(std::vector<std::shared_ptr<Output>> ys,
                                                          std::vector<std::shared_ptr<Output>> xs,
                                                          std::vector<std::shared_ptr<Output>> dxs) {
    // TODO it would be good to use make_shared instead, but constructor is private
    auto gradient = std::shared_ptr<Gradient>(new Gradient(std::move(ys), std::move(xs), std::move(dxs)));

    std::vector<std::shared_ptr<Output>> ret;
    for (unsigned int i = 0; i < gradient->xs.size(); ++i) {
        auto out = std::make_shared<Output>(gradient, i);
        gradient->outputs.emplace_back(out); // create a weak pointer
        ret.push_back(out);
    }

    return ret;
}

void Gradient::add_to_graph(GraphSession &graph) {
    std::vector<TF_Output> y_outputs;
    for (auto &y : ys) {
        y_outputs.push_back(graph.add_output(y.get()));
    }
    std::vector<TF_Output> x_outputs;
    for (auto &x : xs) {
        x_outputs.push_back(graph.add_output(x.get()));
    }
    std::vector<TF_Output> dx_outputs;
    for (auto &dx : dxs) {
        dx_outputs.push_back(graph.add_output(dx.get()));
    }

    std::vector<TF_Output> results(xs.size());
    run_with_status<void>(std::bind(TF_AddGradients, graph.get_underlying(), y_outputs.data(), y_outputs.size(),
                                    x_outputs.data(), x_outputs.size(), dx_outputs.data(), std::placeholders::_1, results.data()));

    for (size_t i = 0; i < results.size(); ++i) {
        auto out = outputs[i].lock(); // promote to shared_ptr, will return null if pointer is already disposed of
        if (out) {
            graph.register_output_hash(out->hashcode(), results[i]);
        }
    }
}

size_t Gradient::hashcode() const {
    return hash;
}