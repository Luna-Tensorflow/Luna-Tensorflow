//
// Created by wojtek on 13.02.19.
//

#include "Output.h"

Output::Output(std::shared_ptr<Binder> binder) : binder(std::move(binder)) {
    static size_t hash_ctr = 0;
    hash = ++hash_ctr;
}


TF_Output Output::add_to_graph(GraphSession& graph) const {
    if (!graph.exists(this)) {
        binder->add_to_graph(graph);
    }

    return graph.get_output(this);
};

size_t Output::hashcode() const {
    return hash;
}

std::shared_ptr<Tensor> Output::eval() const {
    GraphSession graph;
    graph.add_fetched_output(graph.add_output(this));

    Tensor** values = LifetimeManager::instance().addOwnershipOfArray(graph.eval()->outputs);

    auto ptr = values[0];
    free(values);

    return LifetimeManager::instance().accessOwned(ptr);
}