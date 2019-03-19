//
// Created by wojtek on 13.02.19.
//

#include "Output.h"

Output::Output(std::shared_ptr<Binder> binder, size_t index) : binder(std::move(binder)) {
    hash = std::hash<int>()(index);
    hash = hash_combine(hash, this->binder->hashcode());
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

std::shared_ptr<Binder> Output::get_binder() {
    return binder;
}

Output::~Output()
{
    LOG("finalizing ~", binder->hash_log());
}
