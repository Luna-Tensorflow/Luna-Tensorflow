//
// Created by Radek on 27.02.2019.
//

#include "Sequence.h"

Sequence::Sequence(std::shared_ptr<Output> sideffect, std::shared_ptr<Output> value)
    : side_effect(sideffect), value(value) {
        hash = sideffect->hashcode();
        hash = hash_combine(hash, value->hashcode());
    }

std::shared_ptr<Output> Sequence::make_sequence(std::shared_ptr<Output> sideffect,
                                                std::shared_ptr<Output> value) {
    auto assignment = std::shared_ptr<Sequence>(new Sequence(sideffect, value));

    auto out = std::make_shared<Output>(assignment, 0);
    assignment->output = out;

    return out;
}

void Sequence::add_to_graph(GraphSession &graph) {
    TF_Output effectOut = side_effect->add_to_graph(graph);
    graph.register_sideefect(effectOut.oper);

    LOG_GRAPH(hash_log(), "[side_effect] " + side_effect->get_binder()->hash_log(), "[value] " + value->get_binder()->hash_log());

    auto out = output.lock();
    if (out) {
        // add input unit to graph
        TF_Output valueout = value->add_to_graph(graph);

        // register our output's hash to point to the same output that our unit gave
        graph.register_output_hash(out->hashcode(), valueout);
    } else {
        // this will never happen, because Sequence cannot outlive its only Output
        throw std::logic_error("Causality violation");
    }
}

size_t Sequence::hashcode() const {
    return hash;
} 

std::string Sequence::hash_log() const {
    return "Sequence: " + std::to_string(hash);
}
