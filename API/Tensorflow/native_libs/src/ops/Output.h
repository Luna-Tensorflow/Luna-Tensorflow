#ifndef TFL_OUTPUT_H
#define TFL_OUTPUT_H

#include <vector>
#include <memory>

#include "../helpers/utils.h"
#include "../tensor/Tensor.h"
#include "Node.h"

class GraphSession;
class Node;

/*
 * An Output represents one of the outputs of the computation
 * represented by it's parent Node.
 * When creating a Node, we fetch its Outputs and may use them as inputs into other operations.
 * Outputs are what is provided for a graph during evaluation to compute.
 *
 * Output's lifetime is managed by its owner and the shared_ptr reference to its Node
 * makes sure the Node is alive as long as at least one of its Outputs need it.
 */
class Output {
public:
    Output(std::shared_ptr<Node> binder, size_t index);

    TF_Output add_to_graph(GraphSession& graph) const;

    size_t hashcode() const;

    std::shared_ptr<Tensor> eval() const;
    std::shared_ptr<Node> get_binder();

    ~Output();
private:
    size_t hash;
    std::shared_ptr<Node> binder;
};

#endif //TFL_OUTPUT_H
