//
// Created by wojtek on 13.02.19.
//

#ifndef TFL_OUTPUT_H
#define TFL_OUTPUT_H

#include <vector>

#include "../graph/GraphSession.h"
#include "../helpers/utils.h"
#include "Binder.h"

class GraphSession;
class Binder;

class Output {
public:
    Output(std::shared_ptr<Binder> binder);

    TF_Output add_to_graph(GraphSession& graph) const;

    size_t hashcode() const;

    std::shared_ptr<Tensor> eval() const;
private:
    size_t hash;
    std::shared_ptr<Binder> binder;
};

#endif //TFL_OUTPUT_H
