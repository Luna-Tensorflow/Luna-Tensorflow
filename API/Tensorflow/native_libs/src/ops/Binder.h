//
// Created by wojtek on 14.02.19.
//

#ifndef TFL_BINDER_H
#define TFL_BINDER_H

#include "../graph/GraphSession.h"
#include "Output.h"

class GraphSession;
class Output;

class Binder {
protected:
    Binder() = default;

public:
    virtual void add_to_graph(GraphSession &graph) = 0;

    virtual ~Binder() = default;

protected:
    std::vector<Output*> outputs;
};

#endif //TFL_BINDER_H
