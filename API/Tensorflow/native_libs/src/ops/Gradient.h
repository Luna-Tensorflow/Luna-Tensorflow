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
#include "Output.h"

class Gradient : public Binder {
private:
    Gradient(std::vector<std::shared_ptr<Output>> ys,
            std::vector<std::shared_ptr<Output>> xs,
            std::vector<std::shared_ptr<Output>> dxs);

public:
    static std::vector<std::shared_ptr<Output>> add_gradients(std::vector<std::shared_ptr<Output>> ys,
            std::vector<std::shared_ptr<Output>> xs, std::vector<std::shared_ptr<Output>> dxs);

    void add_to_graph(GraphSession &graph);

private:
    std::vector<std::shared_ptr<Output>> ys;
    std::vector<std::shared_ptr<Output>> xs;
    std::vector<std::shared_ptr<Output>> dxs;
};

#endif //TFL_GRADIENT_H
