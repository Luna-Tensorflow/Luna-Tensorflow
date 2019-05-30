#ifndef TFL_GRADIENT_H
#define TFL_GRADIENT_H

#include <vector>
#include <memory>
#include <tensorflow/c/c_api.h>

#include "../graph/GraphSession.h"
#include "../helpers/utils.h"
#include "Output.h"

class Gradient : public Node {
private:
    Gradient(std::vector<std::shared_ptr<Output>> ys,
            std::vector<std::shared_ptr<Output>> xs,
            std::vector<std::shared_ptr<Output>> dxs);

public:
    static std::vector<std::shared_ptr<Output>> add_gradients(std::vector<std::shared_ptr<Output>> ys,
            std::vector<std::shared_ptr<Output>> xs, std::vector<std::shared_ptr<Output>> dxs);

    void add_to_graph(GraphSession &graph) override;
    size_t hashcode() const override;
    std::string hash_log() const override;

private:
    std::vector<std::shared_ptr<Output>> ys;
    std::vector<std::shared_ptr<Output>> xs;
    std::vector<std::shared_ptr<Output>> dxs;
    std::vector<std::weak_ptr<Output>> outputs;
    size_t hash;
};

#endif //TFL_GRADIENT_H
