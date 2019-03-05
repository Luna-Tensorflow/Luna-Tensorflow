//
// Created by wojtek on 13.02.19.
//

#ifndef TFL_OPERATION_H
#define TFL_OPERATION_H

#include <vector>
#include <memory>
#include <tensorflow/c/c_api.h>

#include "../graph/GraphSession.h"
#include "../helpers/utils.h"
#include "Output.h"
#include "Attr.h"
#include "Binder.h"

class Operation : public Binder {
private:
    Operation(std::string name, std::vector<std::shared_ptr<Output>> inputs,
            std::vector<std::shared_ptr<Attr>> attrs, std::string chosen_name);

public:
    static std::vector<std::shared_ptr<Output>> make_operation(std::string name,
                                                               std::vector<std::shared_ptr<Output>> inputs,
                                                               int num_outputs,
                                                               std::vector<std::shared_ptr<Attr>> attrs = {},
                                                               std::string chosen_name = "");

    void add_to_graph(GraphSession &graph) override;
    size_t hashcode() const override;

private:
    std::string name;
    std::vector<std::shared_ptr<Output>> inputs;
    std::vector<std::shared_ptr<Attr>> attrs;
    std::string chosen_name;
    std::vector<std::weak_ptr<Output>> outputs;
    size_t hash;
};

#endif //TFL_OPERATION_H
