#ifndef TFL_OPERATION_H
#define TFL_OPERATION_H

#include <vector>
#include <memory>
#include <tensorflow/c/c_api.h>

#include "../graph/GraphSession.h"
#include "../helpers/utils.h"
#include "Output.h"
#include "Attr.h"
#include "Node.h"

class Operation : public Node {
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
    std::string hash_log() const override;

    std::string get_operation_name() const {
        return name;
    }
    bool has_custom_chosen_name() const {
        return is_chosen_name_custom;
    }

private:
    std::string name;
    std::vector<std::shared_ptr<Output>> inputs;
    std::vector<std::shared_ptr<Attr>> attrs;
    std::string chosen_name;
    bool is_chosen_name_custom;
    std::vector<std::weak_ptr<Output>> outputs;
    size_t hash;
};

#endif //TFL_OPERATION_H
