#ifndef TFL_OPERATION_H
#define TFL_OPERATION_H

#include <cstddef>
#include <memory>
#include <tensorflow/c/c_api.h>

#include "../graph/GraphSession.h"
#include "../helpers/utils.h"
#include "../tensor/Tensor.h"

template<TF_DataType DataTypeLabel>
class Operation {
public:
    virtual size_t hashcode() const = 0;
    virtual ~Operation() = default;

    std::shared_ptr<Tensor<DataTypeLabel>> eval() const {
        GraphSession graph;
        graph.add_output(graph.add_operation(this));

        Tensor<DataTypeLabel>** values = graph.eval<DataTypeLabel>();

        auto ptr = values[0];
        free(values);

        return LifetimeManager::instance().accessOwned(ptr);
    }

    virtual TF_Output add_to_graph(GraphSession& graph) const = 0;
};


#endif //TFL_OPERATION_H
