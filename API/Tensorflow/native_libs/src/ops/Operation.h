#ifndef TFL_OPERATION_H
#define TFL_OPERATION_H

#include <cstddef>
#include <memory>
#include <cstdlib>
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
        char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
        putenv(suppress_tf_log);

        GraphSession graph;
        TF_Tensor *output_value;

        TF_Output output = graph.add(this);

        run_with_status<void>(std::bind(TF_SessionRun,
                                        graph.get_underlying_session(),
                                        nullptr,
                                        nullptr, nullptr, 0,
                                        &output, &output_value, 1,
                                        nullptr, 0,
                                        nullptr,
                                        std::placeholders::_1));

        return std::make_shared<Tensor<DataTypeLabel>>(output_value);
    }

    virtual TF_Output add_to_graph(GraphSession& graph) const = 0;
};


#endif //TFL_OPERATION_H
