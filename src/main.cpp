#include <iostream>
#include <tensorflow/c/c_api.h>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;

    int64_t dims[] = {n, m};
    TF_Tensor* tensors[2];

    // Allocating input tensors
    tensors[0] = TF_AllocateTensor(TF_INT32, dims, 2, n * m * TF_DataTypeSize(TF_INT32));
    tensors[1] = TF_AllocateTensor(TF_INT32, dims, 2, n * m * TF_DataTypeSize(TF_INT32));
    TF_Tensor* out_tensor;

    // Reading input tensor values
    for (auto &tensor : tensors) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int64_t index = i * TF_Dim(tensor, 1) + j;
                char* adr = (char*) TF_TensorData(tensor) + TF_DataTypeSize(TF_TensorType(tensor)) * index;

                cin >> *(int32_t*)adr;
            }
        }
    }

    TF_Graph* graph = TF_NewGraph();

    // Preparing descriptions of three operations (in other words: nodes) - two variable nodes and an addition node
    TF_OperationDescription* add_description = TF_NewOperation(graph, "Add", "Add_operation");
    TF_OperationDescription* variable_descriptions[2];
    for (int k = 0; k < 2; ++k) {
        variable_descriptions[k] = TF_NewOperation(graph, "Variable", ("Var" + to_string(k)).c_str());
    }

    TF_Operation* variables[2];
    TF_Status* status = TF_NewStatus();
    for (int k = 0; k < 2; ++k) {
        // Setting attributes of a variable node
        TF_SetAttrShape(variable_descriptions[k], "shape", dims, 2);
        TF_SetAttrType(variable_descriptions[k], "dtype", TF_INT32);

        // Actually creating the node
        variables[k] = TF_FinishOperation(variable_descriptions[k], status);
        if (TF_GetCode(status) != TF_OK) cout << "TF_FinishOperation message: " << TF_Message(status) << "\n";
    }

    // Connecting the outputs of variable nodes (which are their values) to the inputs of the addition node
    TF_Output variable_outputs[2];
    for (int k = 0; k < 2; ++k) {
        variable_outputs[k] = {
                .oper = variables[k],
                .index = 0
        };
        TF_AddInput(add_description, variable_outputs[k]);
    }

    // Actually creating the addition node
    TF_Operation* add_operation = TF_FinishOperation(add_description, status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_FinishOperation message: " << TF_Message(status) << "\n";

    // Creating a session to run the calculation
    TF_SessionOptions* options = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, options, status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_NewSession message: " << TF_Message(status) << "\n";

    // A variable representing the output of the addition node - necessary to extract this output
    TF_Output addition_output = {
            .oper = add_operation,
            .index = 0
    };

    // Do the calculation
    TF_SessionRun(session,
            nullptr,
            variable_outputs, tensors, 2, // we provide our input tensors by setting the outputs of the variable nodes
            &addition_output, &out_tensor, 1, // we specify which outputs we want to learn
            nullptr, 0, // we could provide some nodes that we want to execute, but don't need their output (it makes sense because of nodes like "Assign")
            nullptr,
            status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_SessionRun message: " << TF_Message(status) << "\n";

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int64_t index = i * TF_Dim(out_tensor, 1) + j;
            char* adr = (char*) TF_TensorData(out_tensor) + TF_DataTypeSize(TF_TensorType(out_tensor)) * index;

            cout << *(int32_t*)adr << " ";
        }
        cout << "\n";
    }

    TF_CloseSession(session, status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_CloseSession message: " << TF_Message(status) << "\n";
    TF_DeleteSessionOptions(options);
    TF_DeleteSession(session, status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_DeleteSession message: " << TF_Message(status) << "\n";
    TF_DeleteGraph(graph);
    for (auto &tensor : tensors) {
        TF_DeleteTensor(tensor);
    }
    TF_DeleteTensor(out_tensor);
    return 0;
}
