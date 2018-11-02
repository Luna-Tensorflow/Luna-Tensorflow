#include <iostream>
#include <tensorflow/c/c_api.h>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;

    int64_t dims[] = {n, m};
    TF_Tensor* tensors[2];

    tensors[0] = TF_AllocateTensor(TF_INT32, dims, 2, n * m * TF_DataTypeSize(TF_INT32));
    tensors[1] = TF_AllocateTensor(TF_INT32, dims, 2, n * m * TF_DataTypeSize(TF_INT32));
    TF_Tensor* out_tensor = TF_AllocateTensor(TF_INT32, dims, 2, n * m * TF_DataTypeSize(TF_INT32));

    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int index = i * TF_Dim(tensors[k], 1) + j;
                char* adr = (char*) TF_TensorData(tensors[k]) + TF_DataTypeSize(TF_TensorType(tensors[k])) * index;

                cin >> *(int32_t*)adr;
            }
        }
    }

    TF_Graph* graph = TF_NewGraph();

    TF_OperationDescription* add_description = TF_NewOperation(graph, "Add", "Add_operation");
    TF_OperationDescription* variable_descriptions[2];
    for (int k = 0; k < 2; ++k) {
        variable_descriptions[k] = TF_NewOperation(graph, "Variable", "Var" + k);
    }

    TF_Operation* variables[2];
    TF_Status* status = TF_NewStatus();
    for (int k = 0; k < 2; ++k) {
        TF_SetAttrShape(variable_descriptions[k], "shape", dims, 2);
        TF_SetAttrType(variable_descriptions[k], "dtype", TF_INT32);
        variables[k] = TF_FinishOperation(variable_descriptions[k], status);
        if (TF_GetCode(status) != TF_OK) cout << "TF_FinishOperation message: " << TF_Message(status) << "\n";
    }

    TF_Output variable_outputs[2];
    for (int k = 0; k < 2; ++k) {
        variable_outputs[k] = {
                .oper = variables[k],
                .index = 0
        };
        TF_AddInput(add_description, variable_outputs[k]);
    }

    TF_Operation* add_operation = TF_FinishOperation(add_description, status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_FinishOperation message: " << TF_Message(status) << "\n";

    TF_SessionOptions* options = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, options, status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_NewSession message: " << TF_Message(status) << "\n";

    TF_Output output = {
            .oper = add_operation,
            .index = 0
    };

    TF_SessionRun(session,
            NULL,
            variable_outputs, tensors, 2,
            &output, &out_tensor, 1,
            &add_operation, 1,
            NULL,
            status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_SessionRun message: " << TF_Message(status) << "\n";

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int index = i * TF_Dim(out_tensor, 1) + j;
            char* adr = (char*) TF_TensorData(out_tensor) + TF_DataTypeSize(TF_TensorType(out_tensor)) * index;

            cout << *(int32_t*)adr << " ";
        }
        cout << "\n";
    }

    TF_DeleteSessionOptions(options);
    TF_DeleteSession(session, status);
    if (TF_GetCode(status) != TF_OK) cout << "TF_DeleteSession message: " << TF_Message(status) << "\n";
    TF_DeleteGraph(graph);
    for (int k = 0; k < 2; ++k) {
        TF_DeleteTensor(tensors[k]);
    }
    TF_DeleteTensor(out_tensor);
    return 0;
}