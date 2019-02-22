from tensorflow.core.framework import op_def_pb2
from cffi import FFI

generated_ops_file = '../src/GeneratedOps.luna'
file_header = """import Tensorflow.CWrappers.Operations
import Tensorflow.Types
import Tensorflow.Operations"""


def op_code(name, nargs, noutputs):
    head = 'def ' + name.lower().replace('_', 'underscore')  # TODO: what to do with this?
    for i in range(nargs):
        head += ' in' + str(i)
    head += ':'
    make_wrappers = 'wrappers = makeOutputWrappers "' + name + '" ['
    for i in range(nargs):
        make_wrappers += 'in' + str(i)
        if i < nargs - 1:
            make_wrappers += ', '
    make_wrappers += '] ' + str(noutputs) + ' ""'
    if noutputs == 1:
        return_line = 'TFOutput wrappers.head.get'
    else:
        return_line = 'wrappers.each (wrapper: TFOutput wrapper)'
    return head + '\n    ' + make_wrappers + '\n    ' + return_line


ffi = FFI()
ffi.cdef("""
    typedef struct TF_Buffer {
        const void* data;
        size_t length;
        void (*data_deallocator)(void* data, size_t length);
    } TF_Buffer;

    TF_Buffer* TF_GetAllOpList();
    
    void TF_DeleteBuffer(TF_Buffer*);
""")

TF = ffi.dlopen('tensorflow')

ops = TF.TF_GetAllOpList()

opList = op_def_pb2.OpList()
opList.ParseFromString(ffi.buffer(ops.data, ops.length)[:])

with open(generated_ops_file, 'w+') as f:
    f.write(file_header)
    for op in opList.op:
        if not op.is_stateful:
            f.write('\n\n' + op_code(op.name, len(op.input_arg), len(op.output_arg)))
    f.write('\n')

TF.TF_DeleteBuffer(ops)
