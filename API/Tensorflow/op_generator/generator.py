from tensorflow.core.framework import op_def_pb2
from cffi import FFI

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

for op in opList.op:
    #print(op.HasField("deprecation"))
    if len(op.output_arg) > 1:
        print(op.name)

TF.TF_DeleteBuffer(ops)
