from tensorflow.core.framework import op_def_pb2
from cffi import FFI

generated_ops_file = '../src/GeneratedOps.luna'
file_header = """import Std.Foreign
import Std.Foreign.C.Value
import Tensorflow.CWrappers.Operations
import Tensorflow.Types
import Tensorflow.Operations"""

underscore_replacement = 'x'  # TODO: what to do with this?


def op_code(operation):
    nargs = len(operation.input_arg)
    noutputs = len(operation.output_arg)
    nattrs = len(operation.attr)

    head = 'def ' + (operation.name[:1].lower() + operation.name[1:]).replace('_', underscore_replacement)
    head += 'Gen'  # TODO: this is only temporary, to avoid conflict with existing functions
    for i in range(nargs):
        head += ' ' + operation.input_arg[i].name.replace('_', underscore_replacement)
    for i in range(nattrs):
        head += ' ' + operation.attr[i].name.replace('_', underscore_replacement)
    head += ':\n'

    if nattrs > 0:
        attr_setup = """    initAttrList = lookupSymbol "TFL" "attr_list_init"
    addAttrType = lookupSymbol "TFL" "add_attr_type"
    addAttrShape = lookupSymbol "TFL" "add_attr_shape"
    addAttrTensor = lookupSymbol "TFL" "add_attr_tensor"
    releaseMethod = lookupSymbol "TFL" "release"
    attrListPtr = initAttrList.call (Pointer None) []
    attrList = ManagedPointer None . fromPointer releaseMethod attrListPtr
"""
        for attr in operation.attr:
            attr_setup += '    nameCStr = CString.fromText "{}"\n'.format(attr.name)
            if attr.type == "type":
                attr_setup += '    addAttrType.call None [attrList.toCArg, nameCStr.toCArg, CInt32.fromInt {} . toCArg]\n'.format(attr.name.replace('_', underscore_replacement))
            elif attr.type == "shape":
                attr_setup += '    cdims = {}.map CInt64.fromInt\n'.format(attr.name.replace('_', underscore_replacement))
                attr_setup += '    cdimsArray = (Array CInt64) . fromList cdims\n'
                attr_setup += '    addAttrShape.call None [attrList.toCArg, nameCStr.toCArg, cdimsArray.toCArg, CInt64.fromInt {}.length . toCArg]\n'.format(attr.name.replace('_', underscore_replacement))
                attr_setup += '    cdimsArray.free\n'
            elif attr.type == "tensor":
                attr_setup += '    addAttrTensor.call None [attrList.toCArg, nameCStr.toCArg, {}.ptr.toCArg]\n'.format(attr.name.replace('_', underscore_replacement))
            attr_setup += '    nameCStr.free\n'
    else:
        attr_setup = ''

    make_wrappers = '    wrappers = makeOutputWrappers "' + operation.name + '" ['
    for i in range(nargs):
        make_wrappers += operation.input_arg[i].name.replace('_', underscore_replacement)
        if i < nargs - 1:
            make_wrappers += ', '
    make_wrappers += '] ' + str(noutputs) + ' attrList ""\n'

    if noutputs == 1:
        return_line = '    TFOutput wrappers.head.get'
    else:
        return_line = '    wrappers.each (wrapper: TFOutput wrapper)'
    return head + attr_setup + make_wrappers + return_line


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
            f.write('\n\n' + op_code(op))
    f.write('\n')

TF.TF_DeleteBuffer(ops)
