from tensorflow.core.framework import op_def_pb2
from cffi import FFI

generated_ops_file = '../src/GeneratedOps.luna'
file_header = """import Std.Foreign
import Std.Foreign.C.Value
import Tensorflow.CWrappers.Operations
import Tensorflow.Types
import Tensorflow.Operations"""

underscore_replacement = 'x'  # TODO: what to do with this?


def attr_code(attr):
    adjusted_name = attr.name.replace('_', underscore_replacement)

    attr_function_calls = {
        'type': '    addAttrType.call None [attrList.toCArg, nameCStr.toCArg, CInt32.fromInt {} . toCArg]\n'
                .format(adjusted_name),
        'shape': '    cdims = {0}.map CInt64.fromInt\n'
                 '    cdimsArray = (Array CInt64) . fromList cdims\n'
                 '    addAttrShape.call None '
                 '[attrList.toCArg, nameCStr.toCArg, cdimsArray.toCArg, CInt64.fromInt {0}.length . toCArg]\n'
                 '    cdimsArray.free\n'.format(adjusted_name),
        'tensor': '    addAttrTensor.call None [attrList.toCArg, nameCStr.toCArg, {}.ptr.toCArg]\n'.format(adjusted_name),
        'int': '    addAttrInt.call None [attrList.toCArg, nameCStr.toCArg, CInt64.fromInt {} . toCArg]\n'
                .format(adjusted_name),
        'float': '    addAttrFloat.call None [attrList.toCArg, nameCStr.toCArg, CFloat.fromReal {} . toCArg]\n'
                .format(adjusted_name),
        'bool': '    addAttrBool.call '
                'None [attrList.toCArg, nameCStr.toCArg, CUChar.fromInt (if {} then 1 else 0) . toCArg]\n'
                .format(adjusted_name),
        'string': '    valCStr = CString.fromText "{}"\n'
                  '    addAttrString.call None [attrList.toCArg, nameCStr.toCArg, valCStr.toCArg]\n'
                  '    valCStr.free\n'.format(adjusted_name)
    }

    return '    nameCStr = CString.fromText "{}"\n'.format(attr.name)\
        + attr_function_calls.get(attr.type, '')\
        + '    nameCStr.free\n'  #TODO: support all types


def op_code(operation):
    nargs = len(operation.input_arg)
    noutputs = len(operation.output_arg)
    nattrs = len(operation.attr)

    attr_function_loads = {
        'type': '    addAttrType = lookupSymbol "TFL" "add_attr_type"\n',
        'shape': '    addAttrShape = lookupSymbol "TFL" "add_attr_shape"\n',
        'tensor': '    addAttrTensor = lookupSymbol "TFL" "add_attr_tensor"\n',
        'int': '    addAttrInt = lookupSymbol "TFL" "add_attr_int"\n',
        'float': '    addAttrFloat = lookupSymbol "TFL" "add_attr_float"\n',
        'bool': '    addAttrBool = lookupSymbol "TFL" "add_attr_bool"\n',
        'string': '    addAttrString = lookupSymbol "TFL" "add_attr_string"\n'
    }

    head = 'def ' + (operation.name[:1].lower() + operation.name[1:]).replace('_', underscore_replacement)
    head += 'Gen'  # TODO: this is only temporary, to avoid conflict with existing functions
    for i in range(nargs):
        head += ' ' + operation.input_arg[i].name.replace('_', underscore_replacement)
    for i in range(nattrs):
        head += ' ' + operation.attr[i].name.replace('_', underscore_replacement)
    head += ':\n'

    if nattrs > 0:
        attr_setup = '    initAttrList = lookupSymbol "TFL" "attr_list_init"\n'
        for attr_type in set(attr.type for attr in operation.attr):
            attr_setup += attr_function_loads.get(attr_type, '')  # TODO: support all types
        attr_setup += '    releaseMethod = lookupSymbol "TFL" "release"\n'\
            '    attrListPtr = initAttrList.call (Pointer None) []\n'\
            '    attrList = ManagedPointer None . fromPointer releaseMethod attrListPtr\n'

        for attr in operation.attr:
            attr_setup += attr_code(attr)
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
