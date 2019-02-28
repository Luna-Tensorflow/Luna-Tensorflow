from tensorflow.core.framework import op_def_pb2
from cffi import FFI

generated_ops_file = '../src/GeneratedOps.luna'
file_header = """import Std.Foreign
import Std.Foreign.C.Value
import Tensorflow.CWrappers.Operations
import Tensorflow.Types
import Tensorflow.Operations"""

underscore_replacement = 'x'  # TODO: what to do with this?


def lunify_name(name):
    return (name[:1].lower() + name[1:]).replace('_', underscore_replacement)


def attr_code(attr):
    adjusted_name = lunify_name(attr.name)

    attr_function_calls = {
        'type': '    addAttrType.call None [attrList.toCArg, nameCStr.toCArg, CInt32.fromInt {} . toCArg]\n'
                .format(adjusted_name),
        'list(type)': '    ctypes = {0}.map CInt32.fromInt\n'
                      '    ctypesArray = (Array CInt32) . fromList ctypes\n'
                      '    addAttrTypeList.call None '
                      '[attrList.toCArg, nameCStr.toCArg, ctypesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                      '    ctypesArray.free\n'.format(adjusted_name),
        'shape': '    cdims = {0}.map CInt64.fromInt\n'
                 '    cdimsArray = (Array CInt64) . fromList cdims\n'
                 '    addAttrShape.call None '
                 '[attrList.toCArg, nameCStr.toCArg, cdimsArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                 '    cdimsArray.free\n'.format(adjusted_name),
        'list(shape)': '    len = {0}.length\n'
                       '    cValues = ManagedPointer (Pointer CInt64) . mallocElems len\n'
                       '    indexed = 0.upto len . zip {0}\n'
                       '    indexed.each (idx, elem):\n'
                       '            cValues.moveElems idx . write (Array CInt64 . fromList (elem.map CInt64.fromInt))\n'
                       '    cLengths = ManagedPointer CUInt32 . mallocElems len\n'
                       '    indexed.each (ids, elem): cLengths.moveElems idx . write (CUInt32.fromInt elem.length)\n'
                       '    addAttrShapeList.call None '
                       '[attrList.toCArg, nameCStr.toCArg, cValues.toCArg, cLengths.toCArg, '
                       'CUInt32.fromInt len . toCArg]\n'
                       '    cValues.toList.each (str: str.free)\n'
                       '    0.upto (len - 1) . each (idx: cValues.moveElems idx . read . free)\n'.format(adjusted_name),
        'tensor': '    addAttrTensor.call None [attrList.toCArg, nameCStr.toCArg, {}.ptr.toCArg]\n'
                .format(adjusted_name),
        'list(tensor)': '    ctensors = {0}.map (elem: elem.ptr.ptr)\n'
                        '    ctensorsArray = (Array (Pointer None)) . fromList ctensors\n'
                        '    addAttrTensorList.call None '
                        '[attrList.toCArg, nameCStr.toCArg, ctensorsArray.toCArg, '
                        'CUInt32.fromInt {0}.length . toCArg]\n'
                        '    ctensorsArray.free\n'.format(adjusted_name),
        'int': '    addAttrInt.call None [attrList.toCArg, nameCStr.toCArg, CInt64.fromInt {} . toCArg]\n'
                .format(adjusted_name),
        'list(int)': '    cvalues = {0}.map CInt64.fromInt\n'
                     '    cvaluesArray = (Array CInt64) . fromList cvalues\n'
                     '    addAttrIntList.call None '
                     '[attrList.toCArg, nameCStr.toCArg, cvaluesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                     '    cvaluesArray.free\n'.format(adjusted_name),
        'float': '    addAttrFloat.call None [attrList.toCArg, nameCStr.toCArg, CFloat.fromReal {} . toCArg]\n'
                .format(adjusted_name),
        'list(float)': '    cvalues = {0}.map CFloat.fromReal\n'
                     '    cvaluesArray = (Array CFloat) . fromList cvalues\n'
                     '    addAttrFloatList.call None '
                     '[attrList.toCArg, nameCStr.toCArg, cvaluesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                     '    cvaluesArray.free\n'.format(adjusted_name),
        'bool': '    addAttrBool.call '
                'None [attrList.toCArg, nameCStr.toCArg, CUChar.fromInt (if {} then 1 else 0) . toCArg]\n'
                .format(adjusted_name),
        'list(bool)': '    cvalues = {0}.map (b: CUChar.fromInt (if b then 1 else 0))\n'
                     '    cvaluesArray = (Array CUChar) . fromList cvalues\n'
                     '    addAttrBoolList.call None '
                     '[attrList.toCArg, nameCStr.toCArg, cvaluesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                     '    cvaluesArray.free\n'.format(adjusted_name),
        'string': '    valCStr = CString.fromText "{}"\n'
                  '    addAttrString.call None [attrList.toCArg, nameCStr.toCArg, valCStr.toCArg]\n'
                  '    valCStr.free\n'.format(adjusted_name),
        'list(string)': '    len = {0}.length\n'
                        '    cValues = ManagedPointer (Pointer CInt64) . mallocElems len\n'
                        '    indexed = 0.upto len . zip {0}\n'
                        '    indexed.each (idx, elem):\n'
                        '            cValues.moveElems idx . write (CString.fromText elem)\n'
                        '    addAttrStringList.call None '
                        '[attrList.toCArg, nameCStr.toCArg, cValues.toCArg, CUInt32.fromInt len . toCArg]\n'
                        '    0.upto (len - 1) . each (idx: cValues.moveElems idx . read . free)\n'.format(adjusted_name),
        'func': '    funcNameCStr = CString.fromText "{}"\n'
                '    addAttrFuncName.call None [attrList.toCArg, nameCStr.toCArg, funcNameCStr.toCArg]\n'
                '    funcNameCStr.free\n'.format(adjusted_name)
    }

    return '    nameCStr = CString.fromText "{}"\n'.format(attr.name)\
        + attr_function_calls[attr.type]\
        + '    nameCStr.free\n'


def op_code(operation):
    nargs = len(operation.input_arg)
    noutputs = len(operation.output_arg)
    nattrs = len(operation.attr)

    attr_function_loads = {
        'type': '    addAttrType = lookupSymbol "TFL" "add_attr_type"\n',
        'list(type)': '    addAttrTypeList = lookupSymbol "TF" "add_attr_type_list\n',
        'shape': '    addAttrShape = lookupSymbol "TFL" "add_attr_shape"\n',
        'list(shape)': '    addAttrShapeList = lookupSymbol "TFL" "add_attr_shape_list"\n',
        'tensor': '    addAttrTensor = lookupSymbol "TFL" "add_attr_tensor"\n',
        'list(tensor)': '    addAttrTensorList = lookupSymbol "TFL" "add_attr_tensor_list"\n',
        'int': '    addAttrInt = lookupSymbol "TFL" "add_attr_int"\n',
        'list(int)': '    addAttrIntList = lookupSymbol "TFL" "add_attr_int_list"\n',
        'float': '    addAttrFloat = lookupSymbol "TFL" "add_attr_float"\n',
        'list(float)': '    addAttrFloatList = lookupSymbol "TFL" "add_attr_float_list"\n',
        'bool': '    addAttrBool = lookupSymbol "TFL" "add_attr_bool"\n',
        'list(bool)': '    addAttrBoolList = lookupSymbol "TFL" "add_attr_bool_list"\n',
        'string': '    addAttrString = lookupSymbol "TFL" "add_attr_string"\n',
        'list(string)': '    addAttrStringList = lookupSymbol "TFL" "add_attr_string_list"\n',
        'func': '    addAttrFuncName = lookupSymbol "TFL" "add_attr_func_name"\n'
    }

    head = 'def ' + lunify_name(operation.name)
    head += 'Gen'  # TODO: this is only temporary, to avoid conflict with existing functions
    head += ' chosenName'
    for i in range(nargs):
        head += ' ' + lunify_name(operation.input_arg[i].name)
    for i in range(nattrs):
        head += ' ' + lunify_name(operation.attr[i].name)
    head += ':\n'

    if nattrs > 0:
        attr_setup = '    initAttrList = lookupSymbol "TFL" "attr_list_init"\n'
        for attr_type in set(attr.type for attr in operation.attr):
            attr_setup += attr_function_loads[attr_type]
        attr_setup += '    releaseMethod = lookupSymbol "TFL" "release"\n'\
            '    attrListPtr = initAttrList.call (Pointer None) []\n'\
            '    attrList = ManagedPointer None . fromPointer releaseMethod attrListPtr\n'

        for attr in operation.attr:
            attr_setup += attr_code(attr)
    else:
        attr_setup = ''

    make_wrappers = '    wrappers = makeOutputWrappers "' + operation.name + '" ['
    for i in range(nargs):
        make_wrappers += lunify_name(operation.input_arg[i].name)
        if i < nargs - 1:
            make_wrappers += ', '
    make_wrappers += '] ' + str(noutputs) + ' attrList chosenName\n'

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
