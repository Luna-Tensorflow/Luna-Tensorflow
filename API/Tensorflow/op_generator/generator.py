from tensorflow.core.framework import op_def_pb2
from cffi import FFI
from itertools import chain

generated_ops_file = '../src/GeneratedOps.luna'
file_header = """import Std.Foreign
import Std.Foreign.C.Value
import Tensorflow.CWrappers.Operations
import Tensorflow.CWrappers.Helpers
import Tensorflow.Types
import Tensorflow.Operations"""

typetags = {
    1: 'FloatType',
    2: 'DoubleType',
    3: 'Int32Type',
    4: 'UInt8Type',
    5: 'Int16Type',
    6: 'Int8Type',
    7: 'StringType',
    9: 'Int64Type',
    10: 'BoolType',
    17: 'UInt16Type',
    22: 'UInt32Type',
    23: 'UInt64Type'
}


def lunify_name(name):
    name_parts = name.split('_')
    lunified_name = ''.join(name_parts[:1] + list(map(lambda part: part[:1].upper() + part[1:], name_parts[1:])))
    return lunified_name[:1].lower() + lunified_name[1:]


def attr_code(attr):
    adjusted_name = lunify_name(attr.name)

    attr_function_calls = {
        'type': '        callHandlingError "add_attr_type" None [attrList.toCArg, nameCStr.toCArg, '
                'CInt32.fromInt {}.num . toCArg]\n'
                .format(adjusted_name),
        'list(type)': '        ctypes = {0}.map (type: CInt32.fromInt type.num)\n'
                      '        Array CInt32 . with ctypes ctypesArray:\n'
                      '            callHandlingError "add_attr_type_list" None '
                      '[attrList.toCArg, nameCStr.toCArg, ctypesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                      .format(adjusted_name),
        'shape': '        cdims = {0}.map CInt64.fromInt\n'
                 '        Array CInt64 . with cdims cdimsArray:\n'
                 '            callHandlingError "add_attr_shape" None '
                 '[attrList.toCArg, nameCStr.toCArg, cdimsArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                 .format(adjusted_name),
        'list(shape)': '        len = {0}.length\n'
                       '        cValues = ManagedPointer (Pointer CInt64) . mallocElems len\n'
                       '        indexed = 0.upto (len - 1) . zip {0}\n'
                       '        indexed.each (idx, elem):\n'
                       '                cValues.moveElems idx . write '
                       '(Array CInt64 . fromList (elem.map CInt64.fromInt))\n'
                       '        cLengths = ManagedPointer CUInt32 . mallocElems len\n'
                       '        indexed.each (ids, elem): cLengths.moveElems idx . write '
                       '(CUInt32.fromInt elem.length)\n'
                       '        callHandlingError "add_attr_shape_list" None '
                       '[attrList.toCArg, nameCStr.toCArg, cValues.toCArg, cLengths.toCArg, '
                       'CUInt32.fromInt len . toCArg]\n'
                       '        cValues.toList.each (str: str.free)\n'
                       '        0.upto (len - 1) . each (idx: cValues.moveElems idx . read . free)\n'
                       .format(adjusted_name),
        'tensor': '        callHandlingError "add_attr_tensor" None [attrList.toCArg, nameCStr.toCArg, {}.ptr.toCArg]\n'
                  .format(adjusted_name),
        'list(tensor)': '        ctensors = {0}.map (elem: elem.ptr.ptr)\n'
                        '        Array (Pointer None) . with ctensors ctensorsArray:\n'
                        '            callHandlingError "add_attr_tensor_list" None '
                        '[attrList.toCArg, nameCStr.toCArg, ctensorsArray.toCArg, '
                        'CUInt32.fromInt {0}.length . toCArg]\n'.format(adjusted_name),
        'int': '        callHandlingError "add_attr_int" None [attrList.toCArg, nameCStr.toCArg, '
               'CInt64.fromInt {} . toCArg]\n'
               .format(adjusted_name),
        'list(int)': '        cvalues = {0}.map CInt64.fromInt\n'
                     '        Array CInt64 . with cvalues cvaluesArray:\n'
                     '            callHandlingError "add_attr_int_list" None '
                     '[attrList.toCArg, nameCStr.toCArg, cvaluesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                     .format(adjusted_name),
        'float': '        callHandlingError "add_attr_float" None [attrList.toCArg, nameCStr.toCArg, '
                 'CFloat.fromReal {} . toCArg]\n'
                 .format(adjusted_name),
        'list(float)': '        cvalues = {0}.map CFloat.fromReal\n'
                       '        Array CFloat . with cvalues cvaluesArray:\n'
                       '            callHandlingError "add_attr_float_list" None '
                       '[attrList.toCArg, nameCStr.toCArg, cvaluesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                       .format(adjusted_name),
        'bool': '        callHandlingError "add_attr_bool" '
                'None [attrList.toCArg, nameCStr.toCArg, CUChar.fromInt (if {} then 1 else 0) . toCArg]\n'
                .format(adjusted_name),
        'list(bool)': '        cvalues = {0}.map (b: CUChar.fromInt (if b then 1 else 0))\n'
                      '        Array CUChar . with cvalues cvaluesArray:\n'
                      '            callHandlingError "add_attr_bool_list" None '
                      '[attrList.toCArg, nameCStr.toCArg, cvaluesArray.toCArg, CUInt32.fromInt {0}.length . toCArg]\n'
                      .format(adjusted_name),
        'string': '        CString.with {} valCStr:\n'
                  '            callHandlingError "add_attr_string" None [attrList.toCArg, nameCStr.toCArg, '
                  'valCStr.toCArg]\n'.format(adjusted_name),
        'list(string)': '        len = {0}.length\n'
                        '        cValues = ManagedPointer (Pointer CInt64) . mallocElems len\n'
                        '        indexed = 0.upto (len - 1) . zip {0}\n'
                        '        indexed.each (idx, elem):\n'
                        '                cValues.moveElems idx . write (CString.fromText elem)\n'
                        '        callHandlingError "add_attr_string_list" None '
                        '[attrList.toCArg, nameCStr.toCArg, cValues.toCArg, CUInt32.fromInt len . toCArg]\n'
                        '        0.upto (len - 1) . each (idx: cValues.moveElems idx . read . free)\n'
                        .format(adjusted_name),
        'func': '        CString.with {} funcNameCStr:\n'
                '            callHandlingError "add_attr_func_name" None [attrList.toCArg, nameCStr.toCArg, '
                'funcNameCStr.toCArg]\n'.format(adjusted_name)
    }

    return '    CString.with "{}" nameCStr:\n'.format(attr.name)\
        + attr_function_calls[attr.type]


def output_arg_typetag(output_arg, op):
    if output_arg.type:
        return typetags[output_arg.type]

    for input_arg in op.input_arg:
        if input_arg.type_attr == output_arg.type_attr:
            return '{}.typetag'.format(lunify_name(input_arg.name))

    return lunify_name(output_arg.type_attr)


def op_code(operation):
    nargs = len(operation.input_arg)
    noutputs = len(operation.output_arg)
    nattrs = len(operation.attr)

    head = 'def ' + lunify_name(operation.name)
    head += 'Gen'
    head += ' chosenName'
    for i in range(nargs):
        head += ' ' + lunify_name(operation.input_arg[i].name)
    for i in range(nattrs):
        head += ' ' + lunify_name(operation.attr[i].name)
    head += ':\n'

    if nattrs > 0:
        attr_setup = '    attrListPtr = callHandlingError "attr_list_init" (Pointer None) []\n'\
                     '    attrList = ManagedPointer None . fromPointer releaseMethod attrListPtr\n'

        for attr in operation.attr:
            attr_setup += attr_code(attr)
    else:
        attr_setup = ''

    make_wrappers = '    wrappers = makeOutputWrappers "' + operation.name + '" ['
    for i in range(nargs):
        make_wrappers += lunify_name(operation.input_arg[i].name) + ".wrapper"
        if i < nargs - 1:
            make_wrappers += ', '
    make_wrappers += ']'
    make_wrappers += ' ' + str(noutputs)
    # for i in range(noutputs):
    #     make_wrappers += output_arg_typetag(operation.output_arg[i], operation)
    #     if i < noutputs - 1:
    #         make_wrappers += ', '
    make_wrappers += ' attrList chosenName\n'

    if noutputs == 0:
        return_line = '    None'
    elif noutputs == 1:
        return_line = '    TFOutput wrappers.head.get ' + output_arg_typetag(operation.output_arg[0], operation)
    else:
        return_line = '    ('
        for i in range(noutputs):
            typetag = output_arg_typetag(operation.output_arg[i], operation)
            return_line += f"TFOutput (wrappers.getAt {i}) {typetag}"
            if i < noutputs - 1:
                return_line += ', '
        return_line += ')'
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

print("Loading the library")
TF = ffi.dlopen('../../../tensorflow/lib/libtensorflow.so')

print("Fetching operations list")
ops = TF.TF_GetAllOpList()

opList = op_def_pb2.OpList()
opList.ParseFromString(ffi.buffer(ops.data, ops.length)[:])


def has_tensor_list(op):
    return any(arg.type_list_attr or arg.number_attr for arg in chain(op.input_arg, op.output_arg))


def has_supported_types(op):
    for arg in chain(op.input_arg, op.output_arg):
        if arg.type:
            if arg.type not in typetags.keys():
                return False
        else:
            type_attr = next(filter(lambda attr: attr.name == arg.type_attr, op.attr))
            if type_attr.HasField('allowed_values') and \
                    not any(dtype in typetags.keys() for dtype in type_attr.allowed_values.list.type):
                return False
    return True


def is_internal(op):
    return op.name[0] == '_'


def is_supported(op):
    return not op.is_stateful and not has_tensor_list(op) and has_supported_types(op) and not is_internal(op)


print("Generating wrappers")
skipped = 0
written = 0
with open(generated_ops_file, 'w') as f:
    f.write(file_header)
    for op in opList.op:
        if is_supported(op):
            f.write('\n\n' + op_code(op))
            written += 1
        else:
            skipped += 1
    f.write('\n')

TF.TF_DeleteBuffer(ops)
print(f"Written {written} supported operations, skipped {skipped}.")
