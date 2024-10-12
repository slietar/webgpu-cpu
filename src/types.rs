use cranelift::codegen::ir;


pub fn translate_primitive_type(input_type: &naga::TypeInner) -> ir::types::Type {
    match input_type {
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }) => ir::types::F32,
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 8 }) => ir::types::F64,
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, width: 4 }) => ir::types::I32,
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 }) => ir::types::I32,
        _ => unimplemented!(),
    }
}
