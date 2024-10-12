use std::collections::HashMap;

use crate::config::Config;
use naga::proc::TypeResolution;
// use cranelift::prelude::Configurable;
use cranelift::codegen::ir;


pub(crate) struct ModuleContext<'a, 'b> {
    pub config: &'a Config,
    pub constant_map: &'a HashMap<naga::Handle<naga::Constant>, usize>,
    pub constants_data_id: cranelift::module::DataId,
    pub cl_module: &'b mut dyn cranelift::module::Module,
    pub global_var_map: &'a HashMap<naga::Handle<naga::GlobalVariable>, usize>,
    pub layouter: &'a naga::proc::Layouter,
    pub module_info: &'a naga::valid::ModuleInfo,
    pub module: &'a naga::Module,
    pub pointer_type: ir::types::Type,
}

impl<'a> ModuleContext<'a, '_> {
    pub fn get_type_item_size(&self, shader_type: &naga::TypeInner) -> usize {
        match shader_type {
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width }) => *width as usize,
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, width }) => *width as usize,
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width }) => *width as usize,
            naga::TypeInner::Pointer { .. } => self.pointer_type.bytes() as usize,
            naga::TypeInner::Struct { span, .. } => *span as usize,
            _ => panic!("Unsupported type {:?}", shader_type),
        }
    }

    pub fn translate_type(&self, shader_type: &naga::Type) -> (ir::types::Type, usize /* vector count */) {
        let item_size = self.get_type_item_size(&shader_type.inner);

        let (lane_count, vector_count) = self.config.compute_sizes(item_size);

        let cl_type = match (&shader_type.inner, lane_count) {
            (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }), 4) => ir::types::F32X4,
            (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 8 }), 2) => ir::types::F64X2,
            (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, width: 4 }), 4) => ir::types::I32X4,
            (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 }), 4) => ir::types::I32X4,
            _ => unimplemented!(),
        };

        (cl_type, vector_count)
    }

    pub fn resolve_inner_type(&self, expr_info: &'a naga::valid::ExpressionInfo) -> &'a naga::TypeInner {
        match &expr_info.ty {
            TypeResolution::Handle(handle) => &self.module.types[*handle].inner,
            TypeResolution::Value(type_value) => &type_value,
        }
    }
}


pub(crate) struct FunctionContext<'a, 'b> {
    pub arguments: &'a [crate::translate::Argument],
    // pub builder: &'a mut cranelift::frontend::FunctionBuilder<'a>,
    pub constants_global_value: ir::GlobalValue,
    pub function_info: &'a naga::valid::FunctionInfo,
    pub function: &'a naga::Function,
    pub module: &'a ModuleContext<'a, 'b>,
}
