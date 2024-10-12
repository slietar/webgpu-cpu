use std::collections::HashMap;

use crate::config::Config;
use naga::proc::TypeResolution;
// use cranelift::prelude::Configurable;
use cranelift::codegen::ir;


#[derive(Debug)]
pub(crate) struct ModuleContext<'a> {
    pub config: &'a Config,
    pub global_var_map: &'a HashMap<naga::Handle<naga::GlobalVariable>, usize>,
    pub module_info: &'a naga::valid::ModuleInfo,
    pub module: &'a naga::Module,
    pub pointer_type: ir::types::Type,
}

impl<'a> ModuleContext<'a> {
    pub fn translate_type(&self, shader_type: &naga::Type) -> (ir::types::Type, usize /* vector count */) {
        let item_size = get_type_item_size(&shader_type.inner);

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

pub fn get_type_item_size(shader_type: &naga::TypeInner) -> usize {
    match shader_type {
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width }) => *width as usize,
        naga::TypeInner::Pointer { base, space } => 8,
        _ => unimplemented!(),
    }
}


pub(crate) struct FunctionContext<'a> {
    pub arg_offsets: &'a [usize],
    pub builder: &'a mut cranelift::frontend::FunctionBuilder<'a>,
    pub function_info: &'a naga::valid::FunctionInfo,
    pub function: &'a naga::Function,
    pub module: &'a ModuleContext<'a>,
}
