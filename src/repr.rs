use cranelift::codegen::ir;
use cranelift::codegen::ir::{InstBuilder, Value};
use cranelift::frontend::FunctionBuilder;
use crate::context::FunctionContext;


#[derive(Clone, Debug)]
pub enum ExprRepr {
    Constant(Value, Option<ir::Type>),
    Scalars(Vec<Value>, Option<ir::Type>),
    Vectors(Vec<Value>),
}

impl ExprRepr {
    pub fn into_scalars(self, func_context: &FunctionContext, builder: &mut FunctionBuilder<'_>) -> Vec<Value> {
        let config = func_context.module.config;

        match self {
            ExprRepr::Constant(value, _) => vec![value; config.simul_thread_count],
            ExprRepr::Scalars(scalars, _) => scalars,
            ExprRepr::Vectors(vectors) => {
                let vector_count = vectors.len();
                let lane_count = config.simul_thread_count / vector_count;

                (0..config.simul_thread_count).map(|item_index| {
                    builder.ins().extractlane(vectors[item_index / lane_count], (item_index % lane_count) as u8)
                }).collect::<Vec<_>>()
            },
        }
    }

    pub fn into_vectors(self, func_context: &FunctionContext, builder: &mut FunctionBuilder<'_>) -> Vec<Value> {
        let config = func_context.module.config;

        match self {
            ExprRepr::Constant(value, Some(ty)) => vec![builder.ins().splat(ty, value); config.simul_thread_count],
            ExprRepr::Scalars(scalars, Some(ty)) => {
                let vector_count = scalars.len();
                let lane_count = config.simul_thread_count / vector_count;

                (0..lane_count).map(|lane_index| {
                    let init_vector = builder.ins().scalar_to_vector(ty, scalars[0]);

                    scalars.iter().copied().enumerate().skip(1).fold(init_vector, |acc, (scalar_index, scalar)| {
                        builder.ins().insertlane(acc, scalar, scalar_index as u8)
                    })
                }).collect::<Vec<_>>()
            },
            ExprRepr::Vectors(vectors) => vectors,
            _ => unreachable!(),
        }
    }
}
