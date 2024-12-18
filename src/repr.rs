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
            _ => unreachable!("{:?}", self),
        }
    }

    pub fn and(&self, other: &ExprRepr, func_context: &FunctionContext, builder: &mut FunctionBuilder<'_>) -> ExprRepr {
        match (self, other) {
            // Expected types are i8
            (ExprRepr::Constant(lhs, _), ExprRepr::Constant(rhs, _))
                => ExprRepr::Constant(builder.ins().band(*lhs, *rhs), Some(ir::types::I32X4)),
            _ => {
                let lhs_vectors = self.clone().into_vectors(func_context, builder);
                let rhs_vector = other.clone().into_vectors(func_context, builder);

                ExprRepr::Vectors(
                    lhs_vectors
                        .into_iter()
                        .zip(rhs_vector.into_iter())
                        .map(|(lhs, rhs)| {
                            builder.ins().band(lhs, rhs)
                        })
                        .collect::<Vec<_>>()
                )
            },
        }
    }

    pub fn not(&self, func_context: &FunctionContext, builder: &mut FunctionBuilder<'_>) -> ExprRepr {
        match self {
            // Expected types are i8
            ExprRepr::Constant(value, _)
                => ExprRepr::Constant(builder.ins().bnot(*value), None),
            _ => {
                let vectors = self.clone().into_vectors(func_context, builder);

                ExprRepr::Vectors(
                    vectors
                        .into_iter()
                        .map(|vector| {
                            builder.ins().bnot(vector)
                        })
                        .collect::<Vec<_>>()
                )
            },
        }
    }
}
