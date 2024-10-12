use std::collections::HashMap;

use codegen::ir;
use cranelift::codegen::ir::types as types;
use cranelift::prelude::*;
use cranelift::codegen::ir::{AbiParam, Block, UserFuncName, Function, InstBuilder, Signature, Value};
use cranelift::codegen::isa::CallConv;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use naga::ResourceBinding;
use crate::context::{FunctionContext, ModuleContext};
use crate::types::translate_primitive_type;


#[derive(Debug)]
enum ExprRepr {
    Constant(Value, Option<ir::Type>),
    Scalars(Vec<Value>, Option<ir::Type>),
    Vectors(Vec<Value>),
}

impl ExprRepr {
    fn into_scalars(self, func_context: &FunctionContext, builder: &mut FunctionBuilder<'_>) -> Vec<Value> {
        let config = func_context.module.config;

        match self {
            ExprRepr::Constant(value, _) => vec![value; config.subgroup_width],
            ExprRepr::Scalars(scalars, _) => scalars,
            ExprRepr::Vectors(vectors) => {
                let vector_count = vectors.len();
                let lane_count = config.subgroup_width / vector_count;

                (0..config.subgroup_width).map(|item_index| {
                    builder.ins().extractlane(vectors[item_index / lane_count], (item_index % lane_count) as u8)
                }).collect::<Vec<_>>()
            },
        }
    }

    fn into_vectors(self, func_context: &FunctionContext, builder: &mut FunctionBuilder<'_>) -> Vec<Value> {
        let config = func_context.module.config;

        match self {
            ExprRepr::Constant(value, Some(ty)) => vec![builder.ins().splat(ty, value); config.subgroup_width],
            ExprRepr::Scalars(scalars, Some(ty)) => {
                let vector_count = scalars.len();
                let lane_count = config.subgroup_width / vector_count;

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


fn translate_expr(
    func_context: &FunctionContext,
    builder: &mut FunctionBuilder<'_>,
    block: Block,
    expr: &naga::Expression,
    expr_info: &naga::valid::ExpressionInfo,
) -> ExprRepr {
    let module_context = func_context.module;
    // let expr_type = match &expr_info.ty {
    //     TypeResolution::Handle(handle) => &module.types[*handle].inner,
    //     TypeResolution::Value(type_value) => &type_value,
    // };

    let expr_type = module_context.resolve_inner_type(expr_info);
    let item_size = module_context.get_type_item_size(expr_type);
    let (lane_count, vector_count) = module_context.config.compute_sizes(item_size);

    match expr {
        naga::Expression::As { expr, kind, convert } => {
            let values_raw = translate_expr(func_context, builder, block, &func_context.function.expressions[*expr], &func_context.function_info[*expr]); //.into_scalars(func_context, builder);
            let input_type = module_context.resolve_inner_type(&func_context.function_info[*expr]);

            match (input_type, expr_type) {
                (
                    naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 }),
                    naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, width: 4 })
                )
                    => values_raw,
                _ => unimplemented!(),
            }
        },

/*         naga::Expression::As { expr, kind, convert } => {
            let values = translate_expr(fn_builder, module, block, function, function_info, &function.expressions[*expr], &function_info[*expr], arg_offsets, global_var_map, config);

            let input_type = resolve_inner_type(module, &function_info[*expr]);
            let input_item_size = get_type_item_size(input_type);

            match input_type {
                naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: _ }) => {
                    match (input_item_size, convert, lane_count) {
                        (4, Some(8), 2) => {
                            (0..(vector_count / 2)).flat_map(|input_vector_index| {
                                let mem_flags = ir::MemFlags::new().with_endianness(ir::Endianness::Little);

                                let output_low = fn_builder.ins().fvpromote_low(values[input_vector_index]);

                                let swizzle_indices_const = fn_builder.func.dfg.constants.insert(
                                    [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0].iter().copied().collect::<_>(),
                                );

                                let swizzle_indices = fn_builder.ins().vconst(types::I8X16, swizzle_indices_const);
                                let byte_vector = fn_builder.ins().bitcast(types::I8X16, mem_flags, values[input_vector_index]);
                                let swizzled_byte_vector = fn_builder.ins().swizzle(byte_vector, swizzle_indices);

                                let swizzled = fn_builder.ins().bitcast(types::F32X4, mem_flags, swizzled_byte_vector);
                                let output_high = fn_builder.ins().fvpromote_low(swizzled);

                                vec![output_low, output_high]
                            }).collect::<Vec<_>>()
                        },
                        (8, Some(4), 4) => {
                            todo!()
                        },
                        _ => unimplemented!(),
                    }
                },
                _ => todo!(),
            }
        }, */
        naga::Expression::Literal(literal) => {
            match literal {
                naga::Literal::F32(v) => {
                    let item = builder.ins().f32const(*v);
                    ExprRepr::Constant(item, Some(types::F32X4))
                },
                naga::Literal::F64(v) => {
                    let item = builder.ins().f64const(*v);
                    ExprRepr::Constant(item, Some(types::F64X2))
                },
                naga::Literal::I32(v) => {
                    let item = builder.ins().iconst(types::I32, *v as i64);
                    ExprRepr::Constant(item, Some(types::I32X4))
                },
                naga::Literal::U32(v) => {
                    let item = builder.ins().iconst(types::I32, *v as i64);
                    ExprRepr::Constant(item, Some(types::I32X4))
                },
                _ => {
                    panic!("unimplemented: {:?}", literal);
                },
            }
        },
        naga::Expression::FunctionArgument(arg) => {
            match func_context.arguments[*arg as usize] {
                Argument::BuiltIn(builtin) => {
                    let builtins_pointer = builder.block_params(block)[0];

                    match builtin {
                        naga::BuiltIn::LocalInvocationIndex => {
                            let subgroup_index = builder.ins().load(types::I32, ir::MemFlags::new(), builtins_pointer, 0);
                            let first_subgroup_thread_index = builder.ins().imul_imm(subgroup_index, module_context.config.subgroup_width as i64);

                            ExprRepr::Scalars(
                                (0..func_context.module.config.subgroup_width)
                                    .map(|index| {
                                        builder.ins().iadd_imm(first_subgroup_thread_index, index as i64)
                                    })
                                    .collect::<Vec<_>>(),
                                Some(types::I32X4),
                            )
                        },
                        naga::BuiltIn::SubgroupId => {
                            ExprRepr::Constant(
                                builder.ins().load(types::I32, ir::MemFlags::new(), builtins_pointer, 0),
                                Some(types::I32X4),
                            )
                        },
                        _ => unreachable!(),
                    }
                },
                Argument::Regular(offset) => {
                    let scalar = builder.block_params(block)[offset];
                    ExprRepr::Scalars(vec![scalar; vector_count], None)
                },
            }
        },
        naga::Expression::Binary { op, left, right } => {
            let left_values = translate_expr(func_context, builder, block, &func_context.function.expressions[*left], &func_context.function_info[*left]).into_vectors(func_context, builder);
            let right_values = translate_expr(func_context, builder, block, &func_context.function.expressions[*right], &func_context.function_info[*right]).into_vectors(func_context, builder);

            ExprRepr::Vectors(
                (0..vector_count).map(|vector_index| {
                    let ins = builder.ins();
                    let left = left_values[vector_index];
                    let right = right_values[vector_index];

                    match op {
                        naga::BinaryOperator::Add => ins.fadd(left, right),
                        naga::BinaryOperator::Subtract => ins.fsub(left, right),
                        naga::BinaryOperator::Multiply => ins.fmul(left, right),
                        naga::BinaryOperator::Divide => ins.fdiv(left, right),
                        _ => unimplemented!(),
                    }
                }).collect::<Vec<_>>()
            )
        },
/*         naga::Expression::Math { fun, arg, arg1, arg2, arg3 } => {
            let args = [Some(*arg), *arg1, *arg2, *arg3].iter().filter_map(|x| x.and_then(|handle| {
                Some(translate_expr(fn_builder, block, function, function_info, &function.expressions[handle], &function_info[handle]))
            })).collect::<Vec<_>>();

            let ins = fn_builder.ins();

            match fun {
                naga::MathFunction::Sqrt => {
                    assert!(args.len() == 1);
                    ins.sqrt(args[0])
                },
                _ => unimplemented!(),
            }
        }, */
/*         naga::Expression::Load { pointer: pointer_handle } => {
            let pointer = translate_expr(fn_builder, module, block, function, function_info, &function.expressions[*pointer_handle], &function_info[*pointer_handle], arg_offsets, global_var_map, config);
            let pointer_scalars = pointer.into_scalars(fn_builder, config);

            // let pointer_offset = match pointer {
            //     ExprRepr::Constant(value, _) => value,
            //     _ => unreachable!(),
            // };

            ExprRepr::Scalars(
                pointer_scalars.iter().map(|pointer_offset| {
                    fn_builder.ins().load(types::F32, ir::MemFlags::new(), *pointer_offset, 0)
                }).collect::<Vec<_>>(),
                types::F32X4,
            )
        }, */
        naga::Expression::Access { base: base_handle, index: index_handle } => {
            let base_value = translate_expr(func_context, builder, block, &func_context.function.expressions[*base_handle], &func_context.function_info[*base_handle]).into_scalars(func_context, builder);
            let index_value = translate_expr(func_context, builder, block, &func_context.function.expressions[*index_handle], &func_context.function_info[*index_handle]).into_scalars(func_context, builder);

            let base_pointer_type = module_context.resolve_inner_type(&func_context.function_info[*base_handle]);
            let base_type_handle = if let naga::TypeInner::Pointer { base, .. } = base_pointer_type { base } else { unreachable!() };
            let base_type = &module_context.module.types[*base_type_handle].inner;

            ExprRepr::Scalars(
                base_value
                    .iter()
                    .zip(index_value.iter())
                    .map(|(base_scalar, index_scalar)| {
                        let index_scalar_i64 = builder.ins().uextend(types::I64, *index_scalar);

                        let offset = match base_type {
                            naga::TypeInner::Array { stride, .. } => builder.ins().imul_imm(index_scalar_i64, *stride as i64),
                            _ => todo!(),
                        };

                        builder.ins().iadd(*base_scalar, offset)
                    })
                    .collect::<Vec<_>>(),
                None
            )
        },
        naga::Expression::AccessIndex { base: base_handle, index } => {
            let base_value = translate_expr(func_context, builder, block, &func_context.function.expressions[*base_handle], &func_context.function_info[*base_handle]).into_scalars(func_context, builder);

            let base_pointer_type = module_context.resolve_inner_type(&func_context.function_info[*base_handle]);
            let base_type_handle = if let naga::TypeInner::Pointer { base, .. } = base_pointer_type { base } else { unreachable!() };
            let base_type = &module_context.module.types[*base_type_handle].inner;

            ExprRepr::Scalars(
                base_value
                    .iter()
                    .map(|base_scalar| {
                        let offset = match base_type {
                            naga::TypeInner::Array { stride, .. } => (stride * index),
                            naga::TypeInner::Struct { members, .. } => members[*index as usize].offset,
                            _ => todo!(),
                        };

                        builder.ins().iadd_imm(*base_scalar, offset as i64)
                    })
                    .collect::<Vec<_>>(),
                None
            )
        },
        naga::Expression::Constant(handle) => {
            let constants_pointer = builder.ins().global_value(module_context.pointer_type, func_context.constants_global_value);
            let constant_index = module_context.constant_map[handle];

            let result_type = translate_primitive_type(expr_type);

            ExprRepr::Constant(
                builder.ins().load(
                    result_type,
                    ir::MemFlags::new(),
                    constants_pointer,
                    (constant_index as i32) * (module_context.pointer_type.bytes() as i32)
                ),
                Some(result_type),
            )
        },
        naga::Expression::GlobalVariable(global_var_handle) => {
            let global_var_index = module_context.global_var_map[global_var_handle];
            let global_vars_pointer = builder.block_params(block)[1];

            ExprRepr::Constant(builder.ins().load(
                module_context.pointer_type,
                ir::MemFlags::new(),
                global_vars_pointer,
                (global_var_index as i32) * (module_context.pointer_type.bytes() as i32)
            ), Some(types::F32X4))
        },
        _ => {
            panic!("unimplemented: {:?}", expr);
        },
    }
}


pub(crate) enum Argument {
    BuiltIn(naga::BuiltIn),
    Regular(usize),
}

pub fn translate_func(
    module_context: &ModuleContext<'_, '_>,
    function: &naga::Function,
    function_info: &naga::valid::FunctionInfo,
) -> (cranelift::codegen::ir::Function, Signature) {
    // General

    let mut cl_signature = Signature::new(CallConv::Fast);


    // Parameters

    // Pointer to builtins
    cl_signature.params.push(AbiParam::new(module_context.pointer_type));

    // Pointer to global variables
    cl_signature.params.push(AbiParam::new(module_context.pointer_type));


    let mut arguments = Vec::with_capacity(function.arguments.len());

    for arg in function.arguments.iter() {
        let shader_type = &module_context.module.types[arg.ty];
        let (cl_type, cl_type_count) = module_context.translate_type(shader_type);

        match arg.binding {
            Some(naga::Binding::BuiltIn(built_in)) => {
                arguments.push(Argument::BuiltIn(built_in));
            },
            Some(_) => unreachable!(),
            None => {
                arguments.push(Argument::Regular(cl_signature.params.len()));
                cl_signature.params.push(AbiParam::new(cl_type));
            },
        }
    }


    // Return values

    if let Some(result) = &function.result {
        let shader_type = &module_context.module.types[result.ty];
        let (cl_type, cl_type_count) = module_context.translate_type(shader_type);

        for _ in 0..cl_type_count {
            cl_signature.returns.push(AbiParam::new(cl_type));
        }
    }

    // cl_signature.params.push(AbiParam::new(module_context.pointer_type));


    // Builder

    let mut cl_fn_builder_ctx = FunctionBuilderContext::new();
    let mut cl_func = Function::with_name_signature(UserFuncName::user(0, 0), cl_signature.clone());

    let constants_global_value = module_context.cl_module.declare_data_in_func(module_context.constants_data_id, &mut cl_func);

    let mut builder = FunctionBuilder::new(&mut cl_func, &mut cl_fn_builder_ctx);
    let block = builder.create_block();

    builder.append_block_params_for_function_params(block);
    builder.switch_to_block(block);


    let func_context = FunctionContext {
        arguments: &arguments,
        constants_global_value,
        module: module_context,
        function,
        function_info,
    };

    // eprintln!("{:#?}", function);

    for stat in function.body.iter() {
        match stat {
            naga::Statement::Store { pointer: pointer_handle, value: value_handle } => {
                let pointer = translate_expr(&func_context, &mut builder, block, &function.expressions[*pointer_handle], &function_info[*pointer_handle]);
                let value = translate_expr(&func_context, &mut builder, block, &function.expressions[*value_handle], &function_info[*value_handle]);

                // eprintln!("Pointer = {:#?}", pointer);
                // eprintln!("Value = {:#?}", value);

                let pointer_scalars = pointer.into_scalars(&func_context, &mut builder);
                let value_scalars = value.into_scalars(&func_context, &mut builder);

                for (pointer_scalar, value_scalar) in pointer_scalars.iter().zip(value_scalars.iter()) {
                    builder.ins().store(ir::MemFlags::new(), *value_scalar, *pointer_scalar, 0);
                }
            },
            // naga::Statement::Return { value } => {
            //     let expr = &function.expressions[value.unwrap()];
            //     let expr_info = &function_info[value.unwrap()];
            //     let return_values = translate_expr(&mut fn_builder, module, block, function, function_info, expr, expr_info, &arg_offsets, global_var_map, config);

            //     fn_builder.ins().return_(&return_values);
            // },
            _ => {},
        }
    }

    builder.ins().return_(&[]);

    // builder.seal_block(block);

    builder.seal_all_blocks();
    builder.finalize();

    eprintln!("-- OUTPUT --------------------------------------------------------\n{:#?}", cl_func);

    (cl_func, cl_signature)
}
