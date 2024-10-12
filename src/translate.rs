use std::collections::HashMap;

use codegen::ir;
use cranelift::codegen::ir::types as types;
use cranelift::prelude::*;
use cranelift::codegen::ir::{AbiParam, Block, UserFuncName, Function, InstBuilder, Signature, Value};
use cranelift::codegen::isa::CallConv;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use naga::ResourceBinding;
use crate::context::{FunctionContext, ModuleContext};


#[derive(Debug)]
enum ExprRepr {
    Constant(Value, ir::Type),
    Scalars(Vec<Value>, ir::Type),
    Vectors(Vec<Value>),
}

impl ExprRepr {
    fn into_scalars(self, func_context: &mut FunctionContext) -> Vec<Value> {
        let config = func_context.module.config;

        match self {
            ExprRepr::Constant(value, _) => vec![value; config.subgroup_width],
            ExprRepr::Scalars(scalars, _) => scalars,
            ExprRepr::Vectors(vectors) => {
                let vector_count = vectors.len();
                let lane_count = config.subgroup_width / vector_count;

                (0..config.subgroup_width).map(|item_index| {
                    func_context.builder.ins().extractlane(vectors[item_index / lane_count], (item_index % lane_count) as u8)
                }).collect::<Vec<_>>()
            },
        }
    }

    fn into_vectors(self, func_context: &mut FunctionContext) -> Vec<Value> {
        let config = func_context.module.config;

        match self {
            ExprRepr::Constant(value, ty) => vec![func_context.builder.ins().splat(ty, value); config.subgroup_width],
            ExprRepr::Scalars(scalars, ty) => {
                let vector_count = scalars.len();
                let lane_count = config.subgroup_width / vector_count;

                (0..lane_count).map(|lane_index| {
                    let init_vector = func_context.builder.ins().scalar_to_vector(ty, scalars[0]);

                    scalars.iter().copied().enumerate().skip(1).fold(init_vector, |acc, (scalar_index, scalar)| {
                        func_context.builder.ins().insertlane(acc, scalar, scalar_index as u8)
                    })
                }).collect::<Vec<_>>()
            },
            ExprRepr::Vectors(vectors) => vectors,
        }
    }
}



fn translate_expr(
    func_context: &mut FunctionContext,
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
    let item_size = crate::context::get_type_item_size(expr_type);
    let (lane_count, vector_count) = module_context.config.compute_sizes(item_size);

    match expr {
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
                    let item = func_context.builder.ins().f32const(*v);
                    ExprRepr::Constant(item, types::F32X4)
                },
                naga::Literal::F64(v) => {
                    let item = func_context.builder.ins().f64const(*v);
                    ExprRepr::Constant(item, types::F64X2)

                    // let vector = fn_builder.ins().splat(types::F64X2, item);
                    // vec![vector; vector_count]
                },
                _ => {
                    panic!("unimplemented: {:?}", literal);
                },
            }
        },
        naga::Expression::FunctionArgument(arg) => {
            let offset = func_context.arg_offsets[*arg as usize];

            ExprRepr::Vectors(func_context.builder.block_params(block)[offset..(offset + vector_count)].to_vec())
        },
        naga::Expression::Binary { op, left, right } => {
            let left_values = translate_expr(func_context, block, &func_context.function.expressions[*left], &func_context.function_info[*left]).into_vectors(func_context);
            let right_values = translate_expr(func_context, block, &func_context.function.expressions[*right], &func_context.function_info[*right]).into_vectors(func_context);

            ExprRepr::Vectors(
                (0..vector_count).map(|vector_index| {
                    let ins = func_context.builder.ins();
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
        },
        naga::Expression::AccessIndex { base: base_handle, index } => {
            // let base = &function.expressions[*base_handle];
            // let base_info = &function_info[*base_handle];

            let base_value = match function.expressions[*base_handle] {
                naga::Expression::GlobalVariable(global_var_handle) => {
                    let offset = global_var_map[&global_var_handle];
                    let value = fn_builder.block_params(block)[offset];

                    value
                },
                _ => todo!(),
            };

            // let values = translate_expr(fn_builder, module, block, function, function_info, base, base_info, arg_offsets, global_var_map, config);
            // eprintln!("> {:#?}", values);
            // eprintln!("> {:#?}", index);

            ExprRepr::Constant(fn_builder.ins().iadd_imm(base_value, *index as i64), types::F32X4)
        }, */
        // naga::Expression::GlobalVariable(handle) => {
        //     let offset = global_var_map[handle];
        //     // let global_var = module.global_variables.try_get(*handle).unwrap();

        //     // global_var.ty
        //     // let expr_type = resolve_inner_type(module, expr_info);
        //     // let item_size = get_type_item_size(expr_type);
        //     // let (lane_count, vector_count) = compute_sizes(item_size, config);

        //     ExprRepr::Constant(fn_builder.block_params(block)[offset], POINTER_TYPE)
        //     // ExprRepr::Vectors(fn_builder.block_params(block)[offset..(offset + vector_count)].to_vec())
        // },
        _ => {
            panic!("unimplemented: {:?}", expr);
        },
    }
}

fn translate_func(
    module_context: &ModuleContext,
    function: &naga::Function,
    function_info: &naga::valid::FunctionInfo,
) -> (cranelift::codegen::ir::Function, Signature) {
    let mut cl_signature = Signature::new(CallConv::Fast);

    if let Some(result) = &function.result {
        let shader_type = &module_context.module.types[result.ty];
        let (cl_type, cl_type_count) = module_context.translate_type(shader_type);

        for _ in 0..cl_type_count {
            cl_signature.returns.push(AbiParam::new(cl_type));
        }
    }

    let mut arg_offsets = Vec::new();

    for arg in function.arguments.iter() {
        let shader_type = &module_context.module.types[arg.ty];
        let (cl_type, cl_type_count) = module_context.translate_type(shader_type);

        arg_offsets.push(cl_signature.params.len());

        for _ in 0..cl_type_count {
            cl_signature.params.push(AbiParam::new(cl_type));
        }
    }

    cl_signature.params.push(AbiParam::new(module_context.pointer_type));

    let mut cl_fn_builder_ctx = FunctionBuilderContext::new();
    let mut cl_func = Function::with_name_signature(UserFuncName::user(0, 0), cl_signature.clone());

    let mut fn_builder = FunctionBuilder::new(&mut cl_func, &mut cl_fn_builder_ctx);
    let block = fn_builder.create_block();

    fn_builder.append_block_params_for_function_params(block);
    fn_builder.switch_to_block(block);

    // eprintln!("{:#?}", function);

    for stat in function.body.iter() {
        match stat {
/*             naga::Statement::Store { pointer: pointer_handle, value: value_handle } => {
                let pointer = translate_expr(&mut fn_builder, module, block, function, function_info, &function.expressions[*pointer_handle], &function_info[*pointer_handle], &arg_offsets, &global_var_map, config);
                let value = translate_expr(&mut fn_builder, module, block, function, function_info, &function.expressions[*value_handle], &function_info[*value_handle], &arg_offsets, &global_var_map, config);

                // eprintln!("Pointer = {:#?}", pointer);
                // eprintln!("Value = {:#?}", value);

                let pointer_scalars = pointer.into_scalars(&mut fn_builder, config);
                let value_scalars = value.into_scalars(&mut fn_builder, config);

                for (pointer_scalar, value_scalar) in pointer_scalars.iter().zip(value_scalars.iter()) {
                    fn_builder.ins().store(ir::MemFlags::new(), *value_scalar, *pointer_scalar, 0);
                }
            }, */
            // naga::Statement::Return { value } => {
            //     let expr = &function.expressions[value.unwrap()];
            //     let expr_info = &function_info[value.unwrap()];
            //     let return_values = translate_expr(&mut fn_builder, module, block, function, function_info, expr, expr_info, &arg_offsets, global_var_map, config);

            //     fn_builder.ins().return_(&return_values);
            // },
            _ => {},
        }
    }

    fn_builder.ins().return_(&[]);

    // fn_builder.seal_block(block);

    fn_builder.seal_all_blocks();
    fn_builder.finalize();

    eprintln!("-- OUTPUT --------------------------------------------------------\n{:#?}", cl_func);

    (cl_func, cl_signature)
}