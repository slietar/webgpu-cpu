use cranelift::codegen::ir;
use cranelift::codegen::ir::types as types;
use cranelift::codegen::ir::{Block, InstBuilder};
use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::IntCC;
use crate::context::FunctionContext;
use crate::translate::Argument;
use crate::types::translate_primitive_type;
use crate::repr::ExprRepr;


pub(crate) fn translate_expr(
    func_context: &FunctionContext,
    builder: &mut FunctionBuilder<'_>,
    block: Block,
    expr: &naga::Expression,
    expr_info: &naga::valid::ExpressionInfo,
) -> ExprRepr {
    let module_context = func_context.module;

    let expr_type = module_context.resolve_inner_type(expr_info);
    let item_size = module_context.get_type_item_size(expr_type);

    let (lane_count, vector_count) = module_context.config.compute_sizes(item_size);

    // eprintln!("Expr type {:?}", expr_type);
    // eprintln!("Item size {:?}", item_size);
    // eprintln!("Lane count {:?}", lane_count);
    // eprintln!("Vector count {:?}", vector_count);
    // eprintln!("");

    match expr {
        naga::Expression::As { expr: expr_handle, kind, convert } => {
            let values_raw = func_context.get_expr(*expr_handle, builder, block);
            let input_type = module_context.resolve_inner_type(&func_context.function_info[*expr_handle]);

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
                            let first_subgroup_thread_index = builder.ins().imul_imm(subgroup_index, module_context.config.simul_thread_count as i64);

                            ExprRepr::Scalars(
                                (0..func_context.module.config.simul_thread_count)
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
        naga::Expression::Binary { op, left: left_handle, right: right_handle } => {
            let left_values = func_context.get_expr(*left_handle, builder, block).into_vectors(func_context, builder);
            let right_values = func_context.get_expr(*right_handle, builder, block).into_vectors(func_context, builder);

            ExprRepr::Vectors(
                (0..vector_count).map(|vector_index| {
                    let ins = builder.ins();
                    let left = left_values[vector_index];
                    let right = right_values[vector_index];

                    match expr_type {
                        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Bool, .. }) => {
                            match op {
                                naga::BinaryOperator::Greater => ins.icmp(IntCC::UnsignedGreaterThan, left, right),
                                naga::BinaryOperator::GreaterEqual => ins.icmp(IntCC::UnsignedGreaterThan, left, right),
                                naga::BinaryOperator::Less => ins.icmp(IntCC::UnsignedLessThan, left, right),
                                naga::BinaryOperator::LessEqual => ins.icmp(IntCC::UnsignedLessThanOrEqual, left, right),
                                naga::BinaryOperator::Equal => ins.icmp(IntCC::Equal, left, right),
                                naga::BinaryOperator::NotEqual => ins.icmp(IntCC::NotEqual, left, right),
                                naga::BinaryOperator::LogicalAnd => ins.band(left, right),
                                naga::BinaryOperator::LogicalOr => ins.bor(left, right),
                                _ => unimplemented!("{:?}", op),
                            }
                        },
                        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, .. }) => {
                            match op {
                                naga::BinaryOperator::Add => ins.fadd(left, right),
                                naga::BinaryOperator::Subtract => ins.fsub(left, right),
                                naga::BinaryOperator::Multiply => ins.fmul(left, right),
                                naga::BinaryOperator::Divide => ins.fdiv(left, right),
                                _ => unimplemented!(),
                            }
                        },
                        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, .. }) => {
                            match op {
                                naga::BinaryOperator::Add => ins.iadd(left, right),
                                naga::BinaryOperator::Subtract => ins.isub(left, right),
                                naga::BinaryOperator::Multiply => ins.imul(left, right),
                                // naga::BinaryOperator::Divide => ins.sdiv(left, right),
                                _ => unimplemented!(),
                            }
                        },
                        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, .. }) => {
                            match op {
                                naga::BinaryOperator::Add => ins.uadd_sat(left, right),
                                naga::BinaryOperator::Subtract => ins.usub_sat(left, right),
                                naga::BinaryOperator::Multiply => ins.umulhi(left, right),
                                naga::BinaryOperator::Divide => ins.udiv(left, right),
                                _ => unimplemented!(),
                            }
                        },
                        _ => unimplemented!("{:?}", expr_type),
                    }
                }).collect::<Vec<_>>()
            )
        },
        naga::Expression::Unary { op, expr: handle } => {
            let values = func_context.get_expr(*handle, builder, block).into_vectors(func_context, builder);

            ExprRepr::Vectors(
                values
                    .iter()
                    .map(|value| {
                        let ins = builder.ins();

                        match expr_type {
                            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Bool, .. }) => {
                                match op {
                                    naga::UnaryOperator::LogicalNot => ins.bnot(*value),
                                    _ => unimplemented!(),
                                }
                            },
                            _ => unimplemented!(),
                        }
                    })
                    .collect::<Vec<_>>()
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
        naga::Expression::Load { pointer: pointer_handle } => {
            let pointer = func_context.get_expr(*pointer_handle, builder, block).into_scalars(func_context, builder);

            ExprRepr::Scalars(
                pointer
                    .iter()
                    .map(|item_pointer| {
                        builder.ins().load(translate_primitive_type(expr_type), ir::MemFlags::new(), *item_pointer, 0)
                    })
                    .collect::<Vec<_>>(),
                Some(module_context.translate_type(expr_type).0),
            )
        },
        naga::Expression::Access { base: base_handle, index: index_handle } => {
            let base_value = func_context.get_expr(*base_handle, builder, block).into_scalars(func_context, builder);
            let index_value = func_context.get_expr(*index_handle, builder, block).into_scalars(func_context, builder);

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
            let base_value = func_context.get_expr(*base_handle, builder, block).into_scalars(func_context, builder);

            let base_pointer_type = module_context.resolve_inner_type(&func_context.function_info[*base_handle]);
            let base_type_handle = if let naga::TypeInner::Pointer { base, .. } = base_pointer_type { base } else { unreachable!() };
            let base_type = &module_context.module.types[*base_type_handle].inner;

            ExprRepr::Scalars(
                base_value
                    .iter()
                    .map(|base_scalar| {
                        let offset = match base_type {
                            naga::TypeInner::Array { stride, .. } => stride * index,
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

            ExprRepr::Constant(
                builder.ins().load(
                    translate_primitive_type(expr_type),
                    ir::MemFlags::new(),
                    constants_pointer,
                    (constant_index as i32) * (module_context.pointer_type.bytes() as i32)
                ),
                Some(module_context.translate_type(expr_type).0),
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
        naga::Expression::LocalVariable(handle) => {
            let slot = func_context.local_var_slot.unwrap();
            let offset = func_context.local_var_offsets[handle];

            ExprRepr::Constant(
                builder.ins().stack_addr(module_context.pointer_type, slot, offset as i32),
                None,
            )
        },
        _ => {
            panic!("unimplemented: {:?}", expr);
        },
    }
}
