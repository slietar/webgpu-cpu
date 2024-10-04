use std::any::Any;
use std::collections::HashMap;
use std::io::Write as _;

use codegen::ir;
use cranelift::codegen::ir::types as types;
use cranelift::prelude::*;
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::codegen::ir::{AbiParam, Block, UserFuncName, Function, InstBuilder, Signature, Value};
use cranelift::codegen::isa::CallConv;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::module::Module as _;
use naga::proc::TypeResolution;
use naga::ResourceBinding;


const POINTER_TYPE: types::Type = types::I64;


#[derive(Debug)]
struct Config {
    bandwidth_size: usize, // Usually 16
    subgroup_width: usize,
}

#[derive(Debug)]
enum ExprRepr {
    Constant(Value, ir::Type),
    Scalars(Vec<Value>, ir::Type),
    Vectors(Vec<Value>),
}

impl ExprRepr {
    fn into_scalars(self, fn_builder: &mut FunctionBuilder<'_>, config: &Config) -> Vec<Value> {
        match self {
            ExprRepr::Constant(value, _) => vec![value; config.subgroup_width],
            ExprRepr::Scalars(scalars, _) => scalars,
            ExprRepr::Vectors(vectors) => {
                let vector_count = vectors.len();
                let lane_count = config.subgroup_width / vector_count;

                (0..config.subgroup_width).map(|item_index| {
                    fn_builder.ins().extractlane(vectors[item_index / lane_count], (item_index % lane_count) as u8)
                }).collect::<Vec<_>>()
            },
        }
    }

    fn into_vectors(self, fn_builder: &mut FunctionBuilder<'_>, config: &Config) -> Vec<Value> {
        match self {
            ExprRepr::Constant(value, ty) => vec![fn_builder.ins().splat(ty, value); config.subgroup_width],
            ExprRepr::Scalars(scalars, ty) => {
                let vector_count = scalars.len();
                let lane_count = config.subgroup_width / vector_count;

                (0..lane_count).map(|lane_index| {
                    let init_vector = fn_builder.ins().scalar_to_vector(ty, scalars[0]);

                    scalars.iter().copied().enumerate().skip(1).fold(init_vector, |acc, (scalar_index, scalar)| {
                        fn_builder.ins().insertlane(acc, scalar, scalar_index as u8)
                    })
                }).collect::<Vec<_>>()
            },
            ExprRepr::Vectors(vectors) => vectors,
        }
    }
}


fn compute_sizes(item_size: usize, config: &Config) -> (usize, usize) {
    let lane_count = config.bandwidth_size / item_size;
    let vector_count = item_size * config.subgroup_width / config.bandwidth_size;

    (lane_count, vector_count)
}

fn get_type_item_size(shader_type: &naga::TypeInner) -> usize {
    match shader_type {
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width }) => *width as usize,
        naga::TypeInner::Pointer { base, space } => 8,
        _ => unimplemented!(),
    }
}


fn translate_type(shader_type: &naga::Type, config: &Config) -> (types::Type, usize /* vector count */) {
    let item_size = get_type_item_size(&shader_type.inner);

    // total_size = item_size * subgroup_width
    // total_size = bandwidth_size * vector_count
    // bandwidth_size = item_size * lane_count

    // let lane_count = config.bandwidth_size / item_size;
    // let vector_count = item_size * config.subgroup_width / config.bandwidth_size;

    let (lane_count, vector_count) = compute_sizes(item_size, config);

    let cl_type = match (&shader_type.inner, lane_count) {
        (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }), 4) => types::F32X4,
        (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 8 }), 2) => types::F64X2,
        (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, width: 4 }), 4) => types::I32X4,
        (naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 }), 4) => types::I32X4,
        _ => unimplemented!(),
    };

    (cl_type, vector_count)
}


fn resolve_inner_type<'a>(module: &'a naga::Module, expr_info: &'a naga::valid::ExpressionInfo) -> &'a naga::TypeInner {
    match &expr_info.ty {
        TypeResolution::Handle(handle) => &module.types[*handle].inner,
        TypeResolution::Value(type_value) => &type_value,
    }
}


fn translate_expr(
    fn_builder: &mut FunctionBuilder<'_>,
    module: &naga::Module,
    block: Block,
    function: &naga::Function,
    function_info: &naga::valid::FunctionInfo,
    expr: &naga::Expression,
    expr_info: &naga::valid::ExpressionInfo,
    arg_offsets: &[usize],
    global_var_map: &HashMap<naga::Handle<naga::GlobalVariable>, usize>,
    config: &Config,
) -> ExprRepr {
    // let expr_type = match &expr_info.ty {
    //     TypeResolution::Handle(handle) => &module.types[*handle].inner,
    //     TypeResolution::Value(type_value) => &type_value,
    // };

    let expr_type = resolve_inner_type(module, expr_info);
    let item_size = get_type_item_size(expr_type);
    let (lane_count, vector_count) = compute_sizes(item_size, config);

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
                    let item = fn_builder.ins().f32const(*v);
                    ExprRepr::Constant(item, types::F32X4)
                },
                naga::Literal::F64(v) => {
                    let item = fn_builder.ins().f64const(*v);
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
            let offset = arg_offsets[*arg as usize];

            ExprRepr::Vectors(fn_builder.block_params(block)[offset..(offset + vector_count)].to_vec())
        },
        naga::Expression::Binary { op, left, right } => {
            let left_values = translate_expr(fn_builder, module, block, function, function_info, &function.expressions[*left], &function_info[*left], arg_offsets, global_var_map, config).into_vectors(fn_builder, config);
            let right_values = translate_expr(fn_builder, module, block, function, function_info, &function.expressions[*right], &function_info[*right], arg_offsets, global_var_map, config).into_vectors(fn_builder, config);

            ExprRepr::Vectors(
                (0..vector_count).map(|vector_index| {
                    let ins = fn_builder.ins();
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
        naga::Expression::Load { pointer: pointer_handle } => {
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
        },
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

fn translate_func(module: &naga::Module, module_info: &naga::valid::ModuleInfo, function: &naga::Function, function_info: &naga::valid::FunctionInfo, config: &Config) -> (cranelift::codegen::ir::Function, Signature) {
    let mut cl_signature = Signature::new(CallConv::Fast);

    if let Some(result) = &function.result {
        let shader_type = &module.types[result.ty];
        let (cl_type, cl_type_count) = translate_type(shader_type, config);

        for _ in 0..cl_type_count {
            cl_signature.returns.push(AbiParam::new(cl_type));
        }
    }

    let mut arg_offsets = Vec::new();

    for arg in function.arguments.iter() {
        let shader_type = &module.types[arg.ty];
        let (cl_type, cl_type_count) = translate_type(shader_type, config);

        arg_offsets.push(cl_signature.params.len());

        for _ in 0..cl_type_count {
            cl_signature.params.push(AbiParam::new(cl_type));
        }
    }

    let mut sorted_bounded_global_variables = module.global_variables
        .iter()
        .filter_map(|(handle, global_var)| match global_var.binding {
            Some(ResourceBinding { group, binding }) => Some((handle, (group, binding))),
            None => None,
        })
        .collect::<Vec<_>>();

    sorted_bounded_global_variables.sort_by_key(|(_, key)| *key);
    let sorted_bounded_global_variables = sorted_bounded_global_variables.into_iter().map(|(global_var, _)| global_var).collect::<Vec<_>>();

    let mut global_var_map = HashMap::new();

    for handle in sorted_bounded_global_variables {
        global_var_map.insert(handle, cl_signature.params.len());
        cl_signature.params.push(AbiParam::new(POINTER_TYPE));
    }

    let mut cl_fn_builder_ctx = FunctionBuilderContext::new();
    let mut cl_func = Function::with_name_signature(UserFuncName::user(0, 0), cl_signature.clone());

    let mut fn_builder = FunctionBuilder::new(&mut cl_func, &mut cl_fn_builder_ctx);
    let block = fn_builder.create_block();

    fn_builder.append_block_params_for_function_params(block);
    fn_builder.switch_to_block(block);

    // eprintln!("{:#?}", function);

    for stat in function.body.iter() {
        match stat {
            naga::Statement::Store { pointer: pointer_handle, value: value_handle } => {
                let pointer = translate_expr(&mut fn_builder, module, block, function, function_info, &function.expressions[*pointer_handle], &function_info[*pointer_handle], &arg_offsets, &global_var_map, config);
                let value = translate_expr(&mut fn_builder, module, block, function, function_info, &function.expressions[*value_handle], &function_info[*value_handle], &arg_offsets, &global_var_map, config);

                // eprintln!("Pointer = {:#?}", pointer);
                // eprintln!("Value = {:#?}", value);

                let pointer_scalars = pointer.into_scalars(&mut fn_builder, config);
                let value_scalars = value.into_scalars(&mut fn_builder, config);

                for (pointer_scalar, value_scalar) in pointer_scalars.iter().zip(value_scalars.iter()) {
                    fn_builder.ins().store(ir::MemFlags::new(), *value_scalar, *pointer_scalar, 0);
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

    fn_builder.ins().return_(&[]);

    // fn_builder.seal_block(block);

    fn_builder.seal_all_blocks();
    fn_builder.finalize();

    eprintln!("-- OUTPUT --------------------------------------------------------\n{:#?}", cl_func);

    (cl_func, cl_signature)
}


fn main() {
    let config = Config {
        bandwidth_size: 16,
        subgroup_width: 4,
    };

    let module = naga::front::wgsl::parse_str("
        @group(0) @binding(0)
        var<storage, read> input: array<f32>;

        @group(0) @binding(1)
        var<storage, read_write> output: array<f32>;

        fn main() {
            // let a = 2;
            output[0] = input[2] * 3.0;
        }
    ").unwrap();

    eprintln!("{:#?}", module);

    let module_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(), naga::valid::Capabilities::default() | naga::valid::Capabilities::FLOAT64,
    ).validate(&module).unwrap();

    // eprintln!("{:#?}", module.functions.iter().next().unwrap().1);
    let target = module.functions.iter().next().unwrap();

    // eprintln!("{:#?}", module_info[target.0]);

    let (func, func_sig) = translate_func(&module, &module_info, target.1, &module_info[target.0], &config);

    // let entry_point = &module.entry_points[0];


    let mut flag_builder = settings::builder();

    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let isa_builder = cranelift::native::builder().unwrap();

    let flags = settings::Flags::new(flag_builder);
    let result = codegen::verify_function(&func, &flags);

    eprintln!("{:?}", result);

    // let isa = isa_builder.finish(flags).unwrap();
    // eprintln!("{:?}", isa.pointer_type());


    // let mut jit_module = JITModule::new(JITBuilder::with_isa(isa.clone(), cranelift::module::default_libcall_names()));

    // let mut ctx = jit_module.make_context();
    // let mut func_ctx = FunctionBuilderContext::new();

    // let mut func_ctx = codegen::Context::for_function(func.clone());

    // let func_id = jit_module
    //     .declare_function("a", cranelift::module::Linkage::Local, &func_sig)
    //     .unwrap();

    // jit_module.define_function(func_id, &mut func_ctx).unwrap();
    // jit_module.finalize_definitions().unwrap();

    // let code = jit_module.get_finalized_function(func_id);
    // let func = unsafe { std::mem::transmute::<_, fn(f32, f32, f32, f32) -> (f32, f32, f32, f32)>(code) };

    // // eprintln!("{:?}", code);
    // eprintln!("{:?}", func(2.0, 3.0, 4.0, 5.0));


    // let mut module2 = cranelift_object::ObjectModule::new(cranelift_object::ObjectBuilder::new(
    //     isa,
    //     "foo",
    //     cranelift::module::default_libcall_names(),
    //     // cranelift_object::ObjectBuilder::default_libcall_names(),
    // ).unwrap());

    // let mut func_ctx = codegen::Context::for_function(func.clone());

    // let func_id = module2
    //     .declare_function("a", cranelift::module::Linkage::Local, &func_sig)
    //     .unwrap();

    // let res = module2.define_function(func_id, &mut func_ctx);
    // res.unwrap();

    // let product = module2.finish();
    // let output = product.emit().unwrap();

    // let mut file = std::fs::File::create("output.o").unwrap();
    // file.write_all(&output).unwrap();


    let input_data = (0..4).map(|x| x as f32).collect::<Vec<_>>();
    let mut output_data = vec![0.0; 4];

    let input_buffer: &[u8] = bytemuck::cast_slice(&input_data);
    let output_buffer: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    let buffers = vec![input_buffer, output_buffer];
}
