use std::io::Write as _;

use cranelift::codegen::ir::types as types;
use cranelift::prelude::*;
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::codegen::ir::{AbiParam, Block, UserFuncName, Function, InstBuilder, Signature, Value};
use cranelift::codegen::isa::CallConv;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::module::Module as _;


fn translate_type(ty: &naga::Type) -> types::Type {
    match ty.inner {
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }) => types::F32X4,
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 8 }) => types::F64X8,
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, width: 4 }) => types::I32X4,
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 }) => types::I32X4,
        _ => unimplemented!(),
    }
}

fn translate_expr(
    fn_builder: &mut FunctionBuilder<'_>,
    block: Block,
    function: &naga::Function,
    function_info: &naga::valid::FunctionInfo,
    expr: &naga::Expression,
    expr_info: &naga::valid::ExpressionInfo,
) -> Value {
    match expr {
        naga::Expression::Literal(literal) => {
            match literal {
                naga::Literal::F32(v) => {
                    let value = fn_builder.ins().f32const(*v);
                    fn_builder.ins().splat(types::F32X4, value)
                },
                _ => unimplemented!(),
            }
        },
        naga::Expression::FunctionArgument(arg) => {
            fn_builder.block_params(block)[*arg as usize]
            // let value = fn_builder.block_params(fn_builder.func.dfg.block_params(fn_builder.func.layout.entry_block));
            // value[*arg as usize]
        },
        naga::Expression::Binary { op, left, right } => {
            let left = translate_expr(fn_builder, block, function, function_info, &function.expressions[*left], &function_info[*left]);
            let right = translate_expr(fn_builder, block, function, function_info, &function.expressions[*right], &function_info[*right]);

            let ins = fn_builder.ins();

            match op {
                naga::BinaryOperator::Add => ins.fadd(left, right),
                naga::BinaryOperator::Subtract => ins.fsub(left, right),
                naga::BinaryOperator::Multiply => ins.fmul(left, right),
                naga::BinaryOperator::Divide => ins.fdiv(left, right),
                _ => unimplemented!(),
            }
        },
        naga::Expression::Math { fun, arg, arg1, arg2, arg3 } => {
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
        },
        _ => {
            panic!("unimplemented: {:?}", expr);
        },
    }
}

fn translate_func(module: &naga::Module, module_info: &naga::valid::ModuleInfo, function: &naga::Function, function_info: &naga::valid::FunctionInfo) -> (cranelift::codegen::ir::Function, Signature) {
    let mut cl_signature = Signature::new(CallConv::Fast);
    cl_signature.returns.push(AbiParam::new(types::F32X4));

    for arg in function.arguments.iter() {
        let t = &module.types[arg.ty];
        cl_signature.params.push(AbiParam::new(translate_type(t)));
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
            naga::Statement::Return { value } => {
                let expr = &function.expressions[value.unwrap()];
                let expr_info = &function_info[value.unwrap()];
                let value = translate_expr(&mut fn_builder, block, function, function_info, expr, expr_info);

                fn_builder.ins().return_(&[value]);
            },
            _ => {},
        }
    }

    // fn_builder.seal_block(block);

    fn_builder.seal_all_blocks();
    fn_builder.finalize();

    eprintln!("-- OUTPUT --------------------------------------------------------\n{:#?}", cl_func);

    (cl_func, cl_signature)
}


fn main() {
    let module = naga::front::wgsl::parse_str("
        fn main(x: f32, y: u32) -> f32 {
            return x * 2.0 + x;
            // return sqrt(x + 4.0) * 2.0;
        }
    ").unwrap();

    let module_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(), naga::valid::Capabilities::default()
    ).validate(&module).unwrap();

    // eprintln!("{:#?}", module.functions.iter().next().unwrap().1);
    let target = module.functions.iter().next().unwrap();

    // eprintln!("{:#?}", module_info[target.0]);

    let (func, func_sig) = translate_func(&module, &module_info, target.1, &module_info[target.0]);

    // let entry_point = &module.entry_points[0];


    let mut flag_builder = settings::builder();

    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let isa_builder = cranelift::native::builder().unwrap();

    let flags = settings::Flags::new(flag_builder);
    let result = codegen::verify_function(&func, &flags);

    eprintln!("{:?}", result);

    let isa = isa_builder.finish(flags).unwrap();

    let mut jit_module = JITModule::new(JITBuilder::with_isa(isa.clone(), cranelift::module::default_libcall_names()));

    // let mut ctx = jit_module.make_context();
    // let mut func_ctx = FunctionBuilderContext::new();

    // let mut func_ctx = codegen::Context::for_function(func.clone());

    // let func_id = jit_module
    //     .declare_function("a", cranelift::module::Linkage::Local, &func_sig)
    //     .unwrap();

    // let res = jit_module.define_function(func_id, &mut func_ctx);

    // jit_module.finalize_definitions().unwrap();

    // let code = jit_module.get_finalized_function(func_id);
    // eprintln!("{:?}", code);
    // eprintln!("{:?}", res);

    // let mut output = String::new();

    // eprintln!("{}", output);

    let mut module2 = cranelift_object::ObjectModule::new(cranelift_object::ObjectBuilder::new(
        isa,
        "foo",
        cranelift::module::default_libcall_names(),
        // cranelift_object::ObjectBuilder::default_libcall_names(),
    ).unwrap());

    let mut func_ctx = codegen::Context::for_function(func.clone());

    let func_id = module2
        .declare_function("a", cranelift::module::Linkage::Local, &func_sig)
        .unwrap();

    let res = module2.define_function(func_id, &mut func_ctx);
    res.unwrap();

    let product = module2.finish();
    let output = product.emit().unwrap();

    let mut file = std::fs::File::create("output.o").unwrap();
    file.write_all(&output).unwrap();
}
