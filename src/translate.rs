use std::collections::HashMap;

use cranelift::codegen::ir;
use cranelift::codegen::ir::{AbiParam, UserFuncName, Function, InstBuilder, Signature};
use cranelift::codegen::isa::CallConv;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::prelude::StackSlotData;
use crate::context::{FunctionContext, ModuleContext};
use crate::expr::translate_expr;
use crate::repr::ExprRepr;
use crate::types::translate_alignment;


pub(crate) enum Argument {
    BuiltIn(naga::BuiltIn),
    Regular(usize),
}

pub fn translate_func(
    module_context: &ModuleContext<'_, '_>,
    function: &naga::Function,
    function_info: &naga::valid::FunctionInfo,
) -> Result<(cranelift::codegen::ir::Function, Signature), Box<dyn std::error::Error>> {
    // General

    let mut cl_signature = Signature::new(CallConv::Fast);


    // Parameters

    // Pointer to builtins
    cl_signature.params.push(AbiParam::new(module_context.pointer_type));

    // Pointer to global variables
    cl_signature.params.push(AbiParam::new(module_context.pointer_type));


    let mut arguments = Vec::with_capacity(function.arguments.len());

    for arg in function.arguments.iter() {
        let shader_type = &module_context.module.types[arg.ty].inner;
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
        let shader_type = &module_context.module.types[result.ty].inner;
        let (cl_type, cl_type_count) = module_context.translate_type(shader_type);

        for _ in 0..cl_type_count {
            cl_signature.returns.push(AbiParam::new(cl_type));
        }
    }


    // Function

    let mut cl_func = Function::with_name_signature(UserFuncName::user(0, 0), cl_signature.clone());

    let constants_global_value = module_context.cl_module.declare_data_in_func(module_context.constants_data_id, &mut cl_func);


    // Builder

    let mut cl_fn_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut cl_func, &mut cl_fn_builder_ctx);
    let root_block = builder.create_block();

    builder.append_block_params_for_function_params(root_block);
    builder.switch_to_block(root_block);


    // Local variables

    let (local_var_slot, local_var_offsets) = if !function.local_variables.is_empty() {
        let (handles, layouts) = function.local_variables
            .iter()
            .map(|(handle, local_var)| (handle, module_context.layouter[local_var.ty]))
            .unzip::<_, _, Vec<_>, Vec<_>>();
            // .collect::<Vec<_>>();

        let (layout, offsets) = crate::jit::pack(&layouts);
        let translated_alignment = translate_alignment(layout.alignment);

        // TODO: Use builder.declare_var() instead for variables of small size
        let slot = builder.create_sized_stack_slot(StackSlotData {
            align_shift: translated_alignment,
            kind: ir::StackSlotKind::ExplicitSlot,
            size: layout.size as u32,
        });

        let offset_map = handles.into_iter().zip(offsets).collect::<HashMap<_, _>>();

        (Some(slot), offset_map)
    } else {
        (None, HashMap::new())
    };


    // Context

    let mut func_context = FunctionContext {
        arguments: &arguments,
        constants_global_value,
        emitted_exprs: HashMap::new(),
        function_info,
        function,
        // local_variable_slots: HashMap::new(),
        local_var_offsets,
        local_var_slot,
        module: module_context,
    };

    // HOME
    eprintln!("{:#?}", function);
    // eprintln!("{:#?}", function_info);


    // Initialize local variables

    if function.local_variables.iter().any(|(_, local_var)| local_var.init.is_some()) {
        let slot = func_context.local_var_slot.unwrap();
        let pointer = builder.ins().stack_addr(module_context.pointer_type, slot, 0 as i32);

        for (handle, local_var) in function.local_variables.iter() {
            if let Some(init_handle) = local_var.init {
                let offset = func_context.local_var_offsets[&handle];
                let value_repr = func_context.get_expr(init_handle, &mut builder, root_block);

                if let ExprRepr::Constant(value, _) = value_repr {
                    builder.ins().store(ir::MemFlags::new(), value, pointer, offset as i32);
                } else {
                    panic!("unimplemented: {:?}", value_repr);
                }
            }
        }
    }


    // Statements

    let init_mask = ExprRepr::Constant(builder.ins().iconst(ir::types::I32, 1), Some(ir::types::I32X4));
    crate::block::translate_block(&function.body, root_block, &init_mask, &mut func_context, &mut builder);

    builder.ins().return_(&[]);

    // builder.seal_block(block);

    builder.seal_all_blocks();
    builder.finalize();

    eprintln!("-- OUTPUT --------------------------------------------------------\n{:#?}", cl_func);

    Ok((cl_func, cl_signature))
}
