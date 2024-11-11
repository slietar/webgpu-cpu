use cranelift::{codegen::ir, prelude::{InstBuilder as _, Value}};

use crate::{context::FunctionContext, expr::translate_expr, repr::ExprRepr};


pub fn translate_block(
    naga_block: &naga::Block,
    cl_entry_block: ir::Block,
    mask: &ExprRepr,
    func_context: &mut FunctionContext,
    builder: &mut cranelift::frontend::FunctionBuilder,
) {
    let cl_block = cl_entry_block;

    // builder.switch_to_block(cl_block);
    // builder.seal_block(cl_block);

    for stat in naga_block.iter() {
        match stat {
            naga::Statement::Store { pointer: pointer_handle, value: value_handle } => {
                let pointer = func_context.get_expr(*pointer_handle, builder, cl_block);
                let value = func_context.get_expr(*value_handle, builder, cl_block);

                let pointer_scalars = pointer.into_scalars(&func_context, builder);
                let value_scalars = value.into_scalars(&func_context, builder);

                for (pointer_scalar, value_scalar) in pointer_scalars.iter().zip(value_scalars.iter()) {
                    builder.ins().store(ir::MemFlags::new(), *value_scalar, *pointer_scalar, 0);
                }
            },
            naga::Statement::Emit(expr_range) => {
                for expr_handle in expr_range.clone().into_iter() {
                    if !func_context.emitted_exprs.contains_key(&expr_handle) {
                        let expr = &func_context.function.expressions[expr_handle];
                        let expr_info = &func_context.function_info[expr_handle];

                        func_context.emitted_exprs.insert(expr_handle, translate_expr(&func_context, builder, cl_block, expr, expr_info));
                    }
                }
            },
            naga::Statement::If { accept, condition, reject } => {
                let condition_repr = func_context.get_expr(*condition, builder, cl_block);

                // Multiply condition_repr by mask

                let condition_mask_accept = mask.and(&condition_repr, func_context, builder);
                let condition_mask_reject = condition_mask_accept.not(func_context, builder);

                translate_block(accept, cl_block, &condition_mask_accept, func_context, builder);
                translate_block(reject, cl_block, &condition_mask_reject, func_context, builder);


                // More sophisticated version:

                // let accept_entry_block = builder.create_block();
                // let reject_entry_block = builder.create_block();

                // let condition_all = builder.ins().icmp_imm(ir::condcodes::IntCC::Equal, condition_repr, 0);
                // let condition_none = builder.ins().icmp_imm(ir::condcodes::IntCC::Equal, condition_repr, 1);

                // builder.ins().brnz(condition_all, accept_entry_block, &[]);
                // builder.ins().brnz(condition_none, reject_entry_block, &[]);

                // translate_block(accept, cl_block, mask, func_context, builder);
                // translate_block(reject, cl_block, /* opposite of */ mask, func_context, builder);


                // cl_block = builder.create_block();
                // builder.switch_to_block(cl_block);

                // let accept_entry_block = translate_block(accept, mask, func_context, builder);
                // let reject_entry_block = translate_block(reject, mask, func_context, builder);

                // let return_block = builder.create_block();
                // builder.ins().jump(return_block, &[]);
                // eprintln!("{:?}", condition_repr);

                // builder.switch_to_block(return_block);
            },
            naga::Statement::Return { value: None } => {},
            _ => panic!("unimplemented: {:?}", stat),
        }
    }
}
