use std::collections::HashMap;

use cranelift::codegen::ir;
use cranelift::jit;
use cranelift::module::{DataDescription, Module};
use cranelift::prelude::Configurable;
use crate::config::Config;
use naga::proc::TypeResolution;


#[derive(Debug)]
pub struct BindGroup<'a> {
    entries: &'a [BindEntry<'a>],
}

#[derive(Debug)]
pub enum BindEntry<'a> {
    ReadableBuffer(&'a [u8]),
    WritableBuffer(&'a mut [u8]),
}


pub struct CompiledPipeline {

}

impl CompiledPipeline {
    fn run(thread_count: usize, bind_groups: &[BindGroup<'_>]) {

    }
}

pub fn jit_compile(source_code: &str, config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    // Parse and validate module

    let module = naga::front::wgsl::parse_str(source_code)?;

    let module_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(), naga::valid::Capabilities::default() | naga::valid::Capabilities::FLOAT64,
    ).validate(&module)?;

    let target = module.functions.iter().next().unwrap();


    // Build ISA

    let mut flag_builder = cranelift::codegen::settings::builder();

    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let isa_builder = cranelift::native::builder().unwrap();
    let flags = cranelift::codegen::settings::Flags::new(flag_builder);
    let isa = isa_builder.finish(flags.clone()).unwrap();

    // eprintln!("{:?}", isa.pointer_type());


    // Build context

    let mut sorted_bounded_global_variables = module.global_variables
        .iter()
        .filter_map(|(handle, global_var)| {
            global_var.binding
                .as_ref()
                .and_then(|binding| {
                    Some((handle, (binding.group, binding.binding)))
                })
        })
        .collect::<Vec<_>>();

    sorted_bounded_global_variables.sort_by_key(|(_, key)| *key);
    let sorted_bounded_global_variables = sorted_bounded_global_variables
        .into_iter()
        .map(|(global_var, _)| global_var)
        .collect::<Vec<_>>();

    let global_var_map = sorted_bounded_global_variables
        .iter()
        .enumerate()
        .map(|(index, handle)| (*handle, index))
        .collect::<HashMap<_, _>>();

    // eprintln!("{:?}", sorted_bounded_global_variables);
    // eprintln!("{:?}", module.global_variables);
    // eprintln!("{:#?}", module.constants);

    // let mut layouter = naga::proc::Layouter::default();
    // layouter.update(module.to_ctx())?;

    // for (handle, constant) in module.constants.iter() {
    //     let ty = constant.ty;
    //     let x = layouter[ty];
    //     // eprintln!("{:?}", x);
    //     // eprintln!("{:#?}", module.global_expressions[constant.init]);
    // }

    let constant_map = module.constants
        .iter()
        .enumerate()
        .map(|(index, (handle, _))| (handle, index))
        .collect::<HashMap<_, _>>();

    // let const_layouts = module.constants.iter().map(|(handle, constant)| {
    //     layouter[constant.ty]
    // }).collect::<Vec<_>>();

    // eprintln!("{:#?}", const_layouts);
    // eprintln!("{:?}", assemble(&const_layouts));

    let mut jit_module = jit::JITModule::new(
        jit::JITBuilder::with_isa(isa.clone(), cranelift::module::default_libcall_names())
    );

    let constants_data_id = jit_module.declare_data("constants", cranelift::module::Linkage::Hidden, false, false)?;
    let mut constants_data_description = DataDescription::new();

    constants_data_description.define_zeroinit(15);
    constants_data_description.set_align(16);

    jit_module.define_data(constants_data_id, &constants_data_description)?;


    let module_context = crate::context::ModuleContext {
        config,
        constant_map: &constant_map,
        constants_data_id,
        cl_module: &mut jit_module,
        global_var_map: &global_var_map,
        module_info: &module_info,
        module: &module,
        pointer_type: isa.pointer_type(),
    };


    let (func, func_sig) = crate::translate::translate_func(&module_context, target.1, &module_info[target.0]);
    cranelift::codegen::verify_function(&func, &flags)?;


    jit_module.finalize_definitions()?;

    let x = jit_module.get_finalized_data(constants_data_id);
    // eprintln!("{:?}", assemble(&[]));

    // eprintln!("{:?}", result);

    Ok(())
}


fn assemble(layouts: &[naga::proc::TypeLayout]) -> (naga::proc::TypeLayout, Vec<usize>) {
    let mut offsets = Vec::with_capacity(layouts.len());
    let mut offset = 0;

    for layout in layouts {
        let item_alignment = (layout.alignment * 1u32) as usize;
        let padding = (item_alignment - (offset % item_alignment)) % item_alignment;

        offset += padding;
        offsets.push(offset);
        offset += layout.size as usize;
    }

    let alignment = layouts
        .iter()
        .map(|layout| layout.alignment)
        .max()
        .unwrap();

    (naga::proc::TypeLayout { size: offset as u32, alignment }, offsets)
}
