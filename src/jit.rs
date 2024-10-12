use std::collections::HashMap;

use crate::config::Config;
use naga::proc::TypeResolution;
use cranelift::prelude::Configurable;
use cranelift::codegen::ir;


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
    let isa = isa_builder.finish(flags).unwrap();

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

    eprintln!("{:?}", sorted_bounded_global_variables);
    eprintln!("{:?}", module.global_variables);

    let context = crate::context::ModuleContext {
        config,
        global_var_map: &HashMap::new(),
        module_info: &module_info,
        module: &module,
        pointer_type: isa.pointer_type(),
    };


    // let (func, func_sig) = crate::translate::translate_func(&module, &module_info, target.1, &module_info[target.0], &config);
    // let result = codegen::verify_function(&func, &flags);

    // eprintln!("{:?}", result);

    Ok(())
}
