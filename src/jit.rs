use std::collections::HashMap;

use cranelift::jit;
use cranelift::module::{DataDescription, Module};
use cranelift::prelude::Configurable;
use crate::config::Config;
use crate::constants::build_constant;


pub type BindGroups<'a> = [BindGroup<'a>];

#[derive(Debug)]
pub struct BindGroup<'a> {
    pub entries: &'a [BindEntry<'a>],
}

#[derive(Debug)]
pub enum BindEntry<'a> {
    ReadableBuffer(&'a [u8]),
    WritableBuffer(&'a mut [u8]),
}

impl<'a, T: bytemuck::NoUninit + bytemuck::AnyBitPattern> From<&'a [T]> for BindEntry<'a> {
    fn from(slice: &'a [T]) -> Self {
        Self::ReadableBuffer(bytemuck::cast_slice(slice))
    }
}

impl<'a, T: bytemuck::NoUninit + bytemuck::AnyBitPattern> From<&'a mut [T]> for BindEntry<'a> {
    fn from(slice: &'a mut [T]) -> Self {
        Self::WritableBuffer(bytemuck::cast_slice_mut(slice))
    }
}


#[derive(Debug)]
#[repr(C, packed)]
pub(crate) struct RuntimeBuiltins {
    subgroup_id: u32,
}


#[derive(Debug)]
pub struct CompiledPipeline {
    config: Config,
    func_pointer: fn(*const u8, *const u8) -> (),
}

impl CompiledPipeline {
    pub fn run(&self, thread_count: usize, bind_groups: &[BindGroup<'_>]) {
        assert_eq!(thread_count % self.config.subgroup_width, 0);
        let subgroup_count = thread_count / self.config.subgroup_width;

        // eprintln!("subgroup_count: {}", subgroup_count);

        let mut builtins = RuntimeBuiltins {
            subgroup_id: 0,
        };

        let builtins_pointer = &builtins as *const _ as *const u8;

        fn get_first_item_pointer<T>(slice: &[T]) -> *const u8 {
            if !slice.is_empty() {
                &slice[0] as *const _ as *const u8
            } else {
                std::ptr::null()
            }
        }

        let global_vars = bind_groups
            .iter()
            .flat_map(|bind_group| {
                bind_group.entries
                    .iter()
                    .map(|entry| {
                        match entry {
                            BindEntry::ReadableBuffer(slice)
                                => get_first_item_pointer(slice),
                            BindEntry::WritableBuffer(slice)
                                => get_first_item_pointer(slice),
                        }
                    })
            })
            .collect::<Vec<_>>();

        let global_vars_pointer = get_first_item_pointer(&global_vars);

        for subgroup_index in 0..subgroup_count {
            builtins.subgroup_id = subgroup_index as u32;
            (self.func_pointer)(builtins_pointer, global_vars_pointer);
        }
    }
}


pub fn jit_compile(source_code: &str, config: &Config) -> Result<CompiledPipeline, Box<dyn std::error::Error>> {
    // Parse and validate module

    let module = naga::front::wgsl::parse_str(source_code)?;

    let module_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(), naga::valid::Capabilities::default() | naga::valid::Capabilities::FLOAT64,
    ).validate(&module)?;

    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx())?;

    let target = &module.entry_points[0];
    let target_info = module_info.get_entry_point(0);

    // eprintln!("{:#?}", target);


    // Build ISA

    let mut flag_builder = cranelift::codegen::settings::builder();

    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let isa_builder = cranelift::native::builder().unwrap();
    let flags = cranelift::codegen::settings::Flags::new(flag_builder);
    let isa = isa_builder.finish(flags.clone()).unwrap();


    // Build JIT module

    let mut jit_module = jit::JITModule::new(
        jit::JITBuilder::with_isa(isa.clone(), cranelift::module::default_libcall_names())
    );


    // Build global variables

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

    // for (handle, constant) in module.constants.iter() {
    //     let ty = constant.ty;
    //     let x = layouter[ty];
    //     // eprintln!("{:?}", x);
    //     // eprintln!("{:#?}", module.global_expressions[constant.init]);
    // }


    // Build constants

    let constant_map = module.constants
        .iter()
        .enumerate()
        .map(|(index, (handle, _))| (handle, index))
        .collect::<HashMap<_, _>>();

    let constant_layouts = module.constants
        .iter()
        .map(|(_, constant)| layouter[constant.ty])
        .collect::<Vec<_>>();

    let (constants_buffer_layout, constants_buffer_offsets) = pack(&constant_layouts);
    let mut constants_buffer = vec![0u8; constants_buffer_layout.size as usize];

    for (((_, constant), offset), layout) in module.constants.iter().zip(constants_buffer_offsets).zip(constant_layouts.iter()) {
        build_constant(&mut constants_buffer[offset..(offset + layout.size as usize)], &module, &layouter, &module.global_expressions[constant.init], &module.types[constant.ty].inner);
    }

    // eprintln!("{:?}", constants_buffer);

    let constants_data_id = jit_module.declare_data("constants", cranelift::module::Linkage::Hidden, false, false)?;
    let mut constants_data_description = DataDescription::new();

    // constants_data_description.define_zeroinit(15);
    constants_data_description.define(constants_buffer.into_boxed_slice());
    constants_data_description.set_align(constants_buffer_layout.alignment.round_up(1) as u64);

    jit_module.define_data(constants_data_id, &constants_data_description)?;


    // Build context

    let module_context = crate::context::ModuleContext {
        config,
        constant_map: &constant_map,
        constants_data_id,
        cl_module: &mut jit_module,
        global_var_map: &global_var_map,
        layouter: &layouter,
        module_info: &module_info,
        module: &module,
        pointer_type: isa.pointer_type(),
    };


    // Translate function

    let (func, func_sig) = crate::translate::translate_func(&module_context, &target.function, &target_info);
    cranelift::codegen::verify_function(&func, &flags)?;

    let mut func_ctx = cranelift::codegen::Context::for_function(func);

    let func_id = jit_module
        .declare_function("a", cranelift::module::Linkage::Local, &func_sig)
        .unwrap();

    jit_module.define_function(func_id, &mut func_ctx)?;
    jit_module.finalize_definitions()?;

    // jit_module.define_function(func_id, &mut func_ctx).unwrap();
    // let x = jit_module.get_finalized_data(constants_data_id);
    // eprintln!("{:?}", assemble(&[]));

    let code = jit_module.get_finalized_function(func_id);
    let real_func = unsafe { std::mem::transmute::<_, fn(*const u8, *const u8) -> ()>(code) };

    // eprintln!("{:?}", result);

    Ok(CompiledPipeline {
        config: config.clone(),
        func_pointer: real_func,
    })
}


fn pack(layouts: &[naga::proc::TypeLayout]) -> (naga::proc::TypeLayout, Vec<usize>) {
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
