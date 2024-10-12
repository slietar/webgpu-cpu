pub fn build_constant(buffer: &mut [u8], module: &naga::Module, layouter: &naga::proc::Layouter, expr: &naga::Expression, ty: &naga::TypeInner) {
    match expr {
        naga::Expression::Literal(naga::Literal::F32(value))
            => buffer.copy_from_slice(&value.to_le_bytes()),
        naga::Expression::Literal(naga::Literal::I32(value))
            => buffer.copy_from_slice(&value.to_le_bytes()),
        naga::Expression::Literal(naga::Literal::U32(value))
            => buffer.copy_from_slice(&value.to_le_bytes()),

        naga::Expression::Compose { ty, components } => {
            // eprintln!("{:?}", module.types[*ty].inner);
            let target = &module.types[*ty].inner;

            match target {
                naga::TypeInner::Struct { ref members, span } => {
                    for (member_index, (member, component)) in members.iter().zip(components.iter()).enumerate() {
                        let start_offset = member.offset as usize;
                        let end_offset = members.get(member_index + 1).map(|member| member.offset).unwrap_or(*span) as usize;

                        build_constant(&mut buffer[start_offset..end_offset], module, layouter, &module.global_expressions[*component], &module.types[member.ty].inner);
                    }
                },
                _ => panic!("Unsupported constant type {:?}", ty),
            }
        },
        _ => panic!("Unsupported constant type {:?}", expr),
    }
}
