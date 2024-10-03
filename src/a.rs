#[derive(Clone, Debug)]
pub struct Api;


struct Adapter {

}

impl wgpu_hal::Adapter for Adapter {
    type A = Api;

    unsafe fn open(
        &self,
        features: wgpu::Features,
        limits: &wgpu::Limits,
        memory_hints: &wgpu::MemoryHints,
    ) -> Result<wgpu_hal::OpenDevice<Self::A>, wgpu_hal::DeviceError> {
        unimplemented!()
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgpu::TextureFormat,
    ) -> wgpu_hal::TextureFormatCapabilities {
        unimplemented!()
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &<Self::A as wgpu_hal::Api>::Surface,
    ) -> Option<wgpu::SurfaceCapabilities> {
        None
    }

    unsafe fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        unimplemented!()
    }
}


#[derive(Debug)]
struct Buffer {
    pointer: *const u8,
    size: usize,
}


#[derive(Debug)]
struct Device {

}

impl wgpu_hal::Device for Device {
    type A = Api;

    unsafe fn create_buffer(
        &self,
        desc: &wgpu::BufferDescriptor,
    ) -> Result<<Self::A as wgpu_hal::Api>::Buffer, wgpu_hal::DeviceError> {
        let layout = std::alloc::Layout::from_size_align(desc.size as usize, 1).unwrap();
        let pointer = std::alloc::alloc(layout);

        Ok(Buffer {
            pointer,
            size: desc.size as usize,
        })
    }
}


#[derive(Debug)]
struct Instance {

}

impl wgpu_hal::Instance for Instance {
    type A = Api;

    unsafe fn init(desc: &wgpu_hal::InstanceDescriptor) -> Result<Self, wgpu_hal::InstanceError> {
        Ok(Self { })
    }

    unsafe fn create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<<Self::A as wgpu_hal::Api>::Surface, wgpu_hal::InstanceError> {
        unimplemented!()
    }

    unsafe fn destroy_surface(&self, surface: <Self::A as wgpu_hal::Api>::Surface) {
        unimplemented!()
    }

    unsafe fn enumerate_adapters(
        &self,
        surface_hint: Option<&<Self::A as wgpu_hal::Api>::Surface>,
    ) -> Vec<wgpu_hal::ExposedAdapter<Self::A>> {
        unimplemented!()
    }
}

impl wgpu_hal::Api for Api {
    type Adapter = Adapter;
    type Instance = Instance;
}

fn main() {
    let instance = Instance { };
    let adapter = unsafe {
        wgpu::Instance::from_hal::<Api>(instance)
    };
}
