#[derive(Clone, Debug)]
pub struct Config {
    pub bandwidth_size: usize, // Usually 16
    pub simul_thread_count: usize,
}

impl Config {
    pub fn compute_sizes(&self, item_size: usize) -> (usize, usize) {
        let lane_count = self.bandwidth_size / item_size;
        let vector_count = item_size * self.simul_thread_count / self.bandwidth_size;

        (lane_count, vector_count)
    }
}
