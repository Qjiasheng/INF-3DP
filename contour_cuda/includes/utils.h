#include <torch/extension.h>
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> extract_isocontours_cu(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& fields,
    const torch::Tensor& iso_values
    );

std::vector<torch::Tensor> get_fields_intersection_inter_slice_cu(
    const torch::Tensor& slice_fields,
    const torch::Tensor& lattice1_fields,
    const torch::Tensor& lattice2_fields,
    const torch::Tensor& slice_iso_values,
    const torch::Tensor& lattice1_iso_values,
    const torch::Tensor& lattice2_iso_values,
    const torch::Tensor& voxel_centers
    );
