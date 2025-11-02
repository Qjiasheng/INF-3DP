#include <torch/extension.h>
#include "includes/utils.h"  


std::vector<torch::Tensor> extract_isocontours(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& fields,
    const torch::Tensor& iso_values
){
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(fields);
    CHECK_INPUT(iso_values);

    return extract_isocontours_cu(vertices, faces, fields, iso_values);
}


std::vector<torch::Tensor> get_fields_intersection_inter_slice(
    const torch::Tensor& slice_fields,
    const torch::Tensor& lattice1_fields,
    const torch::Tensor& lattice2_fields,
    const torch::Tensor& slice_iso_values,
    const torch::Tensor& lattice1_iso_values,
    const torch::Tensor& lattice2_iso_values,
    const torch::Tensor& voxel_centers
){
    CHECK_INPUT(slice_fields);
    CHECK_INPUT(lattice1_fields);
    CHECK_INPUT(lattice2_fields);
    CHECK_INPUT(slice_iso_values);
    CHECK_INPUT(lattice1_iso_values);
    CHECK_INPUT(lattice2_iso_values);
    CHECK_INPUT(voxel_centers);

    return get_fields_intersection_inter_slice_cu(
        slice_fields,
        lattice1_fields,
        lattice2_fields,
        slice_iso_values,
        lattice1_iso_values,
        lattice2_iso_values,
        voxel_centers
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extract_isocontours", &extract_isocontours);  // iso contours extraction, marching triangles
  m.def("get_fields_intersection_inter_slice", &get_fields_intersection_inter_slice);  // 3 fields, interpolate along slice value
}