#include <torch/extension.h>
#include "includes/utils.h"
#include <iostream>
#include <cstdio>
#include <cmath>

// Helper function to compute the min and max values of a voxel's 8 vertices
template <typename scalar_t>
__device__ __forceinline__ bool compute_min_max(
    const scalar_t* voxel_vertices, scalar_t& min_val, scalar_t& max_val) 
{
    min_val = voxel_vertices[0];
    max_val = voxel_vertices[0];

    for (int i = 1; i < 8; ++i) {
        scalar_t val = voxel_vertices[i];

        // If a value is NaN, return early as invalid
        if (isnan(val)) {
            return false;
        }

        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    return true;  // Valid min and max computed
}



template <typename scalar_t>
__global__ void get_intersections_inter_slice_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> slice_fields,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> lattice1_fields,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> lattice2_fields,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> slice_iso_values,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> lattice1_iso_values,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> lattice2_iso_values,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> voxel_centers,
    torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits> inter_mask,
    torch::PackedTensorAccessor<int16_t, 4, torch::RestrictPtrTraits> isovalue_indices,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> intersections,
    const int dim_x, const int dim_y, const int dim_z) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) {
        return;
    }

    // Define corner offsets relative to the voxel center
    const scalar_t voxel_size = 2.0 / static_cast<scalar_t>(dim_x);
    const scalar_t offsets[8][3] = {
        {-0.5, -0.5, -0.5},  // Corner 0
        { 0.5, -0.5, -0.5},  // Corner 1
        {-0.5,  0.5, -0.5},  // Corner 2
        { 0.5,  0.5, -0.5},  // Corner 3
        {-0.5, -0.5,  0.5},  // Corner 4
        { 0.5, -0.5,  0.5},  // Corner 5
        {-0.5,  0.5,  0.5},  // Corner 6
        { 0.5,  0.5,  0.5}   // Corner 7
    };

    // Define the 12 primary edges of the voxel as pairs of corner indices
    const int edges[12][2] = {
        {0, 1}, {0, 2}, {1, 3}, {2, 3},  // Bottom face edges
        {4, 5}, {4, 6}, {5, 7}, {6, 7},  // Top face edges
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges connecting top and bottom
    };

    // Initialize variables to track min and max values for each field
    scalar_t lattice1_min, lattice1_max;
    scalar_t lattice2_min, lattice2_max;
    scalar_t slice_min, slice_max;

    // Compute min and max for the lattice1 field
    if (!compute_min_max<scalar_t>(&lattice1_fields[x][y][z][0], lattice1_min, lattice1_max)) {
        return;
    }

    // Compute min and max for the lattice2 field
    if (!compute_min_max<scalar_t>(&lattice2_fields[x][y][z][0], lattice2_min, lattice2_max)) {
        return;
    }

    // Compute min and max for the slice field
    if (!compute_min_max<scalar_t>(&slice_fields[x][y][z][0], slice_min, slice_max)) {
        return;
    }

    // Iterate over all combinations of iso-values from the three fields
    // slice inter is after lattice1 and lattice2
    for (int lattice1_idx = 0; lattice1_idx < lattice1_iso_values.size(0); ++lattice1_idx) {
        scalar_t lattice1_iso = lattice1_iso_values[lattice1_idx];
        // Check if the lattice1 iso-value intersects the lattice1 field
        if (lattice1_iso < lattice1_min || lattice1_iso > lattice1_max) {
            continue;  // Skip if the iso-value does not intersect
        }

        for (int lattice2_idx = 0; lattice2_idx < lattice2_iso_values.size(0); ++lattice2_idx) {
            scalar_t lattice2_iso = lattice2_iso_values[lattice2_idx];

            // Check if the lattice2 iso-value intersects the lattice2 field
            if (lattice2_iso < lattice2_min || lattice2_iso > lattice2_max) {
                continue;  // Skip if the iso-value does not intersect
            }

            for (int slice_idx = 0; slice_idx < slice_iso_values.size(0); ++slice_idx) {
                scalar_t slice_iso = slice_iso_values[slice_idx];

                // Check if the slice iso-value intersects the slice field
                if (slice_iso < slice_min || slice_iso > slice_max) {
                    continue;  // Skip if the iso-value does not intersect
                }
                
                // do have intersection, record the indices
                isovalue_indices[slice_idx][lattice1_idx][lattice2_idx][0] = slice_idx;
                isovalue_indices[slice_idx][lattice1_idx][lattice2_idx][1] = lattice1_idx;
                isovalue_indices[slice_idx][lattice1_idx][lattice2_idx][2] = lattice2_idx;

                // mark the intersection 
                inter_mask[slice_idx][lattice1_idx][lattice2_idx] = true;

                // Compute intersection points
                scalar_t slice_intersection[3] = {0, 0, 0};
                scalar_t lattice1_intersection[3] = {0, 0, 0};
                scalar_t lattice2_intersection[3] = {0, 0, 0};
                int slice_count = 0;
                int lattice1_count = 0;
                int lattice2_count = 0;

                // Iterate over all 12 primary edges of the voxel
                for (int edge = 0; edge < 12; ++edge) {
                    int corner1 = edges[edge][0];
                    int corner2 = edges[edge][1];

                    scalar_t value1_slice = slice_fields[x][y][z][corner1];
                    scalar_t value2_slice = slice_fields[x][y][z][corner2];

                    // Check if the slice iso-value intersects this edge
                    if (slice_iso >= min(value1_slice, value2_slice) &&
                        slice_iso <= max(value1_slice, value2_slice)) {
                        
                        // Linear interpolation along the edge for the slice field
                        scalar_t t = (slice_iso - value1_slice) / (value2_slice - value1_slice);

                        // Compute the interpolated intersection point
                        for (int coord = 0; coord < 3; ++coord) {
                            scalar_t coord1 = voxel_centers[x][y][z][coord] + voxel_size * offsets[corner1][coord];
                            scalar_t coord2 = voxel_centers[x][y][z][coord] + voxel_size * offsets[corner2][coord];
                            slice_intersection[coord] += coord1 + t * (coord2 - coord1);
                        }

                        ++slice_count;
                    }
                }
                
                // Process edges for lattice1 field
                for (int edge = 0; edge < 12; ++edge) {
                    int corner1 = edges[edge][0];
                    int corner2 = edges[edge][1];

                    scalar_t value1_lattice1 = lattice1_fields[x][y][z][corner1];
                    scalar_t value2_lattice1 = lattice1_fields[x][y][z][corner2];

                    // Check if the lattice1 iso-value intersects this edge
                    if (lattice1_iso >= min(value1_lattice1, value2_lattice1) &&
                        lattice1_iso <= max(value1_lattice1, value2_lattice1)) {
                        scalar_t t = (lattice1_iso - value1_lattice1) / (value2_lattice1 - value1_lattice1);
                        for (int coord = 0; coord < 3; ++coord) {
                            scalar_t coord1 = voxel_centers[x][y][z][coord] + voxel_size * offsets[corner1][coord];
                            scalar_t coord2 = voxel_centers[x][y][z][coord] + voxel_size * offsets[corner2][coord];
                            lattice1_intersection[coord] += coord1 + t * (coord2 - coord1);
                        }
                        ++lattice1_count;
                    }
                }

                // Process edges for lattice2 field
                for (int edge = 0; edge < 12; ++edge) {
                    int corner1 = edges[edge][0];
                    int corner2 = edges[edge][1];

                    scalar_t value1_lattice2 = lattice2_fields[x][y][z][corner1];
                    scalar_t value2_lattice2 = lattice2_fields[x][y][z][corner2];

                    // Check if the lattice2 iso-value intersects this edge
                    if (lattice2_iso >= min(value1_lattice2, value2_lattice2) &&
                        lattice2_iso <= max(value1_lattice2, value2_lattice2)) {
                        scalar_t t = (lattice2_iso - value1_lattice2) / (value2_lattice2 - value1_lattice2);
                        for (int coord = 0; coord < 3; ++coord) {
                            scalar_t coord1 = voxel_centers[x][y][z][coord] + voxel_size * offsets[corner1][coord];
                            scalar_t coord2 = voxel_centers[x][y][z][coord] + voxel_size * offsets[corner2][coord];
                            lattice2_intersection[coord] += coord1 + t * (coord2 - coord1);
                        }
                        ++lattice2_count;
                    }
                }

                // Average the intersections for slice, lattice1, and lattice2 fields
                scalar_t final_intersection[3] = {0, 0, 0};
                for (int coord = 0; coord < 3; ++coord) {
                    if (slice_count > 0) slice_intersection[coord] /= slice_count;
                    if (lattice1_count > 0) lattice1_intersection[coord] /= lattice1_count;
                    if (lattice2_count > 0) lattice2_intersection[coord] /= lattice2_count;

                    // Final average of the three fields
                    final_intersection[coord] = (slice_intersection[coord] +
                                                lattice1_intersection[coord] +
                                                lattice2_intersection[coord]) / 3.0;
                }

                // Store the final intersection
                for (int coord = 0; coord < 3; ++coord) {
                    intersections[slice_idx][lattice1_idx][lattice2_idx][coord] = final_intersection[coord];
                }

            }

        }
    }
}


std::vector<torch::Tensor> get_fields_intersection_inter_slice_cu(
    const torch::Tensor& slice_fields,
    const torch::Tensor& lattice1_fields,
    const torch::Tensor& lattice2_fields,
    const torch::Tensor& slice_iso_values,
    const torch::Tensor& lattice1_iso_values,
    const torch::Tensor& lattice2_iso_values,
    const torch::Tensor& voxel_centers)
    {   
    // intersections are computed directly
    const int dim_x = slice_fields.size(0);
    const int dim_y = slice_fields.size(1);
    const int dim_z = slice_fields.size(2);

    const int N_slice = slice_iso_values.size(0);
    const int N_lattice1 = lattice1_iso_values.size(0);
    const int N_lattice2 = lattice2_iso_values.size(0);

    //  N x M1 x M2 x 3
    torch::Tensor inter_mask = torch::zeros({N_slice, N_lattice1, N_lattice2}, torch::CUDA(torch::kBool));
    torch::Tensor isovalue_indices = torch::full({N_slice, N_lattice1, N_lattice2, 3}, -1, torch::CUDA(torch::kInt16));
    torch::Tensor intersections = torch::zeros({N_slice, N_lattice1, N_lattice2, 3}, slice_fields.options());

    const dim3 threads_per_block(8, 8, 8);
    const dim3 bloacks_per_grid(
        (dim_x + threads_per_block.x - 1) / threads_per_block.x,
        (dim_y + threads_per_block.y - 1) / threads_per_block.y,
        (dim_z + threads_per_block.z - 1) / threads_per_block.z
        );

    // launch kernel
    AT_DISPATCH_FLOATING_TYPES(slice_fields.type(), "get_intersections_inter_slice_parallel", 
    ([&] {
        get_intersections_inter_slice_kernel<scalar_t><<<bloacks_per_grid, threads_per_block>>>(
            // packed_accessor only requires when Tensor
            slice_fields.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            lattice1_fields.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            lattice2_fields.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            slice_iso_values.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits>(),
            lattice1_iso_values.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits>(),
            lattice2_iso_values.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits>(),
            voxel_centers.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            inter_mask.packed_accessor<bool, 3, torch::RestrictPtrTraits>(),
            isovalue_indices.packed_accessor<int16_t, 4, torch::RestrictPtrTraits>(),
            intersections.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            dim_x, dim_y, dim_z
        );

        // check if kernel launch is successful
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        cudaDeviceSynchronize(); // ensure finish first kernel before next kernel

    }));

    return {inter_mask, isovalue_indices, intersections};

}