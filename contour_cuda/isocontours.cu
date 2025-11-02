#include <torch/extension.h>
#include "includes/utils.h"
#include <iostream>
#include <cstdio>
#include <cmath>


// Helper function for interpolating iso-values between two vertices
__device__ float interpolate_iso_point(float v0, float v1, float f0, float f1, float iso_value) {
    return v0 + (iso_value - f0) * (v1 - v0) / (f1 - f0);
}

// CUDA kernel for extracting iso-contour segments
// marching triangles at each iso-value yields point pairs
// shared edges have repeated interpolated points, then merge 
template <typename scalar_t>
__global__ void extract_contours_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> vertices,
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits> faces,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> fields,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> iso_values,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> contour_points,
    torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits> pairpoint_nums,
    const int num_faces,
    const int num_iso_values)
{
    // parallel along triangle faces
    int face_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_id >= num_faces) return;

    // Load vertex indices for this face
    int v0_idx = faces[face_id][0];
    int v1_idx = faces[face_id][1];
    int v2_idx = faces[face_id][2];

    // Load field values at each vertex for the current face
    scalar_t f0 = fields[v0_idx];
    scalar_t f1 = fields[v1_idx];
    scalar_t f2 = fields[v2_idx];

    // Iterate over iso-values to find contour points
    for (int iso_id = 0; iso_id < num_iso_values; ++iso_id) {
        scalar_t iso_value = iso_values[iso_id];

        // Check if the iso-value crosses between vertices
        bool below[3] = {f0 <= iso_value, f1 <= iso_value, f2 <= iso_value};
        bool above[3] = {f0 >= iso_value, f1 >= iso_value, f2 >= iso_value};

        // If the iso-value does not cross the triangle, skip this face
        if ((below[0] && below[1] && below[2]) || (above[0] && above[1] && above[2])) {
            continue;  // No contour crosses this triangle for this iso-value
        }

        // Find intersection points by interpolating along the edges
        scalar_t points[6];  // Store up to two points (each with 3 coordinates)
        int point_count = 0;

        // Interpolate along the edge v0-v1
        if ((f0 - iso_value) * (f1 - iso_value) < 0) {
            for (int i = 0; i < 3; i++) {
                points[point_count++] = interpolate_iso_point(
                    vertices[v0_idx][i], vertices[v1_idx][i], f0, f1, iso_value);
            }
        }

        // Interpolate along the edge v1-v2
        if ((f1 - iso_value) * (f2 - iso_value) < 0) {
            for (int i = 0; i < 3; i++) {
                points[point_count++] = interpolate_iso_point(
                    vertices[v1_idx][i], vertices[v2_idx][i], f1, f2, iso_value);
            }
        }

        // Interpolate along the edge v2-v0
        if ((f2 - iso_value) * (f0 - iso_value) < 0) {
            for (int i = 0; i < 3; i++) {
                points[point_count++] = interpolate_iso_point(
                    vertices[v2_idx][i], vertices[v0_idx][i], f2, f0, iso_value);
            }
        }

        // Only store the contour if there are exactly two intersection points
        if (point_count == 6) {
            // Atomically increment the index for storing points and get current isovalue index
            int idx = atomicAdd(&pairpoint_nums[iso_id], 1);  // at same iso-value, how many contours are extracted

            // Store the two points into the output tensor
            for (int i = 0; i < 6; ++i) {
                contour_points[iso_id][idx][i] = points[i];    // continous store p1 (x, y, z), p2 (x, y, z)
            }
        }
    }
}

// CUDA kernel for merging point pairs, multiple subcontours in 2nd dim, but insert -1e9 as separator
// subcontours are stored in order without repeated points 
// subcontours are separated by -1e9, NOTE even single contour has a separator at the end.
// [subcontour1, -1e9, 00000...], [subcontour1, -1e9, subcontour2, -1e9, 00000...]
template <typename scalar_t>
__global__ void merge_segments_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> contour_points,
    const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits> pairpoint_nums,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> contours_merged,
    torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits> contour_nums,

    bool* used_global, // use global memory
    const int num_faces,
    const int num_iso_values)
{
    // const float merge_threshold = 1e-6 * 1e-6; // Threshold for merging points
    const float merge_threshold = 1e-6 * 1e-6; // Threshold for merging points
    int iso_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (iso_id >= num_iso_values) return;

    int pair_num = pairpoint_nums[iso_id];
    if (pair_num == 0) return; // no pairs, no need to merge

    // contour_points [num_isovalues, num_faces, 6] stores pairs of intersection points across all triangles 
    // pairpoint_nums [num_isovalues] stores how many pairs are extracted at each iso-value

    // use global memory to store used flags, get current thread used.
    bool* used = used_global + iso_id * num_faces;
    
    // merge pairs
    int current_contour_index = 0;
    int contours_found = 0;

    scalar_t last_point[3]; 

    for (int face_id = 0; face_id < pair_num; ++face_id) {
        if (used[face_id]) continue;

        // start a new contour and set a separator at (x, y, z) = (-1e9, -1e9, -1e9)
        // if more than one subcontours wrt iso-value
        if (contours_found > 0) {
            for (int i = 0; i < 3; ++i) { 
                contours_merged[iso_id][current_contour_index][i] = -1e9; // Separator marker
            }
            current_contour_index++;
        }

        // start from first point in pair at current contour 
        for (int i = 0; i < 3; ++i) {
            contours_merged[iso_id][current_contour_index][i] = contour_points[iso_id][face_id][i];
        }
        current_contour_index++;
        used[face_id] = true;
        contours_found++;

        // last point in pair
        for (int i = 0; i < 3; ++i) {
            last_point[i] = contour_points[iso_id][face_id][3 + i];
        }

        // iterate over remaining pairs to merge
        bool found_segment = true;
        while (found_segment) {
            found_segment = false;

            for (int next_face_id = 0; next_face_id < pair_num; ++next_face_id) {
                if (used[next_face_id]) continue;

                // Retrieve both points of the next segment for comparison
                scalar_t p1[3], p2[3];
                for (int i = 0; i < 3; ++i) {
                    p1[i] = contour_points[iso_id][next_face_id][i];
                    p2[i] = contour_points[iso_id][next_face_id][3 + i];
                }
                // check with distance for connection
                scalar_t d1 = (last_point[0] - p1[0]) * (last_point[0] - p1[0]) +
                              (last_point[1] - p1[1]) * (last_point[1] - p1[1]) +
                              (last_point[2] - p1[2]) * (last_point[2] - p1[2]);

                scalar_t d2 = (last_point[0] - p2[0]) * (last_point[0] - p2[0]) +
                              (last_point[1] - p2[1]) * (last_point[1] - p2[1]) +
                              (last_point[2] - p2[2]) * (last_point[2] - p2[2]);

                if (d1 < merge_threshold) {
                    // Add p2 to contours_merged
                    for (int i = 0; i < 3; ++i) {
                        contours_merged[iso_id][current_contour_index][i] = p2[i];
                        last_point[i] = p2[i]; // Update the last point
                    }
                    current_contour_index++;
                    used[next_face_id] = true;
                    found_segment = true;
                    break;
                } else if (d2 < merge_threshold) {
                    // Add p1 to contours_merged
                    for (int i = 0; i < 3; ++i) {
                        contours_merged[iso_id][current_contour_index][i] = p1[i];
                        last_point[i] = p1[i]; // Update the last point
                    }
                    current_contour_index++;
                    used[next_face_id] = true;
                    found_segment = true;
                    break;
                }
            }

        }

    }
    // Add a final separator after the last contour, even a single contour
    if (contours_found > 0) {
        for (int i = 0; i < 3; ++i) {
            contours_merged[iso_id][current_contour_index][i] = -1e9;
        }
        current_contour_index++;
    }
    // how many contours for one iso-value
    contour_nums[iso_id] = contours_found;
}


std::vector<torch::Tensor> extract_isocontours_cu(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& fields,
    const torch::Tensor& iso_values)
{
    // iso_contours[iso_value] = [contour1, contour2[subcontour1, subcontour2], ...]

    int num_faces = faces.size(0);
    int num_iso_values = iso_values.size(0);

    // std::cout << "num faces " << num_faces << "\n" << "num isovalues "<< num_iso_values << std::endl;

    // every iso-value has num_faces space to store point pairs, even no intersections at an iso-value
    torch::Tensor contour_points = torch::zeros({num_iso_values, num_faces, 6}, vertices.options());
    // for each iso-value, how many contours are extracted
    torch::Tensor pairpoint_nums = torch::zeros({num_iso_values}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device()));
    // merged contours, insert sperator (-1e9) for multiple subcontours since all values in range [-1, 1]
    torch::Tensor contours_merged = torch::zeros({num_iso_values, num_faces, 3}, vertices.options());
    // how many contours are extracted at each iso-value
    torch::Tensor contour_nums = torch::zeros({num_iso_values}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device()));

    int threads = 256;
    int blocks = (num_faces + threads - 1) / threads;
    int blocks_merge = (num_iso_values + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(vertices.type(), "extract_isocontours_parallel", 
    ([&] {
        extract_contours_kernel<scalar_t><<<blocks, threads>>>(
            vertices.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            faces.packed_accessor<int32_t, 2, torch::RestrictPtrTraits>(),
            fields.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits>(),
            iso_values.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits>(),
            contour_points.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            pairpoint_nums.packed_accessor<int32_t, 1, torch::RestrictPtrTraits>(),

            num_faces, num_iso_values
        );

        // check if kernel launch is successful
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        cudaDeviceSynchronize(); // ensure finish first kernel before next kernel

        // std::cout << "First kernel finished" << std::endl;

        // allocate and init global memory for used flags to avoid memory issuses in kernel
        bool* used_global;
        cudaMalloc(&used_global, num_iso_values * num_faces * sizeof(bool));
        cudaMemset(used_global, 0, num_iso_values * num_faces * sizeof(bool));

        merge_segments_kernel<scalar_t><<<blocks_merge, threads>>>(
            contour_points.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            pairpoint_nums.packed_accessor<int32_t, 1, torch::RestrictPtrTraits>(),
            contours_merged.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            contour_nums.packed_accessor<int32_t, 1, torch::RestrictPtrTraits>(),
            used_global,
            num_faces, num_iso_values
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        // everything is completed before returning
        cudaDeviceSynchronize();
        cudaFree(used_global);

        // std::cout << "Second kernel finished" << std::endl;

    }));

    return {contours_merged, contour_nums};
}


