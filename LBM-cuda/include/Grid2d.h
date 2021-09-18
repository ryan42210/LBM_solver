//
// Created by ryan on 2021/8/19.
//

#ifndef LBM_CUDA_GRID2D_H
#define LBM_CUDA_GRID2D_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "cuda_math_helper.h"

struct GridPoint {
    float rho;
    vec2 u;
    float fi[9];
};

class Grid2d {
public:
    Grid2d(size_t res_x, size_t res_y) : res_x(res_x), res_y(res_y) {

    }

    void initialize() {

    }


    inline __device__ void stream(int x, int y, int i){

    }


private:
    size_t res_x, res_y;

};

#endif //LBM_CUDA_GRID2D_H
