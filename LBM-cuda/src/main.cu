#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include "cuda_math_helper.h"
#include "tinycolormap.hpp"
#include <omp.h>
#include <thread>


// basic setting -----------------------------------------
// const int grid_x = 64;
// const int grid_y = 32;
// const int block_x = 16;
// const int block_y = 16;
const int grid_x = 1024;
const int grid_y = 2;
const int block_x = 1;
const int block_y = 256;
const int width = grid_x * block_x;
const int height = grid_y * block_y;
const int size = width * height;
const int step_per_frame = 20;
__constant__ const int INIT_RHO = 2;
__constant__ const int dim = 9;
__constant__ const float tao = 1.0f / (0.5 + 3 * 0.02);
__constant__ const float INIT_V = 0.1;

__constant__ float wi[9] = {4.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/36, 1.0f/36, 1.0f/36, 1.0f/36};
__constant__ float ci_x[9] = {0, 1, -1, 0, 0, 1, 1, -1, -1};
__constant__ float ci_y[9] = {0, 0, 0, 1, -1, 1, -1, 1, -1};
__constant__ float O_x = width / 4;
__constant__ float O_y = height / 2;
__constant__ float radius = 30;
// basic setting



// LBM functions ------------------------------------------
__forceinline__ __device__ int id(int x, int y, int i = 0) {
  // int offset = i * gridDim.x * gridDim.y * blockDim.x * blockDim.y;
  // int block_idx_x = x / blockDim.x;
  // int thread_idx_x = x % blockDim.x;
  // int block_idx_y = y / blockDim.y;
  // int thread_idx_y = y % blockDim.y;
  // return gridDim.x * block_idx_y + block_idx_x + thread_idx_y * blockDim.x + thread_idx_x + offset;
  return i * width * height + y * width + x;
}


__forceinline__ __device__ int getX() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__forceinline__ __device__ int getY() {
  return blockIdx.y * blockDim.y + threadIdx.y;
}

__device__ bool InGeometry(int x, int y) {
  return (x-O_x)*(x-O_x) + (y-O_y)*(y-O_y) < radius * radius;
}

__device__ float fBar(int x, int y, int i, float *rho, float2 *u) {
  int idx = id(x, y);
  float cdotu= ci_x[i] * u[idx].x + ci_y[i] * u[idx].y;
  float f_bar = wi[i] * rho[idx] * (1 + cdotu * 3.0f
    + cdotu * cdotu * 4.5f - length2(u[idx]) * 1.5f);
  return f_bar;
}

// inline __device__ bool inGeometry(int x, int y) {
//   float2 p = make_float2(x, y);
//   float2 O = make_float2(width/4, height/2);
//   if(length2(p - O) < 100) {
//     return true;
//   }
//
//   // if (x < 0 || x >= width) return true;
//   // if (y < 0 || x >= height) return true;
//   return false;
// }

__global__ void Initialize(float* fi_new, float* fi_old, float* rho, float2* u) {
  int x = getX();
  int y = getY();
  int idx = id(x, y);
  u[idx] = make_float2(0, 0);
  // if (x == 0) {
  //   u[id(x, y)] = make_float2(INIT_V, 0);
  // }
  rho[idx] = INIT_RHO;
  for (int i = 0; i < dim; ++i) {
    fi_new[id(x, y, i)] = fBar(x, y, i, rho, u);
    fi_old[id(x, y, i)] = fi_new[id(x, y, i)];
  }
}

__global__ void CollideAndStream(float* fi_new, float* fi_old, float *rho, float2* u) {
  int x = getX();
  int y = getY();
  if (x == 0 || x == width - 1 || y == 0 || y== height - 1) return;

  for (int i = 0; i < 9; ++i) {
    int idx_f = id(x, y, i);
    int x_prime = x - ci_x[i];
    int y_prime = y - ci_y[i];

    int idx_f_source = id(x_prime, y_prime, i);
    fi_new[idx_f] = fi_old[idx_f_source]
        - tao * (fi_old[idx_f_source] - fBar(x_prime, y_prime, i, rho, u));
  }
}

__global__ void UpdateMacroVar(float *fi_new, float *fi_old, float *rho, float2* u) {
  int x = getX();
  int y = getY();
  if (x == 0 || x == width - 1 || y == 0 || y== height - 1) return;

  int idx = id(x, y);
  float f = 0;
  float2 f_c = make_float2(0,0);
  for (int i = 0; i < dim; ++i) {
    int idx_f = id(x, y, i);
    fi_old[idx_f] = fi_new[idx_f];
    f += fi_new[idx_f];
    f_c += make_float2(fi_new[idx_f] * ci_x[i], fi_new[idx_f] * ci_y[i]);
  }
  rho[idx] = f;
  u[idx] = f_c / f;
}

__device__ void BoundaryCore(int x, int y, int x_source, int y_source, bool use_source,
                             float *fi_old, float *rho, float2 *u, float2 assigned_u) {
  int idx = id(x, y);
  int idx_source = id(x_source, y_source);
  if (use_source) {
    u[idx] = u[idx_source];
  } else {
    u[idx] = assigned_u;
  }
  rho[idx] = rho[idx_source];
  for (int i = 0; i < 9; i++) {
    int idx_f = id(x, y, i);
    int idx_f_source = id(x_source, y_source, i);
    fi_old[idx_f] = fBar(x, y, i, rho, u) - fBar(x_source, y_source, i, rho, u) + fi_old[idx_f_source];
  }
}

__global__ void ApplyBoundaryCondition(float *fi_old, float *rho, float2 *u) {
  int x = getX();
  int y = getY();
  // top and bottom boundary
  if (y == 0) {
    BoundaryCore(x, y, x, 1, false, fi_old, rho, u, make_float2(0,0));
  } else if (y == height -1) {
    BoundaryCore(x, y, x, height - 2, false, fi_old, rho, u, make_float2(0,0));
  } else if (x == 0) {
    BoundaryCore(x, y, 1, y, false, fi_old, rho, u, make_float2(INIT_V, 0));
  } else if (x == width - 1) {
    BoundaryCore(x, y, width-2, y, true, fi_old, rho, u, make_float2(0,0));
  } else if (InGeometry(x, y)) {
    int x_tgt, y_tgt;
    if (x >= O_x) x_tgt = x+1;
    else x_tgt = x-1;
    if (y >= O_y) y_tgt = y+1;
    else y_tgt = y-1;
    BoundaryCore(x, y, x_tgt, y_tgt, false, fi_old, rho, u, make_float2(0, 0));
  }
}

// LBM functions

inline __forceinline__ __device__ float LinerInterp(float data, float low, float up) {
  return (up -data) / (up - low);
}

__global__ void GenerateFrameData(float2 *u, unsigned char *frame_buffer) {
  int x = getX();
  int y = getY();
  int idx = id(x, y);
  float data = length(u[idx]) / 0.24;
  unsigned char r, g, b;
  if (data < 0.14) {
    float alpha = LinerInterp(data, 0, 0.14);
    r = alpha * 13 + (1-alpha) * 84;
    g = alpha * 8 + (1-alpha) * 2;
    b = alpha * 135 + (1-alpha) * 163;
  } else if (data < 0.29) {
    float alpha = LinerInterp(data, 0.14, 0.29);
    r = alpha * 84 + (1-alpha) * 139;
    g = alpha * 2 + (1-alpha) * 10;
    b = alpha * 163 + (1-alpha) * 165;
  } else if (data < 0.43) {
    float alpha = LinerInterp(data, 0.29, 0.43);
    r = alpha * 139 + (1-alpha) * 185;
    g = alpha * 10 + (1-alpha) * 50;
    b = alpha * 165 + (1-alpha) * 137;
  } else if (data < 0.57) {
    float alpha = LinerInterp(data, 0.43, 0.57);
    r = alpha * 185 + (1-alpha) * 219;
    g = alpha * 50 + (1-alpha) * 92;
    b = alpha * 137 + (1-alpha) * 104;
  } else if (data < 0.71) {
    float alpha = LinerInterp(data, 0.57, 0.71);
    r = alpha * 219 + (1-alpha) * 244;
    g = alpha * 92 + (1-alpha) * 136;
    b = alpha * 104 + (1-alpha) * 73;
  } else if (data < 0.86) {
    float alpha = LinerInterp(data, 0.71, 0.86);
    r = alpha * 244 + (1-alpha) * 254;
    g = alpha * 136 + (1-alpha) * 188;
    b = alpha * 73 + (1-alpha) * 43;
  } else if (data < 1) {
    float alpha = LinerInterp(data, 0.86, 1);
    r = alpha * 254 + (1-alpha) * 240;
    g = alpha * 188 + (1-alpha) * 249;
    b = alpha * 43 + (1-alpha) * 33;
  } else {
    r = 240;
    g = 249;
    b = 33;
  }
  frame_buffer[3 * idx] = r;
  frame_buffer[3 * idx + 1] = g;
  frame_buffer[3 * idx + 2] = b;
}


int SimulateNRound(dim3 gridSize, dim3 blockSize, int num_cycles, float *fi_new, float *fi_old, float *rho, float2* u, unsigned char *d_frame_buffer) {
  for (int i = 0; i < num_cycles; ++i) {
    CollideAndStream<<<gridSize, blockSize>>>(fi_new, fi_old, rho, u);
    if ( cudaSuccess != cudaGetLastError() ) return 1;
    UpdateMacroVar<<<gridSize, blockSize>>>(fi_new, fi_old, rho, u);
    if ( cudaSuccess != cudaGetLastError() ) return 2;
    ApplyBoundaryCondition<<<gridSize, blockSize>>>(fi_old, rho, u);
    if ( cudaSuccess != cudaGetLastError() ) return 3;
  }
  GenerateFrameData<<<gridSize, blockSize>>>(u, d_frame_buffer);
  if ( cudaSuccess != cudaGetLastError() ) return 4;
  else return 0;
}


int main() {
  // allocate space on device ---------------------------
  float *d_rho;
  float2 *d_u;
  float *d_fi_old;
  float *d_fi_new;
  unsigned char *d_frame_buffer;

  cudaMalloc(&d_fi_old, size * sizeof(float) * 9);
  cudaMalloc(&d_fi_new, size * sizeof(float) * 9);
  cudaMalloc(&d_rho, size * sizeof(float));
  cudaMalloc(&d_u, size * sizeof(float2));
  cudaMalloc(&d_frame_buffer, sizeof(unsigned char) * size * 3);

  float *frame_buffer = (float *)malloc(sizeof(unsigned char) * size * 3);

  // window--------------------------------------
  glfwInit();
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  GLFWwindow *window = glfwCreateWindow(width, height, "LBM", NULL, NULL);
  glfwMakeContextCurrent(window);

  dim3 gridSize(grid_x, grid_y);
  dim3 blockSize(block_x, block_y);

  Initialize<<<gridSize, blockSize>>>(d_fi_new, d_fi_old, d_rho, d_u);

  glewInit();
  unsigned int cnt = 0;
  while (!glfwWindowShouldClose(window)) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, true);

    int error = SimulateNRound(gridSize, blockSize, step_per_frame, d_fi_new, d_fi_old, d_rho, d_u, d_frame_buffer);
    if (error != 0) break;
    cudaMemcpy(frame_buffer, d_frame_buffer, 3 * size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cnt * step_per_frame % 1000 == 0) {
      printf("step num: %d @%dstep/frame\n", cnt*step_per_frame, step_per_frame);
    }
    cnt++;

    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, frame_buffer);

    glfwPollEvents();
    glfwSwapBuffers(window);
  }
  glfwTerminate();
  // window--------------------------------------

  std::cout << "loop terminated when frame = " << cnt << std::endl;

  cudaFree(d_fi_old);
  cudaFree(d_fi_new);
  cudaFree(d_rho);
  cudaFree(d_u);
  cudaFree(d_frame_buffer);

  free(frame_buffer);

  return 0;
}
