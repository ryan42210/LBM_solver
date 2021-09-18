//
// Created by ryan on 2021/8/18.
//

#ifndef LBMPLAYGROUND_GRID2D_H
#define LBMPLAYGROUND_GRID2D_H

#include <vector>
#include <glm/glm.hpp>
#include <omp.h>

using namespace glm;

const float kinematicViscosity = 0.01;
const float INIT_V = 0.1;
//const int REYNOLDS_NUMBER = 100000;
const float init_rho = 2.7;
const float wi[9] = {4.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/36, 1.0f/36, 1.0f/36, 1.0f/36};
const int ci_x[9] = {0, 1, -1, 0, 0, 1, 1, -1, -1};
const int ci_y[9] = {0, 0, 0, 1, -1, 1, -1, 1, -1};
const float cs = 0.577f;
const int back[9] = {0, 2, 1, 4, 3, 8, 7, 6, 5};
const float RADIUS = 40;

class Grid2d {
public:
    Grid2d(int resx, int resy) : res_x(resx), res_y(resy) {
        fi_old.resize(9 * res_x * res_y, 0);
        fi_new.resize(9 * res_x * res_y, 0);
        rho.resize(res_x * res_y, init_rho);
        u.resize(res_x * res_y, vec2(0,0));
        geometry.resize(res_x * res_y, 0);
        tao = 0.5 + 3.0 * kinematicViscosity;
        tao = 1 / tao;
        O = vec2(res_x / 4, res_y / 2);
        R =RADIUS;
        init();
        std::cout << "initialized" << std::endl;
    }


    void simNStep(int step) {
        for (int i = 0; i < step; ++i) {
            CollideAndStream();
            UpdateMacroVar();
            ApplyBoundaryCondition();
        }
    }

    void output(float *buffer) {
        for (int i = 0; i < res_x*res_y; ++i) {
            buffer[i] = u[i][0] * u[i][0] + u[i][1] *u[i][1];
            buffer[i] = sqrtf(buffer[i]) / 0.15;
        }
    }

private:
    inline int at(int x, int y, int i = 0) {
        return i * res_x * res_y + y * res_x + x;
    }

    float fBar(int x, int y, int i) {
        int idx = at(x, y);
        float cidotu = ci_x[i] * u[idx][0] + ci_y[i] * u[idx][1];
        float f_bar = wi[i] * rho[idx] * ( 1 + cidotu * 3.0f
                + cidotu * cidotu * 4.5f
                - dot(u[idx], u[idx]) * 1.5f);
        return f_bar;
    }


    void init() {
        for (int x = 0; x < res_x; ++x) {
            for (int y = 0; y < res_y; y++) {
                for (int i = 0; i < 9; ++i) {
                    fi_old[i * res_x * res_y + y * res_x + x] = fBar(x, y, i);
                    fi_new[i * res_x * res_y + y * res_x + x] = fBar(x, y, i);
                }
                vec2 P(x, y);
//                if (dot(P-O, P-O) < R*R) {
//                    geometry[at(x, y)] = 1.0f;
//                }
                if (x < 200 && x > 150 && y < res_y / 2 + 20 && y > res_y / 2 -20)
                    geometry[at(x, y)] = 1.0f;
            }
        }
    }


    void CollideAndStream() {
#pragma omp parallel for
        for (int x = 1; x < res_x-1; ++x) {
            for (int y = 1; y < res_y-1; ++y) {
                for (int i = 0; i < 9; i++) {
                    int idx_f = at(x, y, i);

                    int x_prime = x - ci_x[i];
                    int y_prime = y - ci_y[i];

                    int idx_f_source = at(x_prime, y_prime, i);
                    fi_new[idx_f] = fi_old[idx_f_source]
                            - tao * (fi_old[idx_f_source] - fBar(x_prime, y_prime, i));

                }
            }
        }
    }

    void UpdateMacroVar() {
//        fi_old = fi_new;
# pragma omp parallel for
        for (int x = 1; x < res_x-1; x++) {
            for (int y = 1; y < res_y-1; y++) {
                int idx = at(x, y);
                u[idx] = vec2(0,0);
                rho[idx] = 0.0f;

                for (int i = 0; i < 9; ++i) {
                    int idx_f = at(x, y, i);
                    fi_old[idx_f] = fi_new[idx_f];
                    rho[idx] += fi_new[at(x, y, i)];
                    u[idx] += vec2(fi_new[idx_f] * ci_x[i], fi_new[idx_f] * ci_y[i]);
                }
                u[idx][0] /= rho[idx];
                u[idx][1] /= rho[idx];
            }
        }
    }

    void ApplyBoundaryCondition() {
        // fist handle four boundary
        // left and right boundary
# pragma omp parallel for
        for (int y = 1; y < res_y - 1; y++) {
            u[at(0, y)] = vec2(INIT_V, 0);
            rho[at(0, y)] = rho[at(1, y)];
            u[at(res_x - 1, y)] = u[at(res_x - 2, y)];
            rho[at(res_x - 1, y)] = rho[at(res_x - 2, y)];
            for (int i = 0; i < 9; i++) {
                fi_old[at(0, y, i)] = fBar(0, y, i) - fBar(1, y, i) + fi_old[at(1, y, i)];
                fi_old[at(res_x - 1, y, i)]
                    = fBar(res_x - 1, y, i) - fBar(res_x - 2, y, i) + fi_old[at(res_x - 2, y, i)];
            }
        }
        // up and down boundary
# pragma omp parallel for
        for (int x = 0; x < res_x; x++) {
            u[at(x, 0)] = vec2(0,0);
            rho[at(x, 0)] = rho[at(x, 1)];
            u[at(x, res_y - 1)] = vec2(0,0);
            rho[at(x, res_y - 1)] = rho[at(x, res_y - 2)];
            for (int i = 0; i < 9; i++) {
                fi_old[at(x, 0, i)] = fBar(x, 0, i) - fBar(x, 1, i) + fi_old[at(x, 1, i)];
                fi_old[at(x, res_y - 1, i)]
                    = fBar(x, res_y - 1, i) - fBar(x, res_y - 2, i) + fi_old[at(x, res_y - 2, i)];
            }
        }
//         then handle geometry
# pragma omp parallel for
        for (int x = 1; x < res_x - 1; x++) {
            for (int y = 1; y < res_y - 1; y++) {
                if (geometry[at(x, y)] == 1.0f) {
                    int x_tgt, y_tgt;
                    if (x >= O[0]) x_tgt = x + 1;
                    else x_tgt = x - 1;
                    if (y >= O[1]) y_tgt = y + 1;
                    else y_tgt = y - 1;
                    u[at(x, y)] = vec2(0,0);
                    rho[at(x, y)] = rho[at(x_tgt, y_tgt)];
                    for (int i = 0; i < 9; i++) {
                        fi_old[at(x, y, i)] = fBar(x, y, i) - fBar(x_tgt, y_tgt, i) + fi_old[at(x_tgt, y_tgt, i)];
                    }
                }
            }
        }
    }

    int res_x, res_y;
    float tao;
    std::vector<float> fi_old;
    std::vector<float> fi_new;
    std::vector<float> geometry;
    std::vector<float> rho;
    std::vector<vec2> u;

    vec2 O;
    float R;
};

#endif //LBMPLAYGROUND_GRID2D_H
