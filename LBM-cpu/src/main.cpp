#include <iostream>
#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include "Grid2d.h"
#include "tinycolormap.hpp"

const int res_x = 800, res_y = 400;
const int N = 8;

void updateColor(float *frame_data, float *frame_buffer) {
    for (int x = 0; x < res_x; x++) {
        for (int y = 0; y < res_y; y++) {
            const tinycolormap::Color color
                = tinycolormap::GetColor(frame_data[x + y * res_x], tinycolormap::ColormapType::Plasma);
            frame_buffer[(x + res_x * y) * 3] = color.r();
            frame_buffer[(x + res_x * y) * 3+1] = color.g();
            frame_buffer[(x + res_x * y) * 3+2] = color.b();
        }
    }
}

int main() {
    Grid2d grid(res_x, res_y);

    float *frame_buffer = new float[res_x * res_y * 3];
    float *frame_data = new float[res_x * res_y];

    omp_set_num_threads(16);
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(res_x, res_y, "LBM-test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    unsigned int step = 0;
    while(!glfwWindowShouldClose(window)) {
        grid.output(frame_data);
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        grid.simNStep(N);

        updateColor(frame_data, frame_buffer);
        glDrawPixels(res_x, res_y, GL_RGB, GL_FLOAT, frame_buffer);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    delete[] frame_buffer;
    delete[] frame_data;

    return 0;
}

