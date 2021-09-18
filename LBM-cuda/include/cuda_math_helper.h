//
// Created by ryan on 2021/8/23.
//

#ifndef LBM_CUDA_CUDA_MATH_HELPER_H
#define LBM_CUDA_CUDA_MATH_HELPER_H

#define __hostdev__ __host__ __device__

using vec2 = float2;

inline __hostdev__ vec2 operator-(const vec2 &v) {
  return make_float2(-v.x, -v.y);
}

inline __hostdev__ vec2 operator*(const vec2 &vec, const float v) {
  return make_float2(vec.x * v, vec.y * v);
}

inline __hostdev__ vec2 operator*(const float v, const vec2 &vec) {
  return make_float2(vec.x * v, vec.y * v);
}

inline __hostdev__ vec2 operator*(const vec2 &a, const vec2 &b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

inline __hostdev__ vec2 &operator*=(vec2 &a, const vec2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

inline __hostdev__ vec2 &operator*=(vec2 &a, const float v) {
  a.x *= v;
  a.y *= v;
  return a;
}

inline __hostdev__ vec2 operator+(const vec2 &a, const vec2 &b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __hostdev__ vec2 &operator+=(vec2 &a, const vec2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

inline __hostdev__ vec2 operator-(const vec2 &a, const vec2 &b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __hostdev__ float length(const vec2 &v) {
  return sqrtf(v.x * v.x + v.y * v.y);
}

inline __hostdev__ float length2(const vec2 &v) {
  return v.x * v.x + v.y * v.y;
}

inline __hostdev__ vec2 operator/(const vec2 &vec, const float v) {
  return vec * (1.0f / v);
}

inline __hostdev__ vec2 operator/=(vec2 &vec, const float v) {
  return vec *= (1.0f / v);
}

inline __hostdev__ float dot(const vec2 &a, const vec2 &b) {
  return a.x * b.x + a.y * b.y;
}

inline __hostdev__ vec2 normalize(const vec2 &v) {
  return v / length(v);
}

#endif //LBM_CUDA_CUDA_MATH_HELPER_H
