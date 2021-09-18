//
// Created by ryan on 2021/8/19.
//

#ifndef LBMPLAYGROUND_VEC_H
#define LBMPLAYGROUND_VEC_H

//template<typename T>
//class vec2 {
//public:
//    template<typename T1> friend class vec2;
//
//    vec2() : _x(0), _y(0) {}
//    vec2(T x, T y): _x(x), _y(y) {}
//
//    template<typename T1>
//    inline auto operator*(const vec2<T1> rhs) {
//        return vec2<T>(rhs._x * _x, rhs._y * _y);
//    }
//
//    template<typename T1>
//    inline vec2<T> operator*(const T1 rhs) {
//        return vec2<T>(rhs * _x, rhs * _y);
//    }
//
//    template<typename T1>
//    inline vec2<T> operator/(const T1 rhs) {
//        return vec2<T>(rhs / _x, rhs / _y);
//
//    }
//
//    template<typename T1, typename T2>
//    inline auto operator+(const vec2<T1> rhs) {
//        return vec2<T2>(rhs._x + _x, rhs._y + _y);
//    }
//
//    template<typename T1>
//    inline auto& operator+=(const vec2<T1> &rhs) {
//        _x += rhs._x;
//        _y += rhs._y;
//        return *this;
//    }
//
//    template<typename RT, typename T1>
//    RT dot(const vec2<T1> &rhs) {
//        return rhs._x * _x + rhs._y * _y;
//    }
//
//    inline T x() const { return _x; }
//    inline T y() const { return _y; }
//private:
//    T _x, _y;
//};

#endif //LBMPLAYGROUND_VEC_H
