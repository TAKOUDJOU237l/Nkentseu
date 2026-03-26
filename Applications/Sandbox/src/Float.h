// ============================================================
// FILE: Float.h
// TP3 — Utilitaires arithmétique flottante
// ============================================================
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>

namespace NkMath {

    // -------------------------------------------------------
    // Constantes
    // -------------------------------------------------------
    constexpr double kEps  = 1e-9;
    constexpr float  kFEps = 1e-6f;
    constexpr float  kPiF  = 3.14159265358979323846f;
    constexpr double kPi   = 3.14159265358979323846;
    constexpr float  kDegToRadF = kPiF / 180.0f;
    constexpr double kDegToRad  = kPi  / 180.0;

    // -------------------------------------------------------
    // isFiniteValid — ni NaN ni Inf
    // -------------------------------------------------------
    inline bool isFiniteValid(float  x) { return std::isfinite(x); }
    inline bool isFiniteValid(double x) { return std::isfinite(x); }

    // -------------------------------------------------------
    // nearlyZero
    // -------------------------------------------------------
    inline bool nearlyZero(float  x, float  eps = kFEps) {
        return std::abs(x) < eps;
    }
    inline bool nearlyZero(double x, double eps = kEps) {
        return std::abs(x) < eps;
    }

    // -------------------------------------------------------
    // approxEq — égalité relative
    // -------------------------------------------------------
    inline bool approxEq(float a, float b, float eps = kFEps) {
        if (a == b) return true;
        float maxAB = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= eps * std::max(1.0f, maxAB);
    }
    inline bool approxEq(double a, double b, double eps = kEps) {
        if (a == b) return true;
        double maxAB = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= eps * std::max(1.0, maxAB);
    }

    // -------------------------------------------------------
    // kahanSum
    // -------------------------------------------------------
    inline float kahanSum(const float* data, int n) {
        float sum  = 0.0f;
        float comp = 0.0f;
        for (int i = 0; i < n; i++) {
            float y = data[i] - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum  = t;
        }
        return sum;
    }
    inline double kahanSum(const double* data, int n) {
        double sum  = 0.0;
        double comp = 0.0;
        for (int i = 0; i < n; i++) {
            double y = data[i] - comp;
            double t = sum + y;
            comp = (t - sum) - y;
            sum  = t;
        }
        return sum;
    }

    // -------------------------------------------------------
    // ULP — Unit in the Last Place
    // -------------------------------------------------------
    inline float ulp(float x) {
        uint32_t bits;
        std::memcpy(&bits, &x, 4);
        bits++;
        float next;
        std::memcpy(&next, &bits, 4);
        return next - x;
    }

    // -------------------------------------------------------
    // Clamp
    // -------------------------------------------------------
    inline float  Clamp(float  v, float  lo, float  hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    inline double Clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

} // namespace NkMath