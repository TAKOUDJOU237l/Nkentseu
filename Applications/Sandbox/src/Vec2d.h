// ============================================================
// FILE: Vec2d.h
// TP4 — Vecteur 2D double précision
// ============================================================
#pragma once
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstddef>  // offsetof
#include "../nkentseu/Modules/Foundation/NKMath/src/NKMath/Float.h"

namespace NkMath {

struct Vec2d {
    double x, y;

    // Constructeurs
    Vec2d() : x(0.0), y(0.0) {}
    Vec2d(double x, double y) : x(x), y(y) {}
    explicit Vec2d(double s) : x(s), y(s) {}  // fill constructor

    // Accès par index — légal C++17 standard-layout
    double& operator[](int i) {
        assert(i >= 0 && i < 2 && "Vec2d index out of bounds");
        return (&x)[i];
    }
    const double& operator[](int i) const {
        assert(i >= 0 && i < 2 && "Vec2d index out of bounds");
        return (&x)[i];
    }

    // Opérateurs arithmétiques
    Vec2d operator+(const Vec2d& o) const { return {x+o.x, y+o.y}; }
    Vec2d operator-(const Vec2d& o) const { return {x-o.x, y-o.y}; }
    Vec2d operator*(double s)       const { return {x*s,   y*s  }; }
    Vec2d operator/(double s)       const {
        assert(!nearlyZero(s) && "Division par zero");
        return {x/s, y/s};
    }
    Vec2d operator-() const { return {-x, -y}; }

    // Opérateurs composés
    Vec2d& operator+=(const Vec2d& o) { x+=o.x; y+=o.y; return *this; }
    Vec2d& operator-=(const Vec2d& o) { x-=o.x; y-=o.y; return *this; }
    Vec2d& operator*=(double s)       { x*=s;   y*=s;   return *this; }
    Vec2d& operator/=(double s)       {         // ← manquait
        assert(!nearlyZero(s) && "Division par zero");
        x/=s; y/=s;
        return *this;
    }

    // Égalité
    bool operator==(const Vec2d& o) const {
        return approxEq(x, o.x) && approxEq(y, o.y);
    }
    bool operator!=(const Vec2d& o) const { return !(*this == o); }

    // Norme
    double Norm2() const { return x*x + y*y; }
    double Norm()  const { return std::sqrt(Norm2()); }

    // Normalisation — retourne (0,0) si vecteur nul — jamais NaN
    Vec2d Normalized() const {
        double n = Norm();
        if (nearlyZero(n)) return Vec2d(0.0);
        return {x/n, y/n};
    }

    bool IsNormalized(double eps = kEps) const {
        return approxEq(Norm2(), 1.0, eps);
    }

    // Affichage
    void Print() const {                        // ← manquait
        printf("Vec2d(%.6f, %.6f)\n", x, y);
    }
};

// Dot product : a·b = |a||b|cos(θ)
inline double Dot(const Vec2d& a, const Vec2d& b) {
    return a.x*b.x + a.y*b.y;
}

// Cross 2D (scalaire)
// Positif si b est à gauche de a, négatif si à droite
inline double Cross2D(const Vec2d& a, const Vec2d& b) {
    return a.x*b.y - a.y*b.x;
}

// Lerp
inline Vec2d Lerp(const Vec2d& a, const Vec2d& b, double t) {
    return {a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t};
}

// Commutativité scalaire : s * v == v * s
inline Vec2d operator*(double s, const Vec2d& v) { return v * s; }

// Garantie layout mémoire pour glVertexAttribPointer
static_assert(sizeof(Vec2d) == 16,
    "Vec2d doit faire 16 bytes (2 doubles)");
static_assert(offsetof(Vec2d, x) == 0,   
    "x doit etre en premier");
static_assert(offsetof(Vec2d, y) == 8,   
    "y doit etre a l offset 8");

} // namespace NkMath