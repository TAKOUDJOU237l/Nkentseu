// ============================================================
// FILE: Vec4d.h
// TP6 — Vecteur 4D double précision (coordonnées homogènes)
// ============================================================
#pragma once
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstddef>
#include "../tp3-float-h/Float.h"
#include "../tp5-vec3d-gramschmidt/Vec3d.h"

namespace NkMath {

struct Vec4d {
    double x, y, z, w;

    // Constructeurs
    Vec4d() : x(0.0), y(0.0), z(0.0), w(0.0) {}
    Vec4d(double x, double y, double z, double w)
        : x(x), y(y), z(z), w(w) {}

    // Depuis Vec3d
    // w=1 → POINT (translation s'applique)
    // w=0 → DIRECTION (translation ne s'applique PAS)
    Vec4d(const Vec3d& v, double w) : x(v.x), y(v.y), z(v.z), w(w) {}

    // Accès par index
    double& operator[](int i) {
        assert(i >= 0 && i < 4 && "Vec4d index out of bounds");
        return (&x)[i];
    }
    const double& operator[](int i) const {
        assert(i >= 0 && i < 4 && "Vec4d index out of bounds");
        return (&x)[i];
    }

    // Opérateurs arithmétiques
    Vec4d operator+(const Vec4d& o) const {
        return {x+o.x, y+o.y, z+o.z, w+o.w};
    }
    Vec4d operator-(const Vec4d& o) const {
        return {x-o.x, y-o.y, z-o.z, w-o.w};
    }
    Vec4d operator*(double s) const {
        return {x*s, y*s, z*s, w*s};
    }
    Vec4d operator/(double s) const {
        assert(!nearlyZero(s) && "Division par zero");
        return {x/s, y/s, z/s, w/s};
    }
    Vec4d operator-() const { return {-x, -y, -z, -w}; }

    // Opérateurs composés
    Vec4d& operator+=(const Vec4d& o) {
        x+=o.x; y+=o.y; z+=o.z; w+=o.w; return *this;
    }
    Vec4d& operator-=(const Vec4d& o) {
        x-=o.x; y-=o.y; z-=o.z; w-=o.w; return *this;
    }
    Vec4d& operator*=(double s) {
        x*=s; y*=s; z*=s; w*=s; return *this;
    }
    Vec4d& operator/=(double s) {
        assert(!nearlyZero(s) && "Division par zero");
        x/=s; y/=s; z/=s; w/=s; return *this;
    }

    // Égalité
    bool operator==(const Vec4d& o) const {
        return approxEq(x,o.x) && approxEq(y,o.y) &&
               approxEq(z,o.z) && approxEq(w,o.w);
    }
    bool operator!=(const Vec4d& o) const { return !(*this == o); }

    // Dot product 4D
    double Dot(const Vec4d& o) const {
        return x*o.x + y*o.y + z*o.z + w*o.w;
    }

    // Norme
    double Norm2() const { return x*x + y*y + z*z + w*w; }
    double Norm()  const { return std::sqrt(Norm2()); }

    // Déhomogénéisation → Vec3d (divise par w)
    // w=0 → direction, pas de division
    Vec3d ToVec3() const {
        assert(!nearlyZero(w) &&
               "Cannot dehomogenize: w=0 (direction)");
        return {x/w, y/w, z/w};
    }

    // Partie XYZ sans division
    Vec3d XYZ() const { return {x, y, z}; }

    void Print() const {
        printf("Vec4d(%.6f, %.6f, %.6f, %.6f)\n", x, y, z, w);
    }
};

// Commutativité scalaire
inline Vec4d operator*(double s, const Vec4d& v) { return v * s; }

// Garantie layout mémoire
static_assert(sizeof(Vec4d) == 32,
    "Vec4d doit faire 32 bytes (4 doubles)");
static_assert(offsetof(Vec4d, x) == 0,  "x en premier");
static_assert(offsetof(Vec4d, y) == 8,  "y a offset 8");
static_assert(offsetof(Vec4d, z) == 16, "z a offset 16");
static_assert(offsetof(Vec4d, w) == 24, "w a offset 24");

} // namespace NkMath