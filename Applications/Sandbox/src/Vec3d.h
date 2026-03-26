// ============================================================
// FILE: Vec3d.h
// TP5 — Vecteur 3D double précision
// ============================================================
#pragma once
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstddef>  // offsetof
#include "../nkentseu/Modules/Foundation/NKMath/src/NKMath/Float.h"

namespace NkMath {

struct Vec3d {
    double x, y, z;

    // Constructeurs
    Vec3d() : x(0.0), y(0.0), z(0.0) {}
    Vec3d(double x, double y, double z) : x(x), y(y), z(z) {}
    explicit Vec3d(double s) : x(s), y(s), z(s) {}

    // Accès par index — légal C++17 standard-layout
    double& operator[](int i) {
        assert(i >= 0 && i < 3 && "Vec3d index out of bounds");
        return (&x)[i];
    }
    const double& operator[](int i) const {
        assert(i >= 0 && i < 3 && "Vec3d index out of bounds");
        return (&x)[i];
    }

    // Opérateurs arithmétiques
    Vec3d operator+(const Vec3d& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3d operator-(const Vec3d& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3d operator*(double s)       const { return {x*s,   y*s,   z*s  }; }
    Vec3d operator/(double s)       const {
        assert(!nearlyZero(s) && "Division par zero");
        return {x/s, y/s, z/s};
    }
    Vec3d operator-() const { return {-x, -y, -z}; }

    // Opérateurs composés
    Vec3d& operator+=(const Vec3d& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3d& operator-=(const Vec3d& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3d& operator*=(double s)       { x*=s;   y*=s;   z*=s;   return *this; }
    Vec3d& operator/=(double s)       {
        assert(!nearlyZero(s) && "Division par zero");
        x/=s; y/=s; z/=s;
        return *this;
    }

    // Égalité
    bool operator==(const Vec3d& o) const {
        return approxEq(x,o.x) && approxEq(y,o.y) && approxEq(z,o.z);
    }
    bool operator!=(const Vec3d& o) const { return !(*this == o); }

    // Norme
    double Norm2() const { return x*x + y*y + z*z; }
    double Norm()  const { return std::sqrt(Norm2()); }

    // Normalisation — retourne (0,0,0) si vecteur nul — jamais NaN
    Vec3d Normalized() const {
        double n = Norm();
        if (nearlyZero(n)) return Vec3d(0.0);
        return {x/n, y/n, z/n};
    }

    bool IsNormalized(double eps = kEps) const {
        return approxEq(Norm2(), 1.0, eps);
    }

    void Print() const {
        printf("Vec3d(%.6f, %.6f, %.6f)\n", x, y, z);
    }
};

// Dot product
inline double Dot(const Vec3d& a, const Vec3d& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Cross product 3D — règle main droite
// Résultat perpendiculaire à a et b
// Mnémotechnique permutations cycliques : x→y→z→x
inline Vec3d Cross(const Vec3d& a, const Vec3d& b) {
    return {
        a.y*b.z - a.z*b.y,  // composante x
        a.z*b.x - a.x*b.z,  // composante y
        a.x*b.y - a.y*b.x   // composante z
    };
}

// Lerp
inline Vec3d Lerp(const Vec3d& a, const Vec3d& b, double t) {
    return a + (b - a) * t;
}

// Projection de a sur b
inline Vec3d Project(const Vec3d& a, const Vec3d& b) {
    double b2 = b.Norm2();
    assert(!nearlyZero(b2) && "Cannot project onto zero vector");
    return b * (Dot(a, b) / b2);
}

// Rejection de a par rapport à b
inline Vec3d Reject(const Vec3d& a, const Vec3d& b) {
    return a - Project(a, b);
}

// Commutativité scalaire
inline Vec3d operator*(double s, const Vec3d& v) { return v * s; }

// ---------------------------------------------------------------
// Gram-Schmidt — orthogonalisation d'une base
// ---------------------------------------------------------------
struct OrthoBasis {
    Vec3d u, v, w;  // garantis orthogonaux et normalisés
};

inline OrthoBasis GramSchmidt(Vec3d a, Vec3d b, Vec3d c) {
    // Étape 1 : normaliser a → u
    Vec3d u = a.Normalized();
    assert(!nearlyZero(u.Norm()) && "Premier vecteur est nul");

    // Étape 2 : ôter la composante u de b → v
    Vec3d v = (b - Project(b, u)).Normalized();
    assert(!nearlyZero(v.Norm()) && "Vecteurs colineaires");

    // Étape 3 : ôter les composantes u et v de c → w
    Vec3d w = (c - Project(c, u) - Project(c, v)).Normalized();
    assert(!nearlyZero(w.Norm()) && "Vecteurs coplanaires");

    return {u, v, w};
}

// Garantie layout mémoire
static_assert(sizeof(Vec3d) == 24,
    "Vec3d doit faire 24 bytes (3 doubles)");
static_assert(offsetof(Vec3d, x) == 0,  "x doit etre en premier");
static_assert(offsetof(Vec3d, y) == 8,  "y doit etre a l offset 8");
static_assert(offsetof(Vec3d, z) == 16, "z doit etre a l offset 16");

} // namespace NkMath