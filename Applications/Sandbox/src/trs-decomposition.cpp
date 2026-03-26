// ============================================================
// FILE: trs-decomposition.cpp
// TP9 — TRS + Décomposition inverse
// Tests 38 à 40
// ============================================================
#include <cstdio>
#include <cmath>
#include <cassert>
#include "../tp7-mat4d-inverse/Mat4d.h"

using namespace NkMath;

// ---------------------------------------------------------------
// Générateur pseudo-aléatoire
// ---------------------------------------------------------------
double pseudoRand(int& seed) {
    seed = seed * 1664525 + 1013904223;
    return (double)(seed & 0x7FFFFFFF) /
           (double)0x7FFFFFFF * 2.0 - 1.0;
}

// Rand dans [min, max]
double randRange(int& seed, double min, double max) {
    return min + (pseudoRand(seed) * 0.5 + 0.5) * (max - min);
}

// ---------------------------------------------------------------
// Décomposition TRS depuis une matrice M
//
// M = T * R * S
//
// Étape 1 : T = colonne 3 de M (translation)
// Étape 2 : S = norme de chaque colonne 0,1,2 (scale)
// Étape 3 : R = M sans scale (divise chaque colonne par S)
// ---------------------------------------------------------------
struct TRSResult {
    Vec3d  translation;
    Mat4d  rotation;     // matrice de rotation pure
    Vec3d  scale;
    double angleRad;     // angle extrait de R
    Vec3d  axis;         // axe extrait de R
};

TRSResult decomposeTRS(const Mat4d& M) {
    TRSResult res;

    // ---- Étape 1 : Translation ----
    // La 4ème colonne contient la translation
    res.translation = Vec3d(
        M(0,3), M(1,3), M(2,3)
    );

    // ---- Étape 2 : Scale ----
    // Norme de chaque colonne (0,1,2)
    double sx = std::sqrt(
        M(0,0)*M(0,0) + M(1,0)*M(1,0) + M(2,0)*M(2,0)
    );
    double sy = std::sqrt(
        M(0,1)*M(0,1) + M(1,1)*M(1,1) + M(2,1)*M(2,1)
    );
    double sz = std::sqrt(
        M(0,2)*M(0,2) + M(1,2)*M(1,2) + M(2,2)*M(2,2)
    );
    res.scale = Vec3d(sx, sy, sz);

    // ---- Étape 3 : Rotation ----
    // Divise chaque colonne par son scale
    res.rotation = Mat4d::Zero();
    if (!nearlyZero(sx)) {
        res.rotation(0,0) = M(0,0)/sx;
        res.rotation(1,0) = M(1,0)/sx;
        res.rotation(2,0) = M(2,0)/sx;
    }
    if (!nearlyZero(sy)) {
        res.rotation(0,1) = M(0,1)/sy;
        res.rotation(1,1) = M(1,1)/sy;
        res.rotation(2,1) = M(2,1)/sy;
    }
    if (!nearlyZero(sz)) {
        res.rotation(0,2) = M(0,2)/sz;
        res.rotation(1,2) = M(1,2)/sz;
        res.rotation(2,2) = M(2,2)/sz;
    }
    res.rotation(3,3) = 1.0;

    // ---- Étape 4 : Angle depuis R ----
    // trace(R) = 1 + 2*cos(angle)
    double trace = res.rotation(0,0) +
                   res.rotation(1,1) +
                   res.rotation(2,2);
    double cosA = (trace - 1.0) * 0.5;
    // Clamp pour eviter NaN dans acos
    cosA = cosA < -1.0 ? -1.0 : (cosA > 1.0 ? 1.0 : cosA);
    res.angleRad = std::acos(cosA);

    // ---- Étape 5 : Axe depuis R ----
    // axe = [R32-R23, R13-R31, R21-R12] / (2*sin(angle))
    double sinA = std::sin(res.angleRad);
    if (!nearlyZero(sinA)) {
        double inv2s = 1.0 / (2.0 * sinA);
        res.axis = Vec3d(
            (res.rotation(2,1) - res.rotation(1,2)) * inv2s,
            (res.rotation(0,2) - res.rotation(2,0)) * inv2s,
            (res.rotation(1,0) - res.rotation(0,1)) * inv2s
        ).Normalized();
    } else {
        // Angle ≈ 0 → axe arbitraire
        res.axis = Vec3d(0.0, 1.0, 0.0);
    }

    return res;
}

// ---------------------------------------------------------------
// Vérifie que deux matrices sont égales à eps près
// ---------------------------------------------------------------
bool matApproxEq(const Mat4d& a,
                 const Mat4d& b,
                 double eps = 1e-9) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            if (!approxEq(a(r,c), b(r,c), eps))
                return false;
    return true;
}

int main() {
    printf("=== TP9 — TRS + Decomposition ===\n\n");

    int seed = 777;
    int passed = 0;

    for (int i = 0; i < 20; i++) {

        // -----------------------------------------------
        // Test 38 : Génère T, R, S aléatoires
        // Construit M = TRS(T, R, S)
        // -----------------------------------------------

        // Translation aléatoire [-5, 5]
        Vec3d T(
            randRange(seed, -5.0, 5.0),
            randRange(seed, -5.0, 5.0),
            randRange(seed, -5.0, 5.0)
        );

        // Axe de rotation aléatoire normalisé
        Vec3d axis(
            pseudoRand(seed),
            pseudoRand(seed),
            pseudoRand(seed)
        );
        axis = axis.Normalized();
        // Si vecteur nul, utilise Y
        if (nearlyZero(axis.Norm()))
            axis = Vec3d(0.0, 1.0, 0.0);

        // Angle aléatoire [-PI, PI]
        double angle = randRange(seed, -kPi, kPi);

        // Scale aléatoire [0.5, 3.0] — positif !
        Vec3d S(
            randRange(seed, 0.5, 3.0),
            randRange(seed, 0.5, 3.0),
            randRange(seed, 0.5, 3.0)
        );

        // Construit M = TRS
        Mat4d M = Mat4d::TRS(T, axis, angle, S);

        // -----------------------------------------------
        // Test 39 : Décompose M → T', R', S'
        // -----------------------------------------------
        TRSResult res = decomposeTRS(M);

        // -----------------------------------------------
        // Test 40 : Vérifie T' == T, S' == S
        // et que R' reconstruit == R original
        // -----------------------------------------------

        // Vérifie Translation
        bool T_ok =
            approxEq(res.translation.x, T.x, 1e-9) &&
            approxEq(res.translation.y, T.y, 1e-9) &&
            approxEq(res.translation.z, T.z, 1e-9);
        assert(T_ok && "Translation doit correspondre");

        // Vérifie Scale
        bool S_ok =
            approxEq(res.scale.x, S.x, 1e-9) &&
            approxEq(res.scale.y, S.y, 1e-9) &&
            approxEq(res.scale.z, S.z, 1e-9);
        assert(S_ok && "Scale doit correspondre");

        // Vérifie Rotation — reconstruit M depuis T', R', S'
        // et compare avec M original
        Mat4d M_recon = Mat4d::TRS(
            res.translation,
            res.axis,
            res.angleRad,
            res.scale
        );

        // Vérifie partie rotation uniquement (3x3)
        bool R_ok = true;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                if (!approxEq(M(r,c), M_recon(r,c), 1e-9))
                    R_ok = false;

        assert(R_ok && "Rotation doit correspondre");

        passed++;
        printf("   Triplet %2d :"
               " T(%.2f,%.2f,%.2f)"
               " angle=%.1f°"
               " S(%.2f,%.2f,%.2f)\n",
               i+1,
               T.x, T.y, T.z,
               angle * 180.0 / kPi,
               S.x, S.y, S.z);
    }

    printf("\n=== Résumé ===\n");
    printf("Triplets valides : %d / 20\n", passed);
    printf("T retrouve       : \n");
    printf("S retrouve       : \n");
    printf("R retrouve       : \n");
    printf("\n=== Tous les tests 38-40 passes ! ===\n");
    return 0;
}