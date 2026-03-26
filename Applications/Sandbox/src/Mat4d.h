// ============================================================
// FILE: Mat4d.h
// TP7 — Tests 28 à 31 Mat4d + Inverse
// ============================================================
#include <cstdio>
#include <cmath>
#include <cassert>
#include "Mat4d.h"

using namespace NkMath;

// Générateur pseudo-aléatoire
double pseudoRand(int& seed) {
    seed = seed * 1664525 + 1013904223;
    return (double)(seed & 0x7FFFFFFF) /
           (double)0x7FFFFFFF * 4.0 - 2.0;
}

// Génère une matrice aléatoire non-singulière
Mat4d randomMat(int& seed) {
    Mat4d m = Mat4d::Zero();
    // Diagonale dominante pour garantir non-singulière
    for (int r = 0; r < 4; r++) {
        double rowSum = 0.0;
        for (int c = 0; c < 4; c++) {
            if (r != c) {
                m(r,c) = pseudoRand(seed) * 0.3;
                rowSum += m(r,c) < 0 ?
                    -m(r,c) : m(r,c);
            }
        }
        // Diagonale > somme des autres → non-singulière
        m(r,r) = rowSum + 1.0 + pseudoRand(seed)*0.1;
    }
    return m;
}

int main() {
    printf("=== TP7 — Tests Mat4d + Inverse ===\n\n");

    // -------------------------------------------------------
    // Test 28 : M × Identity() == M
    // pour 10 matrices aléatoires
    // -------------------------------------------------------
    printf("--- Test 28 : M x Identity = M ---\n");

    int seed = 12345;
    Mat4d I  = Mat4d::Identity();

    for (int i = 0; i < 10; i++) {
        Mat4d M  = randomMat(seed);
        Mat4d MI = M * I;
        Mat4d IM = I * M;

        // M × I == M
        assert(MI == M && "M x I doit etre M");
        // I × M == M
        assert(IM == M && "I x M doit etre M");

        printf("   Matrice %2d : M x I = M ✓  I x M = M ✓\n",
               i+1);
    }

    // -------------------------------------------------------
    // Test 29 : M × M⁻¹ == Identity() à 1e-10 près
    // pour 10 matrices non-singulières
    // -------------------------------------------------------
    printf("\n--- Test 29 : M x M⁻¹ = I a 1e-10 ---\n");

    seed = 99999;
    for (int i = 0; i < 10; i++) {
        Mat4d M   = randomMat(seed);
        Mat4d Inv = Mat4d::Zero();

        bool ok = M.Inverse(Inv);
        assert(ok && "Inverse doit reussir pour matrice non-singuliere");

        Mat4d MInv  = M * Inv;
        Mat4d InvM  = Inv * M;

        // Vérifie M × M⁻¹ == I
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) {
                double expected = (r == c) ? 1.0 : 0.0;
                double diff = MInv(r,c) - expected;
                diff = diff < 0 ? -diff : diff;
                assert(diff < 1e-10 &&
                       "M x Inv doit etre identite a 1e-10");
            }

        // Vérifie M⁻¹ × M == I
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) {
                double expected = (r == c) ? 1.0 : 0.0;
                double diff = InvM(r,c) - expected;
                diff = diff < 0 ? -diff : diff;
                assert(diff < 1e-10 &&
                       "Inv x M doit etre identite a 1e-10");
            }

        printf("   Matrice %2d : M x M⁻¹ = I ✓"
               "  M⁻¹ x M = I ✓\n", i+1);
    }

    // -------------------------------------------------------
    // Test 30 : Inverse d'une matrice singulière
    // doit retourner false
    // -------------------------------------------------------
    printf("\n--- Test 30 : Matrice singuliere ---\n");

    // Matrice avec une ligne de zéros → singulière
    Mat4d singular = Mat4d::Zero();
    singular(0,0) = 1.0;
    singular(1,1) = 1.0;
    // Ligne 2 et 3 = zéros → det = 0

    Mat4d invSingular;
    bool okSingular = singular.Inverse(invSingular);
    assert(!okSingular &&
           "Inverse doit retourner false pour matrice singuliere");
    printf("   Matrice singuliere → Inverse() = false\n");

    // Matrice avec deux lignes identiques → singulière
    Mat4d singular2 = Mat4d::Identity();
    singular2(2,0) = singular2(3,0) = 1.0;
    singular2(2,1) = singular2(3,1) = 1.0;
    singular2(2,2) = singular2(3,2) = 1.0;
    singular2(2,3) = singular2(3,3) = 1.0;

    Mat4d invSingular2;
    bool okSingular2 = singular2.Inverse(invSingular2);
    assert(!okSingular2 &&
           "Inverse doit retourner false");
    printf("   Matrice lignes identiques"
           " → Inverse() = false\n");

    // -------------------------------------------------------
    // Test 31 : RotateAxis({0,1,0}, PI/2) × {1,0,0,1}
    //           == {0,0,-1,1}
    // -------------------------------------------------------
    printf("\n--- Test 31 : RotateAxis Y PI/2 ---\n");

    Mat4d R = Mat4d::RotateAxis(
        Vec3d(0.0, 1.0, 0.0),
        kPi / 2.0
    );

    Vec4d v(1.0, 0.0, 0.0, 1.0);
    Vec4d result = R * v;

    printf("  RotateAxis(Y, PI/2) x {1,0,0,1} =\n");
    printf("  {%.6f, %.6f, %.6f, %.6f}\n",
           result.x, result.y,
           result.z, result.w);

    assert(approxEq(result.x,  0.0, 1e-10) &&
           "x doit etre 0");
    assert(approxEq(result.y,  0.0, 1e-10) &&
           "y doit etre 0");
    assert(approxEq(result.z, -1.0, 1e-10) &&
           "z doit etre -1");
    assert(approxEq(result.w,  1.0, 1e-10) &&
           "w doit etre 1");
    printf("   RotateAxis(Y,PI/2) x {1,0,0,1}"
           " = {0,0,-1,1}\n");

    // Vérification supplémentaire
    // RotateAxis(Y, PI/2) × {0,0,1,1} = {1,0,0,1}
    Vec4d v2(0.0, 0.0, 1.0, 1.0);
    Vec4d r2 = R * v2;
    assert(approxEq(r2.x,  1.0, 1e-10) &&
           approxEq(r2.z,  0.0, 1e-10));
    printf("   RotateAxis(Y,PI/2) x {0,0,1,1}"
           " = {1,0,0,1}\n");

    printf("\n=== Tous les tests 28-31 passes ! ===\n");
    return 0;
}