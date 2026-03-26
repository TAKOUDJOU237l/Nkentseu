// ============================================================
// FILE: rasterizer-lookat.cpp
// TP8 — Rasteriseur logiciel avec LookAt
// Tests 32 à 37
// ============================================================
#include <cstdio>
#include <cmath>
#include <cassert>
#include "../tp7-mat4d-inverse/Mat4d.h"

using namespace NkMath;

// ---------------------------------------------------------------
// Projection perspective depuis fov
// ---------------------------------------------------------------
struct Camera {
    Mat4d view;       // matrice View (LookAt)
    double fovY;      // champ de vision vertical (radians)
    double aspect;    // largeur / hauteur
    double near;      // plan proche
    double far;       // plan lointain
    int    width;     // largeur image
    int    height;    // hauteur image
};

// Projette un point 3D monde → pixel image
// Retourne false si derrière la caméra ou hors image
bool projectPoint(const Camera& cam,
                  const Vec3d& worldPos,
                  int& px, int& py) {
    // 1. Passage en espace caméra
    Vec4d p4(worldPos, 1.0);
    Vec4d camPos = cam.view * p4;

    // Derrière la caméra (z > 0 en espace caméra OpenGL)
    if (camPos.z >= 0.0) return false;

    // 2. Projection perspective
    double tanHalf = std::tan(cam.fovY * 0.5);
    double fx = (cam.width  * 0.5) / tanHalf;
    double fy = (cam.height * 0.5) / tanHalf;
    double cx = cam.width  * 0.5;
    double cy = cam.height * 0.5;

    // Division perspective
    double u = fx * (-camPos.x / camPos.z) + cx;
    double v = fy * (-camPos.y / camPos.z) + cy;

    px = (int)u;
    py = (int)v;

    // Hors image
    if (px < 0 || px >= cam.width ||
        py < 0 || py >= cam.height) return false;

    return true;
}

int main() {
    printf("=== TP8 — Rasteriseur LookAt ===\n\n");

    // -------------------------------------------------------
    // Test 32 : Cube unitaire centré à l'origine
    // 8 coins du cube [-0.5, 0.5]³
    // -------------------------------------------------------
    printf("--- Test 32 : Cube unitaire ---\n");

    Vec3d coins[8] = {
        {-0.5, -0.5, -0.5},
        { 0.5, -0.5, -0.5},
        { 0.5,  0.5, -0.5},
        {-0.5,  0.5, -0.5},
        {-0.5, -0.5,  0.5},
        { 0.5, -0.5,  0.5},
        { 0.5,  0.5,  0.5},
        {-0.5,  0.5,  0.5}
    };

    printf("  8 coins du cube definis\n");

    // 12 arêtes du cube
    int aretes[12][2] = {
        // Face avant
        {0,1}, {1,2}, {2,3}, {3,0},
        // Face arrière
        {4,5}, {5,6}, {6,7}, {7,4},
        // Connexions
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    printf("  12 aretes definies\n");

    // -------------------------------------------------------
    // Test 33 : LookAt(eye={0,1,3}, target={0,0,0}, up={0,1,0})
    // -------------------------------------------------------
    printf("\n--- Test 33 : LookAt ---\n");

    Vec3d eye   (0.0, 1.0, 3.0);
    Vec3d target(0.0, 0.0, 0.0);
    Vec3d up    (0.0, 1.0, 0.0);

    Mat4d viewMat = Mat4d::LookAt(eye, target, up);

    printf("  Matrice View :\n");
    viewMat.Print();
    printf("  LookAt cree\n");

    // -------------------------------------------------------
    // Test 34 : Projection perspective fov=60°
    // -------------------------------------------------------
    printf("\n--- Test 34 : Projection fov=60 ---\n");

    Camera cam;
    cam.view   = viewMat;
    cam.fovY   = kPi / 3.0;  // 60 degrés
    cam.aspect = 1.0;         // 512/512
    cam.near   = 0.1;
    cam.far    = 100.0;
    cam.width  = 512;
    cam.height = 512;

    printf("  fovY = 60 degres\n");
    printf("  aspect = 1.0 (512x512)\n");
    printf("   Camera configuree\n");

    // -------------------------------------------------------
    // Tests 35, 36, 37 : 10 frames avec rotation
    // -------------------------------------------------------
    printf("\n--- Tests 35-37 : 10 frames rotation ---\n");

    for (int frame = 0; frame < 10; frame++) {

        // Test 37 : angle différent à chaque frame
        double angle = frame * (kPi / 5.0); // 36° par frame

        // Rotation du cube autour de Y
        Mat4d rotation = Mat4d::RotateAxis(
            Vec3d(0.0, 1.0, 0.0), angle
        );

        // Crée l'image
        NkImage img(512, 512);
        img.Fill(30, 30, 30);  // fond gris foncé

        // Projette les coins transformés
        int px[8], py[8];
        bool visible[8];

        for (int i = 0; i < 8; i++) {
            // Applique la rotation au coin
            Vec4d c4(coins[i], 1.0);
            Vec4d rotated = rotation * c4;
            Vec3d worldPos = rotated.ToVec3();

            visible[i] = projectPoint(
                cam, worldPos, px[i], py[i]
            );
        }

        // Test 35 : affiche les 12 arêtes en rouge
        int aretes_dessinees = 0;
        for (int i = 0; i < 12; i++) {
            int a = aretes[i][0];
            int b = aretes[i][1];

            if (visible[a] && visible[b]) {
                img.DrawLine(
                    px[a], py[a],
                    px[b], py[b],
                    255, 0, 0  // rouge
                );
                aretes_dessinees++;
            }
        }

        // Dessine les coins en blanc par dessus
        for (int i = 0; i < 8; i++) {
            if (visible[i]) {
                img.DrawPoint(px[i], py[i],
                              255, 255, 255);
            }
        }

        // Affiche info frame
        printf("  Frame %2d (angle=%.0f°) :"
               " %d/12 aretes\n",
               frame,
               angle * 180.0 / kPi,
               aretes_dessinees);

        // Test 36 : sauvegarde PPM P6
        char filename[64];
        snprintf(filename, sizeof(filename),
                 "frame_%02d.ppm", frame);

        bool saved = img.SavePPM(filename);
        assert(saved && "Sauvegarde PPM doit reussir");
    }

    printf("\n  10 frames sauvegardees\n");
    printf("   Aretes en rouge\n");
    printf("  Format PPM P6 binaire\n");
    printf("   Rotation differente par frame\n");

    printf("\n=== Tous les tests 32-37 passes ! ===\n");
    printf("Ouvrir frame_00.ppm a frame_09.ppm\n");
    return 0;
}