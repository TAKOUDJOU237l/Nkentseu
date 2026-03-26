// ============================================================
// FILE: NkImage.h
// TP6 — Image PPM 512x512 pour affichage du cube
// ============================================================
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cassert>

namespace NkMath {

class NkImage {
public:
    NkImage() : m_width(0), m_height(0), m_data(nullptr) {}

    NkImage(int w, int h) : m_width(w), m_height(h) {
        m_data = new uint8_t[w * h * 4]();  // init à 0
    }

    ~NkImage() { delete[] m_data; }

    // Accès pixel
    uint8_t* At(int x, int y) {
        assert(x >= 0 && x < m_width);
        assert(y >= 0 && y < m_height);
        return m_data + (y * m_width + x) * 4;
    }

    const uint8_t* At(int x, int y) const {
        assert(x >= 0 && x < m_width);
        assert(y >= 0 && y < m_height);
        return m_data + (y * m_width + x) * 4;
    }

    void SetPixelRGBA(int x, int y,
                      uint8_t r, uint8_t g,
                      uint8_t b, uint8_t a = 255) {
        if (x < 0 || x >= m_width ||
            y < 0 || y >= m_height) return;
        uint8_t* p = At(x, y);
        p[0]=r; p[1]=g; p[2]=b; p[3]=a;
    }

    void Fill(uint8_t r, uint8_t g,
              uint8_t b, uint8_t a = 255) {
        for (int y = 0; y < m_height; y++)
            for (int x = 0; x < m_width; x++)
                SetPixelRGBA(x, y, r, g, b, a);
    }

    // Tracé de ligne — algorithme de Bresenham
    void DrawLine(int x0, int y0, int x1, int y1,
                  uint8_t r, uint8_t g, uint8_t b) {
        int dx =  (x1-x0) < 0 ? -(x1-x0) : (x1-x0);
        int dy = -(y1-y0) < 0 ?  (y1-y0) : -(y1-y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;

        while (true) {
            SetPixelRGBA(x0, y0, r, g, b);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    // Dessine un carré 3x3 autour d'un point
    void DrawPoint(int x, int y,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++)
                SetPixelRGBA(x+dx, y+dy, r, g, b);
    }

    // Sauvegarde PPM P6
    bool SavePPM(const char* path) const {
        FILE* f = fopen(path, "wb");
        if (!f) return false;
        fprintf(f, "P6\n%d %d\n255\n", m_width, m_height);
        for (int y = 0; y < m_height; y++)
            for (int x = 0; x < m_width; x++) {
                const uint8_t* p = At(x, y);
                fwrite(p, 1, 3, f);  // RGB seulement
            }
        fclose(f);
        return true;
    }

    int Width()  const { return m_width;  }
    int Height() const { return m_height; }

private:
    int      m_width;
    int      m_height;
    uint8_t* m_data;

    // Pas de copie
    NkImage(const NkImage&) = delete;
    NkImage& operator=(const NkImage&) = delete;
};

} // namespace NkMath