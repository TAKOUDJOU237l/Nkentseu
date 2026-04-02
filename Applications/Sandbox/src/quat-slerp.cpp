#include <cstdio>
#include <cmath>
#include <cassert>
#include "Quat.h"
#include "NkImage.h"

using namespace NkMath;

// Generateur pseudo-aleatoire
static double pseudoRand(int& seed) {
    seed=seed*1664525+1013904223;
    return (double)(seed&0x7FFFFFFF)/(double)0x7FFFFFFF*2.0-1.0;
}

// Projecte un point 3D vers pixels
static bool projectPoint(const Mat4d& view, double fovY,
                          int W, int H, const Vec3d& wp,
                          int& px, int& py) {
    Vec4d p4(wp,1.0);
    Vec4d cp=view*p4;
    if (cp.z>=0.0) return false;
    double tanH=std::tan(fovY*0.5);
    double fx=(W*0.5)/tanH;
    double fy=(H*0.5)/tanH;
    double u=fx*(-cp.x/cp.z)+W*0.5;
    double v=fy*(-cp.y/cp.z)+H*0.5;
    px=(int)u; py=(int)v;
    if (px<0||px>=W||py<0||py>=H) return false;
    return true;
}

// Dessine le cube avec une rotation quaternion donnee
static void drawCube(NkImage& img, const Mat4d& view,
                     const Mat4d& rotation,
                     uint8_t r, uint8_t g, uint8_t b) {
    Vec3d cube[8]={
        {-0.5,-0.5,-0.5},{0.5,-0.5,-0.5},
        {0.5,0.5,-0.5},{-0.5,0.5,-0.5},
        {-0.5,-0.5,0.5},{0.5,-0.5,0.5},
        {0.5,0.5,0.5},{-0.5,0.5,0.5}
    };
    int aretes[12][2]={
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };
    int px[8],py[8]; bool vis[8];
    for (int i=0;i<8;i++) {
        Vec4d c4(cube[i],1.0);
        Vec4d rot=rotation*c4;
        vis[i]=projectPoint(view,kPi/3.0,512,512,
                            rot.ToVec3(),px[i],py[i]);
    }
    for (int i=0;i<12;i++) {
        int a=aretes[i][0],b=aretes[i][1];
        if (vis[a]&&vis[b])
            img.DrawLine(px[a],py[a],px[b],py[b],r,g,b);
    }
    for (int i=0;i<8;i++)
        if (vis[i])
            img.DrawPoint(px[i],py[i],255,255,255);
}

