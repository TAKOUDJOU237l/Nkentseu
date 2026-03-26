// ============================================================================
// Sandbox/src/main.cpp
// Pattern A : Dispatcher typé (push - événementiel)
// ============================================================================

#include "NKWindow/Core/NkWindow.h"
#include "NKWindow/Core/NkSystem.h"
#include "NKWindow/Events/NkEventDispatcher.h"
#include "NKWindow/Events/NkEventSystem.h"
#include "NKWindow/Events/NkGamepadSystem.h"
#include "NKWindow/Core/NkMain.h"
#include "NKRenderer/Deprecate/NkRenderer.h"
#include "NKRenderer/Deprecate/NkRendererConfig.h"
#include "NKTime/NkChrono.h"

#include "NKLogger/NkLog.h"
#include "NKMath/NKMath.h"

#include "NKMemory/NkMemory.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>
#include <cassert>

// ================================================================
// DECLARATIONS
// ================================================================

namespace NkMath {

constexpr double kEps  = 1e-9;
constexpr float  kFEps = 1e-6f;
constexpr float  kPiF  = 3.14159265358979323846f;
constexpr double kPi   = 3.14159265358979323846;

// ---------------------------------------------------------------
// Float utils
// ---------------------------------------------------------------
static uint32_t floatBits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    return bits;
}

static void printBinary(uint32_t bits, int nbBits) {
    for (int i = nbBits - 1; i >= 0; i--)
        printf("%u", (bits >> i) & 1);
}

void inspectFloat(float f) {
    uint32_t bits     = floatBits(f);
    uint32_t sign     = (bits >> 31) & 0x1;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa =  bits        & 0x7FFFFF;
    printf("float = %.10f | hex = 0x%08X\n", f, bits);
    printf("  signe    = %u\n", sign);
    printf("  exposant = %u (2^%d)\n",
           exponent, (int)exponent - 127);
    printf("  mantisse = 0x%06X | bin = ", mantissa);
    printBinary(sign, 1);
    printf(" ");
    printBinary(exponent, 8);
    printf(" ");
    printBinary(mantissa, 23);
    printf("\n");
    if      (exponent == 0xFF && mantissa != 0)
        printf("  type = NaN\n");
    else if (exponent == 0xFF && mantissa == 0)
        printf("  type = %cInf\n", sign ? '-' : '+');
    else if (exponent == 0x00 && mantissa != 0)
        printf("  type = Subnormal\n");
    else if (exponent == 0x00 && mantissa == 0)
        printf("  type = %c0\n", sign ? '-' : '+');
    else
        printf("  type = Normal\n");
    printf("\n");
}

float kahanSum(const float* data, int n) {
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

float varianceNaive(const float* data, int n) {
    float sum  = 0.0f;
    float sum2 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum  += data[i];
        sum2 += data[i] * data[i];
    }
    return (sum2 - sum * sum / n) / (n - 1);
}

float varianceWelford(const float* data, int n) {
    float mean = 0.0f;
    float M2   = 0.0f;
    for (int i = 0; i < n; i++) {
        float delta  = data[i] - mean;
        mean        += delta / (i + 1);
        float delta2 = data[i] - mean;
        M2          += delta * delta2;
    }
    return M2 / (n - 1);
}

float measureEpsilon() {
    float eps = 1.0f;
    while (1.0f + eps / 2.0f != 1.0f)
        eps /= 2.0f;
    return eps;
}

inline bool isFiniteValid(float  x) { return std::isfinite(x); }
inline bool isFiniteValid(double x) { return std::isfinite(x); }

inline bool nearlyZero(float  x, float  eps = 1e-6f) {
    return std::abs(x) < eps;
}
inline bool nearlyZero(double x, double eps = 1e-9) {
    return std::abs(x) < eps;
}

inline bool approxEq(float a, float b, float eps = 1e-6f) {
    if (a == b) return true;
    float maxAB = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= eps * std::max(1.0f, maxAB);
}
inline bool approxEq(double a, double b, double eps = 1e-9) {
    if (a == b) return true;
    double maxAB = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= eps * std::max(1.0, maxAB);
}

// ---------------------------------------------------------------
// Vec2d
// ---------------------------------------------------------------
struct Vec2d {
    double x, y;
    Vec2d() : x(0.0), y(0.0) {}
    Vec2d(double x, double y) : x(x), y(y) {}
    explicit Vec2d(double s) : x(s), y(s) {}
    double& operator[](int i) {
        assert(i >= 0 && i < 2);
        return (&x)[i];
    }
    const double& operator[](int i) const {
        assert(i >= 0 && i < 2);
        return (&x)[i];
    }
    Vec2d operator+(const Vec2d& o) const { return {x+o.x, y+o.y}; }
    Vec2d operator-(const Vec2d& o) const { return {x-o.x, y-o.y}; }
    Vec2d operator*(double s)       const { return {x*s,   y*s  }; }
    Vec2d operator/(double s)       const { return {x/s,   y/s  }; }
    Vec2d operator-()               const { return {-x,    -y   }; }
    Vec2d& operator+=(const Vec2d& o) { x+=o.x; y+=o.y; return *this; }
    Vec2d& operator-=(const Vec2d& o) { x-=o.x; y-=o.y; return *this; }
    Vec2d& operator*=(double s)       { x*=s;   y*=s;   return *this; }
    Vec2d& operator/=(double s)       { x/=s;   y/=s;   return *this; }
    bool operator==(const Vec2d& o) const {
        return approxEq(x,o.x) && approxEq(y,o.y);
    }
    bool operator!=(const Vec2d& o) const { return !(*this == o); }
    double Norm2() const { return x*x + y*y; }
    double Norm()  const { return std::sqrt(Norm2()); }
    Vec2d Normalized() const {
        double n = Norm();
        if (nearlyZero(n)) return Vec2d(0.0);
        return {x/n, y/n};
    }
    bool IsNormalized(double eps = kEps) const {
        return approxEq(Norm2(), 1.0, eps);
    }
    void Print() const {
        printf("Vec2d(%.6f, %.6f)\n", x, y);
    }
};
inline double Dot(const Vec2d& a, const Vec2d& b) {
    return a.x*b.x + a.y*b.y;
}
inline double Cross2D(const Vec2d& a, const Vec2d& b) {
    return a.x*b.y - a.y*b.x;
}
inline Vec2d Lerp(const Vec2d& a, const Vec2d& b, double t) {
    return {a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t};
}
inline Vec2d operator*(double s, const Vec2d& v) { return v*s; }
static_assert(sizeof(Vec2d) == 16, "Vec2d doit faire 16 bytes");
static_assert(offsetof(Vec2d, x) == 0, "x en premier");
static_assert(offsetof(Vec2d, y) == 8, "y a offset 8");

// ---------------------------------------------------------------
// Vec3d
// ---------------------------------------------------------------
struct Vec3d {
    double x, y, z;
    Vec3d() : x(0.0), y(0.0), z(0.0) {}
    Vec3d(double x, double y, double z) : x(x), y(y), z(z) {}
    explicit Vec3d(double s) : x(s), y(s), z(s) {}
    double& operator[](int i) {
        assert(i >= 0 && i < 3);
        return (&x)[i];
    }
    const double& operator[](int i) const {
        assert(i >= 0 && i < 3);
        return (&x)[i];
    }
    Vec3d operator+(const Vec3d& o) const { return {x+o.x,y+o.y,z+o.z}; }
    Vec3d operator-(const Vec3d& o) const { return {x-o.x,y-o.y,z-o.z}; }
    Vec3d operator*(double s)       const { return {x*s,  y*s,  z*s  }; }
    Vec3d operator/(double s)       const { return {x/s,  y/s,  z/s  }; }
    Vec3d operator-()               const { return {-x,   -y,   -z   }; }
    Vec3d& operator+=(const Vec3d& o) { x+=o.x;y+=o.y;z+=o.z; return *this; }
    Vec3d& operator-=(const Vec3d& o) { x-=o.x;y-=o.y;z-=o.z; return *this; }
    Vec3d& operator*=(double s)       { x*=s;  y*=s;  z*=s;   return *this; }
    Vec3d& operator/=(double s)       { x/=s;  y/=s;  z/=s;   return *this; }
    bool operator==(const Vec3d& o) const {
        return approxEq(x,o.x)&&approxEq(y,o.y)&&approxEq(z,o.z);
    }
    bool operator!=(const Vec3d& o) const { return !(*this == o); }
    double Norm2() const { return x*x+y*y+z*z; }
    double Norm()  const { return std::sqrt(Norm2()); }
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
inline double Dot(const Vec3d& a, const Vec3d& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline Vec3d Cross(const Vec3d& a, const Vec3d& b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}
inline Vec3d Lerp(const Vec3d& a, const Vec3d& b, double t) {
    return a + (b-a)*t;
}
inline Vec3d Project(const Vec3d& a, const Vec3d& b) {
    double b2 = b.Norm2();
    assert(!nearlyZero(b2));
    return b * (Dot(a,b) / b2);
}
inline Vec3d Reject(const Vec3d& a, const Vec3d& b) {
    return a - Project(a,b);
}
inline Vec3d operator*(double s, const Vec3d& v) { return v*s; }
static_assert(sizeof(Vec3d) == 24, "Vec3d doit faire 24 bytes");
static_assert(offsetof(Vec3d, x) == 0,  "x en premier");
static_assert(offsetof(Vec3d, y) == 8,  "y a offset 8");
static_assert(offsetof(Vec3d, z) == 16, "z a offset 16");

struct OrthoBasis { Vec3d u, v, w; };
inline OrthoBasis GramSchmidt(Vec3d a, Vec3d b, Vec3d c) {
    Vec3d u = a.Normalized();
    Vec3d v = (b - Project(b,u)).Normalized();
    Vec3d w = (c - Project(c,u) - Project(c,v)).Normalized();
    return {u, v, w};
}

// ---------------------------------------------------------------
// Vec4d
// ---------------------------------------------------------------
struct Vec4d {
    double x, y, z, w;
    Vec4d() : x(0.0), y(0.0), z(0.0), w(0.0) {}
    Vec4d(double x, double y, double z, double w)
        : x(x), y(y), z(z), w(w) {}
    Vec4d(const Vec3d& v, double w)
        : x(v.x), y(v.y), z(v.z), w(w) {}
    double& operator[](int i) {
        assert(i >= 0 && i < 4);
        return (&x)[i];
    }
    const double& operator[](int i) const {
        assert(i >= 0 && i < 4);
        return (&x)[i];
    }
    Vec4d operator+(const Vec4d& o) const {
        return {x+o.x,y+o.y,z+o.z,w+o.w};
    }
    Vec4d operator-(const Vec4d& o) const {
        return {x-o.x,y-o.y,z-o.z,w-o.w};
    }
    Vec4d operator*(double s) const { return {x*s,y*s,z*s,w*s}; }
    Vec4d operator/(double s) const { return {x/s,y/s,z/s,w/s}; }
    Vec4d operator-()         const { return {-x,-y,-z,-w}; }
    Vec4d& operator+=(const Vec4d& o) {
        x+=o.x;y+=o.y;z+=o.z;w+=o.w; return *this;
    }
    Vec4d& operator-=(const Vec4d& o) {
        x-=o.x;y-=o.y;z-=o.z;w-=o.w; return *this;
    }
    Vec4d& operator*=(double s) { x*=s;y*=s;z*=s;w*=s; return *this; }
    Vec4d& operator/=(double s) { x/=s;y/=s;z/=s;w/=s; return *this; }
    bool operator==(const Vec4d& o) const {
        return approxEq(x,o.x)&&approxEq(y,o.y)&&
               approxEq(z,o.z)&&approxEq(w,o.w);
    }
    bool operator!=(const Vec4d& o) const { return !(*this == o); }
    double Dot(const Vec4d& o) const {
        return x*o.x+y*o.y+z*o.z+w*o.w;
    }
    double Norm2() const { return x*x+y*y+z*z+w*w; }
    double Norm()  const { return std::sqrt(Norm2()); }
    Vec3d ToVec3() const {
        assert(!nearlyZero(w));
        return {x/w, y/w, z/w};
    }
    Vec3d XYZ() const { return {x, y, z}; }
    void Print() const {
        printf("Vec4d(%.6f,%.6f,%.6f,%.6f)\n",x,y,z,w);
    }
};
inline Vec4d operator*(double s, const Vec4d& v) { return v*s; }
static_assert(sizeof(Vec4d) == 32, "Vec4d doit faire 32 bytes");
static_assert(offsetof(Vec4d, x) == 0,  "x en premier");
static_assert(offsetof(Vec4d, y) == 8,  "y a offset 8");
static_assert(offsetof(Vec4d, z) == 16, "z a offset 16");
static_assert(offsetof(Vec4d, w) == 24, "w a offset 24");

// ---------------------------------------------------------------
// NkImage
// ---------------------------------------------------------------
class NkImage {
public:
    NkImage(int w, int h) : m_width(w), m_height(h) {
        m_data = new uint8_t[w*h*4]();
    }
    ~NkImage() { delete[] m_data; }
    void SetPixelRGBA(int x, int y,
                      uint8_t r, uint8_t g,
                      uint8_t b, uint8_t a=255) {
        if (x<0||x>=m_width||y<0||y>=m_height) return;
        uint8_t* p = m_data + (y*m_width+x)*4;
        p[0]=r; p[1]=g; p[2]=b; p[3]=a;
    }
    void Fill(uint8_t r, uint8_t g, uint8_t b) {
        for (int y=0; y<m_height; y++)
            for (int x=0; x<m_width; x++)
                SetPixelRGBA(x,y,r,g,b);
    }
    void DrawLine(int x0, int y0, int x1, int y1,
                  uint8_t r, uint8_t g, uint8_t b) {
        int dx = (x1-x0)<0?-(x1-x0):(x1-x0);
        int dy = -(y1-y0)<0?(y1-y0):-(y1-y0);
        int sx = x0<x1?1:-1;
        int sy = y0<y1?1:-1;
        int err = dx+dy;
        while (true) {
            SetPixelRGBA(x0,y0,r,g,b);
            if (x0==x1&&y0==y1) break;
            int e2 = 2*err;
            if (e2>=dy) { err+=dy; x0+=sx; }
            if (e2<=dx) { err+=dx; y0+=sy; }
        }
    }
    void DrawPoint(int x, int y,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (int dy=-2; dy<=2; dy++)
            for (int dx=-2; dx<=2; dx++)
                SetPixelRGBA(x+dx,y+dy,r,g,b);
    }
    bool SavePPM(const char* path) const {
        FILE* f = fopen(path, "wb");
        if (!f) return false;
        fprintf(f, "P6\n%d %d\n255\n", m_width, m_height);
        for (int y=0; y<m_height; y++)
            for (int x=0; x<m_width; x++) {
                uint8_t* p = m_data+(y*m_width+x)*4;
                fwrite(p, 1, 3, f);
            }
        fclose(f);
        return true;
    }
    int Width()  const { return m_width;  }
    int Height() const { return m_height; }
private:
    int      m_width, m_height;
    uint8_t* m_data;
    NkImage(const NkImage&)            = delete;
    NkImage& operator=(const NkImage&) = delete;
};

// ---------------------------------------------------------------
// Mat4d
// ---------------------------------------------------------------
struct Mat4d {
    double data[16];
    Mat4d() {
        for (int i=0; i<16; i++) data[i]=0.0;
        data[0]=data[5]=data[10]=data[15]=1.0;
    }
    double& operator()(int r, int c) {
        assert(r>=0&&r<4&&c>=0&&c<4);
        return data[c*4+r];
    }
    const double& operator()(int r, int c) const {
        return data[c*4+r];
    }
    static Mat4d Identity() { return Mat4d{}; }
    static Mat4d Zero() {
        Mat4d m;
        for (int i=0; i<16; i++) m.data[i]=0.0;
        return m;
    }
    Mat4d Transposed() const {
        Mat4d t = Mat4d::Zero();
        for (int r=0; r<4; r++)
            for (int c=0; c<4; c++)
                t(r,c) = (*this)(c,r);
        return t;
    }
    Mat4d operator*(const Mat4d& o) const {
        Mat4d res = Mat4d::Zero();
        for (int r=0; r<4; r++)
            for (int c=0; c<4; c++) {
                double s=0.0;
                for (int k=0; k<4; k++)
                    s += (*this)(r,k)*o(k,c);
                res(r,c)=s;
            }
        return res;
    }
    Vec4d operator*(const Vec4d& v) const {
        return {
            (*this)(0,0)*v.x+(*this)(0,1)*v.y+
            (*this)(0,2)*v.z+(*this)(0,3)*v.w,
            (*this)(1,0)*v.x+(*this)(1,1)*v.y+
            (*this)(1,2)*v.z+(*this)(1,3)*v.w,
            (*this)(2,0)*v.x+(*this)(2,1)*v.y+
            (*this)(2,2)*v.z+(*this)(2,3)*v.w,
            (*this)(3,0)*v.x+(*this)(3,1)*v.y+
            (*this)(3,2)*v.z+(*this)(3,3)*v.w
        };
    }
    Vec3d operator*(const Vec3d& v) const {
        double w = (*this)(3,0)*v.x+(*this)(3,1)*v.y+
                   (*this)(3,2)*v.z+(*this)(3,3);
        double iw = !nearlyZero(w) ? 1.0/w : 1.0;
        return {
            ((*this)(0,0)*v.x+(*this)(0,1)*v.y+
             (*this)(0,2)*v.z+(*this)(0,3))*iw,
            ((*this)(1,0)*v.x+(*this)(1,1)*v.y+
             (*this)(1,2)*v.z+(*this)(1,3))*iw,
            ((*this)(2,0)*v.x+(*this)(2,1)*v.y+
             (*this)(2,2)*v.z+(*this)(2,3))*iw
        };
    }
    bool operator==(const Mat4d& o) const {
        for (int i=0; i<16; i++)
            if (!approxEq(data[i],o.data[i],1e-10))
                return false;
        return true;
    }
    void ToFloat(float out[16]) const {
        for (int i=0; i<16; i++)
            out[i]=static_cast<float>(data[i]);
    }
    bool Inverse(Mat4d& out) const {
        double aug[4][8];
        for (int r=0; r<4; r++)
            for (int c=0; c<4; c++) {
                aug[r][c]   = (*this)(r,c);
                aug[r][c+4] = (r==c)?1.0:0.0;
            }
        for (int col=0; col<4; col++) {
            int piv=col;
            for (int r=col+1; r<4; r++) {
                double va=aug[r][col]<0?-aug[r][col]:aug[r][col];
                double vb=aug[piv][col]<0?-aug[piv][col]:aug[piv][col];
                if (va>vb) piv=r;
            }
            if (piv!=col)
                for (int c=0; c<8; c++) {
                    double tmp=aug[col][c];
                    aug[col][c]=aug[piv][c];
                    aug[piv][c]=tmp;
                }
            if (nearlyZero(aug[col][col])) return false;
            double inv=1.0/aug[col][col];
            for (int c=0; c<8; c++) aug[col][c]*=inv;
            for (int r=0; r<4; r++) {
                if (r==col) continue;
                double f=aug[r][col];
                for (int c=0; c<8; c++)
                    aug[r][c]-=f*aug[col][c];
            }
        }
        for (int r=0; r<4; r++)
            for (int c=0; c<4; c++)
                out(r,c)=aug[r][c+4];
        return true;
    }
    static Mat4d Translate(const Vec3d& t) {
        Mat4d m;
        m(0,3)=t.x; m(1,3)=t.y; m(2,3)=t.z;
        return m;
    }
    static Mat4d Scale(const Vec3d& s) {
        Mat4d m=Mat4d::Zero();
        m(0,0)=s.x; m(1,1)=s.y;
        m(2,2)=s.z; m(3,3)=1.0;
        return m;
    }
    static Mat4d RotateAxis(const Vec3d& axis, double angle) {
        Vec3d n=axis.Normalized();
        double c=std::cos(angle), s=std::sin(angle), t=1.0-c;
        Mat4d R=Mat4d::Identity();
        R(0,0)=t*n.x*n.x+c;
        R(0,1)=t*n.x*n.y-s*n.z;
        R(0,2)=t*n.x*n.z+s*n.y;
        R(1,0)=t*n.x*n.y+s*n.z;
        R(1,1)=t*n.y*n.y+c;
        R(1,2)=t*n.y*n.z-s*n.x;
        R(2,0)=t*n.x*n.z-s*n.y;
        R(2,1)=t*n.y*n.z+s*n.x;
        R(2,2)=t*n.z*n.z+c;
        return R;
    }
    static Mat4d TRS(const Vec3d& t, const Vec3d& axis,
                     double angle, const Vec3d& s) {
        return Translate(t)*RotateAxis(axis,angle)*Scale(s);
    }
    static Mat4d LookAt(const Vec3d& eye,
                         const Vec3d& target,
                         const Vec3d& up) {
        Vec3d f=(target-eye).Normalized();
        Vec3d r=Cross(f,up).Normalized();
        Vec3d u=Cross(r,f);
        Mat4d V=Mat4d::Identity();
        V(0,0)=r.x; V(0,1)=r.y; V(0,2)=r.z;
        V(1,0)=u.x; V(1,1)=u.y; V(1,2)=u.z;
        V(2,0)=-f.x;V(2,1)=-f.y;V(2,2)=-f.z;
        V(0,3)=-Dot(r,eye);
        V(1,3)=-Dot(u,eye);
        V(2,3)= Dot(f,eye);
        return V;
    }
    void Print() const {
        for (int r=0; r<4; r++) {
            printf("| ");
            for (int c=0; c<4; c++)
                printf("%8.4f ", (*this)(r,c));
            printf("|\n");
        }
    }
};

// ---------------------------------------------------------------
// Projection
// ---------------------------------------------------------------
struct Camera {
    Mat4d  view;
    double fovY, aspect, near, far;
    int    width, height;
};

bool projectPoint(const Camera& cam, const Vec3d& wp,
                  int& px, int& py) {
    Vec4d p4(wp, 1.0);
    Vec4d cp = cam.view * p4;
    if (cp.z >= 0.0) return false;
    double tanH = std::tan(cam.fovY*0.5);
    double fx = (cam.width *0.5)/tanH;
    double fy = (cam.height*0.5)/tanH;
    double cx = cam.width *0.5;
    double cy = cam.height*0.5;
    double u = fx*(-cp.x/cp.z)+cx;
    double v = fy*(-cp.y/cp.z)+cy;
    px=(int)u; py=(int)v;
    if (px<0||px>=cam.width||py<0||py>=cam.height)
        return false;
    return true;
}

// ---------------------------------------------------------------
// TRS Decomposition
// ---------------------------------------------------------------
struct TRSResult {
    Vec3d  translation;
    Mat4d  rotation;
    Vec3d  scale;
    double angleRad;
    Vec3d  axis;
};

TRSResult decomposeTRS(const Mat4d& M) {
    TRSResult res;
    res.translation = Vec3d(M(0,3), M(1,3), M(2,3));
    double sx=std::sqrt(M(0,0)*M(0,0)+M(1,0)*M(1,0)+M(2,0)*M(2,0));
    double sy=std::sqrt(M(0,1)*M(0,1)+M(1,1)*M(1,1)+M(2,1)*M(2,1));
    double sz=std::sqrt(M(0,2)*M(0,2)+M(1,2)*M(1,2)+M(2,2)*M(2,2));
    res.scale = Vec3d(sx, sy, sz);
    res.rotation = Mat4d::Zero();
    if (!nearlyZero(sx)) {
        res.rotation(0,0)=M(0,0)/sx;
        res.rotation(1,0)=M(1,0)/sx;
        res.rotation(2,0)=M(2,0)/sx;
    }
    if (!nearlyZero(sy)) {
        res.rotation(0,1)=M(0,1)/sy;
        res.rotation(1,1)=M(1,1)/sy;
        res.rotation(2,1)=M(2,1)/sy;
    }
    if (!nearlyZero(sz)) {
        res.rotation(0,2)=M(0,2)/sz;
        res.rotation(1,2)=M(1,2)/sz;
        res.rotation(2,2)=M(2,2)/sz;
    }
    res.rotation(3,3)=1.0;
    double trace=res.rotation(0,0)+
                 res.rotation(1,1)+
                 res.rotation(2,2);
    double cosA=(trace-1.0)*0.5;
    cosA=cosA<-1.0?-1.0:(cosA>1.0?1.0:cosA);
    res.angleRad=std::acos(cosA);
    double sinA=std::sin(res.angleRad);
    if (!nearlyZero(sinA)) {
        double inv2s=1.0/(2.0*sinA);
        res.axis=Vec3d(
            (res.rotation(2,1)-res.rotation(1,2))*inv2s,
            (res.rotation(0,2)-res.rotation(2,0))*inv2s,
            (res.rotation(1,0)-res.rotation(0,1))*inv2s
        ).Normalized();
    } else {
        res.axis=Vec3d(0.0,1.0,0.0);
    }
    return res;
}

double pseudoRand(int& seed) {
    seed=seed*1664525+1013904223;
    return (double)(seed&0x7FFFFFFF)/(double)0x7FFFFFFF*2.0-1.0;
}
double randRange(int& seed, double mn, double mx) {
    return mn+(pseudoRand(seed)*0.5+0.5)*(mx-mn);
}

} // namespace NkMath

// ================================================================
// MAIN
// ================================================================
int main() {
    using namespace NkMath;

    // ==========================================================
    // TP1 — inspectFloat (Tests 1 à 6)
    // ==========================================================

    // TP1 - Test 1 : 0.1f
    printf("=== TP1 - Test 1 : 0.1f ===\n");
    inspectFloat(0.1f);

    // TP1 - Test 2 : 1.0f signe=0 exposant=127 mantisse=0
    printf("=== TP1 - Test 2 : 1.0f ===\n");
    inspectFloat(1.0f);
    {
        uint32_t bits; float one=1.0f;
        std::memcpy(&bits,&one,4);
        assert(((bits>>31)&0x1)==0);
        assert(((bits>>23)&0xFF)==127);
        assert((bits&0x7FFFFF)==0);
        printf("  assert signe=0 exposant=127 mantisse=0 OK\n\n");
    }

    // TP1 - Test 3 : +Inf
    printf("=== TP1 - Test 3 : 1.0f/0.0f ===\n");
    inspectFloat(1.0f/0.0f);

    // TP1 - Test 4 : NaN
    printf("=== TP1 - Test 4 : sqrt(-1.0f) ===\n");
    inspectFloat(std::sqrt(-1.0f));
    {
        float nan=std::sqrt(-1.0f);
        assert(nan!=nan);
        printf("  assert NaN!=NaN OK\n\n");
    }

    // TP1 - Test 5 : -0.0f vs +0.0f
    printf("=== TP1 - Test 5 : -0.0f vs +0.0f ===\n");
    inspectFloat(-0.0f);
    inspectFloat(+0.0f);
    assert(-0.0f==+0.0f);
    printf("  assert -0.0f==+0.0f OK\n\n");

    // TP1 - Test 6 : Subnormal
    printf("=== TP1 - Test 6 : numeric_limits::min() ===\n");
    inspectFloat(std::numeric_limits<float>::min());

    // ==========================================================
    // TP2 — Kahan et Welford (Tests 7 à 10)
    // ==========================================================

    // TP2 - Test 7 & 8 : kahanSum vs accumulate
    printf("=== TP2 - Test 7-8 : kahanSum vs accumulate ===\n");
    {
        const int N=1000000;
        std::vector<float> data(N,0.1f);
        float theorique=100000.0f;
        float accum=std::accumulate(data.begin(),data.end(),0.0f);
        float kahan=kahanSum(data.data(),N);
        printf("  theorique  : %.4f\n",theorique);
        printf("  accumulate : %.4f erreur=%.4f\n",
               accum,std::abs(accum-theorique));
        printf("  kahanSum   : %.4f erreur=%.6f\n\n",
               kahan,std::abs(kahan-theorique));
        assert(std::abs(kahan-theorique)
               std::abs(accum-theorique));
        printf("  assert kahan plus precis OK\n\n");
    }

    // TP2 - Test 9 : varianceNaive vs varianceWelford
    printf("=== TP2 - Test 9 : variance ===\n");
    {
        float path[]={1e8f,1e8f,1.0f,2.0f};
        float vn=varianceNaive(path,4);
        float vw=varianceWelford(path,4);
        printf("  varianceNaive   : %.6f %s\n",
               vn,vn<0?"(NEGATIF!)":"");
        printf("  varianceWelford : %.6f\n\n",vw);
        assert(vw>0.0f);
        printf("  assert welford>0 OK\n\n");
    }

    // TP2 - Test 10 : epsilon machine
    printf("=== TP2 - Test 10 : epsilon machine ===\n");
    {
        float eb=measureEpsilon();
        float es=std::numeric_limits<float>::epsilon();
        printf("  boucle : %.10e\n",eb);
        printf("  C++17  : %.10e\n",es);
        assert(eb==es);
        printf("  assert identiques OK\n\n");
    }

    // ==========================================================
    // TP3 — Float.h (Tests 11 à 14)
    // ==========================================================

    // TP3 - Test 11 : isFiniteValid
    printf("=== TP3 - Test 11 : isFiniteValid ===\n");
    assert(!isFiniteValid(std::sqrt(-1.0f)));
    assert(!isFiniteValid(1.0f/0.0f));
    assert(!isFiniteValid(-1.0f/0.0f));
    assert( isFiniteValid(0.0f));
    assert( isFiniteValid(1.0f));
    printf("  NaN +Inf -Inf 0.0f 1.0f OK\n\n");

    // TP3 - Test 12 : nearlyZero
    printf("=== TP3 - Test 12 : nearlyZero ===\n");
    assert( nearlyZero(0.0f));
    assert( nearlyZero(1e-7f));
    assert(!nearlyZero(0.1f));
    assert( nearlyZero(0.0));
    assert( nearlyZero(1e-10));
    assert(!nearlyZero(0.001));
    printf("  tous OK\n\n");

    // TP3 - Test 13 : approxEq
    printf("=== TP3 - Test 13 : approxEq ===\n");
    assert( approxEq(1.0f,1.0f));
    assert( approxEq(1.0f,1.0f+1e-7f));
    assert(!approxEq(1.0f,1.1f));
    assert( approxEq(0.0f,0.0f));
    assert(!approxEq(1.0f,-1.0f));
    assert( approxEq(1.0,1.0+1e-10));
    assert(!approxEq(1.0,1.001));
    printf("  tous OK\n\n");

    // TP3 - Test 14 : kahanSum 10 cas
    printf("=== TP3 - Test 14 : kahanSum 10 cas ===\n");
    {
        float d1[1000]; for(int i=0;i<1000;i++) d1[i]=0.1f;
        assert(approxEq(kahanSum(d1,1000),100.0f,0.001f));
        printf("  cas 1 OK\n");
        float d2[]={42.0f};
        assert(approxEq(kahanSum(d2,1),42.0f));
        printf("  cas 2 OK\n");
        float d3[10]={};
        assert(nearlyZero(kahanSum(d3,10)));
        printf("  cas 3 OK\n");
        float d4[]={-1.0f,-2.0f,-3.0f};
        assert(approxEq(kahanSum(d4,3),-6.0f));
        printf("  cas 4 OK\n");
        float d5[]={1.0f,-1.0f,1.0f,-1.0f};
        assert(nearlyZero(kahanSum(d5,4)));
        printf("  cas 5 OK\n");
        float d6[]={1e6f,1e6f,1e6f};
        assert(approxEq(kahanSum(d6,3),3e6f,1.0f));
        printf("  cas 6 OK\n");
        float d7[]={1e-6f,1e-6f,1e-6f};
        assert(approxEq(kahanSum(d7,3),3e-6f,1e-9f));
        printf("  cas 7 OK\n");
        float d8[]={-99.0f};
        assert(approxEq(kahanSum(d8,1),-99.0f));
        printf("  cas 8 OK\n");
        float d9[]={3.0f,7.0f};
        assert(approxEq(kahanSum(d9,2),10.0f));
        printf("  cas 9 OK\n");
        double d10[100];
        for(int i=0;i<100;i++) d10[i]=0.1;
        double s10=0.0,c10=0.0;
        for(int i=0;i<100;i++){
            double y=d10[i]-c10;
            double t=s10+y;
            c10=(t-s10)-y; s10=t;
        }
        assert(approxEq(s10,10.0,1e-10));
        printf("  cas 10 OK\n\n");
    }

    // ==========================================================
    // TP4 — Vec2d (Tests 15 à 19)
    // ==========================================================

    // TP4 - Test 15 : Dot product
    printf("=== TP4 - Test 15 : Dot product ===\n");
    assert(approxEq(Dot(Vec2d(1,0),Vec2d(0,1)),0.0));
    assert(approxEq(Dot(Vec2d(1,0),Vec2d(1,0)),1.0));
    assert(approxEq(Dot(Vec2d(3,4),Vec2d(3,4)),25.0));
    printf("  tous OK\n\n");

    // TP4 - Test 16 : Cross2D
    printf("=== TP4 - Test 16 : Cross2D ===\n");
    assert(approxEq(Cross2D(Vec2d(1,0),Vec2d(0,1)), 1.0));
    assert(approxEq(Cross2D(Vec2d(0,1),Vec2d(1,0)),-1.0));
    printf("  tous OK\n\n");

    // TP4 - Test 17 : Normalisation
    printf("=== TP4 - Test 17 : Normalisation ===\n");
    {
        Vec2d n=Vec2d(3.0,4.0).Normalized();
        assert(approxEq(n.Norm(),1.0,kEps));
        assert(Vec2d(0,0).Normalized()==Vec2d(0,0));
        assert(n.IsNormalized());
        printf("  tous OK\n\n");
    }

    // TP4 - Test 18 : Operateur []
    printf("=== TP4 - Test 18 : Operateur [] ===\n");
    {
        Vec2d v(5.0,7.0);
        assert(approxEq(v[0],5.0));
        assert(approxEq(v[1],7.0));
        v[0]=99.0;
        assert(approxEq(v[0],99.0));
        assert(approxEq(v.x,99.0));
        printf("  tous OK\n\n");
    }

    // TP4 - Test 19 : Layout memoire
    printf("=== TP4 - Test 19 : Layout memoire ===\n");
    static_assert(sizeof(Vec2d)==16,"Vec2d 16 bytes");
    static_assert(offsetof(Vec2d,x)==0,"x premier");
    static_assert(offsetof(Vec2d,y)==8,"y offset 8");
    printf("  sizeof=16 offsetof OK\n\n");

    // ==========================================================
    // TP5 — Vec3d + Gram-Schmidt (Tests 20 à 23)
    // ==========================================================

    // TP5 - Test 20 : Cross regle main droite
    printf("=== TP5 - Test 20 : Cross main droite ===\n");
    {
        Vec3d X(1,0,0), Y(0,1,0), Z(0,0,1);
        Vec3d r=Cross(X,Y);
        assert(approxEq(r.x,0.0)&&approxEq(r.y,0.0)&&
               approxEq(r.z,1.0));
        assert(Cross(Y,Z)==X);
        assert(Cross(Z,X)==Y);
        printf("  tous OK\n\n");
    }

    // TP5 - Test 21 : Cross non-commutatif
    printf("=== TP5 - Test 21 : Cross non-commutatif ===\n");
    {
        Vec3d X(1,0,0), Y(0,1,0);
        Vec3d r=Cross(Y,X);
        assert(approxEq(r.z,-1.0));
        Vec3d a(1,2,3), b(4,5,6);
        Vec3d ab=Cross(a,b), ba=Cross(b,a);
        assert(approxEq(ab.x,-ba.x)&&
               approxEq(ab.y,-ba.y)&&
               approxEq(ab.z,-ba.z));
        printf("  tous OK\n\n");
    }

    // TP5 - Test 22 : Gram-Schmidt 10 triplets
    printf("=== TP5 - Test 22 : Gram-Schmidt ===\n");
    {
        int seed=42;
        for (int i=0; i<10; i++) {
            Vec3d a(pseudoRand(seed)+2.0,
                    pseudoRand(seed),
                    pseudoRand(seed));
            Vec3d b(pseudoRand(seed),
                    pseudoRand(seed)+2.0,
                    pseudoRand(seed));
            Vec3d c(pseudoRand(seed),
                    pseudoRand(seed),
                    pseudoRand(seed)+2.0);
            OrthoBasis gs=GramSchmidt(a,b,c);
            assert(approxEq(gs.u.Norm(),1.0,1e-9));
            assert(approxEq(gs.v.Norm(),1.0,1e-9));
            assert(approxEq(gs.w.Norm(),1.0,1e-9));
            assert(approxEq(Dot(gs.u,gs.v),0.0,1e-9));
            assert(approxEq(Dot(gs.u,gs.w),0.0,1e-9));
            assert(approxEq(Dot(gs.v,gs.w),0.0,1e-9));
            printf("  triplet %d OK\n",i+1);
        }
        printf("\n");
    }

    // TP5 - Test 23 : Project + Reject = a
    printf("=== TP5 - Test 23 : Project+Reject=a ===\n");
    {
        Vec3d paires[5][2]={
            {Vec3d(3,4,0),Vec3d(1,0,0)},
            {Vec3d(1,2,3),Vec3d(0,1,0)},
            {Vec3d(5,5,5),Vec3d(1,1,0)},
            {Vec3d(2,0,4),Vec3d(0,0,1)},
            {Vec3d(1,3,2),Vec3d(1,1,1)}
        };
        for (int i=0; i<5; i++) {
            Vec3d a=paires[i][0], b=paires[i][1];
            Vec3d proj=Project(a,b), rej=Reject(a,b);
            Vec3d sum=proj+rej;
            assert(approxEq(sum.x,a.x,1e-9)&&
                   approxEq(sum.y,a.y,1e-9)&&
                   approxEq(sum.z,a.z,1e-9));
            assert(approxEq(Dot(proj,rej),0.0,1e-9));
            printf("  paire %d OK\n",i+1);
        }
        printf("\n");
    }

    // ==========================================================
    // TP6 — Vec4d + Projection (Tests 24 à 27)
    // ==========================================================

    // TP6 - Test 24 : 8 coins cube + camera
    printf("=== TP6 - Test 24 : Coins cube ===\n");
    Vec3d coins[8]={
        {-0.5,-0.5,-0.5},{0.5,-0.5,-0.5},
        {0.5,0.5,-0.5},{-0.5,0.5,-0.5},
        {-0.5,-0.5,0.5},{0.5,-0.5,0.5},
        {0.5,0.5,0.5},{-0.5,0.5,0.5}
    };
    double z_cam=2.0;
    Vec3d coins_cam[8];
    for (int i=0; i<8; i++) {
        coins_cam[i]=Vec3d(coins[i].x,
                           coins[i].y,
                           coins[i].z+z_cam);
        assert(coins_cam[i].z>0.0);
    }
    printf("  8 coins devant camera OK\n\n");

    // TP6 - Test 25 : Projection perspective
    printf("=== TP6 - Test 25 : Projection ===\n");
    {
        double fx=500,fy=500,cx=256,cy=256;
        for (int i=0; i<8; i++) {
            double u=fx*(coins_cam[i].x/coins_cam[i].z)+cx;
            double v=fy*(coins_cam[i].y/coins_cam[i].z)+cy;
            assert(u>=0&&u<512&&v>=0&&v<512);
            printf("  coin %d u=%.1f v=%.1f OK\n",i,u,v);
        }
        printf("\n");
    }

    // TP6 - Test 26 : NkImage 512x512
    printf("=== TP6 - Test 26 : NkImage ===\n");
    {
        NkImage img(512,512);
        img.Fill(0,0,0);
        double fx=500,fy=500,cx=256,cy=256;
        for (int i=0; i<8; i++) {
            int px=(int)(fx*(coins_cam[i].x/
                             coins_cam[i].z)+cx);
            int py=(int)(fy*(coins_cam[i].y/
                             coins_cam[i].z)+cy);
            img.DrawPoint(px,py,255,0,0);
        }
        printf("  8 coins dessines OK\n\n");
    }

    // TP6 - Test 27 : 12 aretes
    printf("=== TP6 - Test 27 : 12 aretes ===\n");
    {
        int aretes[12][2]={
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        NkImage img(512,512);
        img.Fill(0,0,0);
        double fx=500,fy=500,cx=256,cy=256;
        int px[8],py[8];
        for (int i=0; i<8; i++) {
            px[i]=(int)(fx*(coins_cam[i].x/
                            coins_cam[i].z)+cx);
            py[i]=(int)(fy*(coins_cam[i].y/
                            coins_cam[i].z)+cy);
        }
        for (int i=0; i<12; i++)
            img.DrawLine(px[aretes[i][0]],py[aretes[i][0]],
                         px[aretes[i][1]],py[aretes[i][1]],
                         0,255,0);
        for (int i=0; i<8; i++)
            img.DrawPoint(px[i],py[i],255,0,0);
        bool saved=img.SavePPM("cube_tp6.ppm");
        assert(saved);
        printf("  12 aretes + PPM OK\n\n");
    }

    // ==========================================================
    // TP7 — Mat4d + Inverse (Tests 28 à 31)
    // ==========================================================

    // TP7 - Test 28 : M x Identity = M
    printf("=== TP7 - Test 28 : M x Identity = M ===\n");
    {
        int seed=12345;
        Mat4d I=Mat4d::Identity();
        for (int i=0; i<10; i++) {
            Mat4d M=Mat4d::Identity();
            M(0,0)=randRange(seed,0.5,3.0);
            M(1,1)=randRange(seed,0.5,3.0);
            M(2,2)=randRange(seed,0.5,3.0);
            assert((M*I)==M);
            assert((I*M)==M);
            printf("  matrice %d OK\n",i+1);
        }
        printf("\n");
    }

    // TP7 - Test 29 : M x M⁻¹ = I a 1e-10
    printf("=== TP7 - Test 29 : M x Inv = I ===\n");
    {
        int seed=99999;
        for (int i=0; i<10; i++) {
            Mat4d M=Mat4d::Zero();
            for (int r=0; r<4; r++) {
                double rs=0.0;
                for (int c=0; c<4; c++) {
                    if (r!=c) {
                        M(r,c)=pseudoRand(seed)*0.3;
                        rs+=M(r,c)<0?-M(r,c):M(r,c);
                    }
                }
                M(r,r)=rs+1.0;
            }
            Mat4d Inv=Mat4d::Zero();
            bool ok=M.Inverse(Inv);
            assert(ok);
            Mat4d MInv=M*Inv;
            for (int r=0; r<4; r++)
                for (int c=0; c<4; c++) {
                    double ex=(r==c)?1.0:0.0;
                    double d=MInv(r,c)-ex;
                    d=d<0?-d:d;
                    assert(d<1e-10);
                }
            printf("  matrice %d OK\n",i+1);
        }
        printf("\n");
    }

    // TP7 - Test 30 : matrice singuliere
    printf("=== TP7 - Test 30 : Singuliere ===\n");
    {
        Mat4d S=Mat4d::Zero();
        S(0,0)=1.0; S(1,1)=1.0;
        Mat4d inv;
        assert(!S.Inverse(inv));
        printf("  Inverse retourne false OK\n\n");
    }

    // TP7 - Test 31 : RotateAxis Y PI/2
    printf("=== TP7 - Test 31 : RotateAxis ===\n");
    {
        Mat4d R=Mat4d::RotateAxis(Vec3d(0,1,0),kPi/2.0);
        Vec4d v(1,0,0,1);
        Vec4d r=R*v;
        assert(approxEq(r.x, 0.0,1e-10));
        assert(approxEq(r.y, 0.0,1e-10));
        assert(approxEq(r.z,-1.0,1e-10));
        assert(approxEq(r.w, 1.0,1e-10));
        printf("  {1,0,0,1} → {0,0,-1,1} OK\n\n");
    }

    // ==========================================================
    // TP8 — Rasteriseur LookAt (Tests 32 à 37)
    // ==========================================================

    // TP8 - Test 32 : Cube unitaire
    printf("=== TP8 - Test 32 : Cube unitaire ===\n");
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
    printf("  8 coins 12 aretes OK\n\n");

    // TP8 - Test 33 : LookAt
    printf("=== TP8 - Test 33 : LookAt ===\n");
    Mat4d viewMat=Mat4d::LookAt(
        Vec3d(0,1,3), Vec3d(0,0,0), Vec3d(0,1,0)
    );
    printf("  LookAt cree OK\n\n");

    // TP8 - Test 34 : Camera fov=60
    printf("=== TP8 - Test 34 : Camera fov=60 ===\n");
    Camera cam;
    cam.view=viewMat;
    cam.fovY=kPi/3.0;
    cam.aspect=1.0;
    cam.near=0.1; cam.far=100.0;
    cam.width=512; cam.height=512;
    printf("  Camera OK\n\n");

    // TP8 - Tests 35 36 37 : 10 frames rotation
    printf("=== TP8 - Tests 35-37 : 10 frames ===\n");
    for (int frame=0; frame<10; frame++) {
        double angle=frame*(kPi/5.0);
        Mat4d rot=Mat4d::RotateAxis(Vec3d(0,1,0),angle);
        NkImage img(512,512);
        img.Fill(30,30,30);
        int px[8],py[8]; bool vis[8];
        for (int i=0; i<8; i++) {
            Vec4d c4(cube[i],1.0);
            Vec4d r=rot*c4;
            vis[i]=projectPoint(cam,r.ToVec3(),px[i],py[i]);
        }
        for (int i=0; i<12; i++) {
            int a=aretes[i][0], b=aretes[i][1];
            if (vis[a]&&vis[b])
                img.DrawLine(px[a],py[a],px[b],py[b],255,0,0);
        }
        for (int i=0; i<8; i++)
            if (vis[i])
                img.DrawPoint(px[i],py[i],255,255,255);
        char fn[64];
        snprintf(fn,sizeof(fn),"frame_%02d.ppm",frame);
        bool saved=img.SavePPM(fn);
        assert(saved);
        printf("  frame %2d angle=%.0f° OK\n",
               frame,angle*180.0/kPi);
    }
    printf("\n");

    // ==========================================================
    // TP9 — TRS + Decomposition (Tests 38 à 40)
    // ==========================================================

    // TP9 - Tests 38 39 40 : 20 triplets TRS
    printf("=== TP9 - Tests 38-40 : TRS 20 triplets ===\n");
    {
        int seed=777;
        for (int i=0; i<20; i++) {

            // TP9 - Test 38 : construit M = TRS
            Vec3d T(randRange(seed,-5,5),
                    randRange(seed,-5,5),
                    randRange(seed,-5,5));
            Vec3d axis(pseudoRand(seed),
                       pseudoRand(seed),
                       pseudoRand(seed));
            axis=axis.Normalized();
            if (nearlyZero(axis.Norm()))
                axis=Vec3d(0,1,0);
            double angle=randRange(seed,-kPi,kPi);
            Vec3d S(randRange(seed,0.5,3.0),
                    randRange(seed,0.5,3.0),
                    randRange(seed,0.5,3.0));
            Mat4d M=Mat4d::TRS(T,axis,angle,S);

            // TP9 - Test 39 : decompose M
            TRSResult res=decomposeTRS(M);

            // TP9 - Test 40 : verifie T S R
            assert(approxEq(res.translation.x,T.x,1e-9)&&
                   approxEq(res.translation.y,T.y,1e-9)&&
                   approxEq(res.translation.z,T.z,1e-9));
            assert(approxEq(res.scale.x,S.x,1e-9)&&
                   approxEq(res.scale.y,S.y,1e-9)&&
                   approxEq(res.scale.z,S.z,1e-9));
            Mat4d Mr=Mat4d::TRS(res.translation,
                                res.axis,
                                res.angleRad,
                                res.scale);
            bool R_ok=true;
            for (int r=0; r<3; r++)
                for (int c=0; c<3; c++)
                    if (!approxEq(M(r,c),Mr(r,c),1e-9))
                        R_ok=false;
            assert(R_ok);
            printf("  triplet %2d T S R OK\n",i+1);
        }
        printf("\n");
    }

    printf("=== Tous les tests TP1 a TP9 passes ! ===\n");
    return 0;
}