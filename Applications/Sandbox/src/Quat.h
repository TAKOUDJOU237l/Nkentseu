#pragma once
#include <cmath>
#include <cstdio>
#include <cassert>
#include "Float.h"
#include "Vec3d.h"
#include "Mat4d.h"

namespace NkMath {

struct Mat3d {
    double data[9];
    Mat3d() {
        for (int i=0;i<9;i++) data[i]=0.0;
        data[0]=data[4]=data[8]=1.0;
    }
    double& operator()(int r, int c) { return data[c*3+r]; }
    const double& operator()(int r, int c) const { return data[c*3+r]; }
    static Mat3d Identity() { return Mat3d{}; }
    static Mat3d Zero() {
        Mat3d m;
        for (int i=0;i<9;i++) m.data[i]=0.0;
        return m;
    }
};

struct Quat {
    double w, x, y, z;

    Quat() : w(1), x(0), y(0), z(0) {}
    Quat(double w, double x, double y, double z)
        : w(w), x(x), y(y), z(z) {}

    double Norm2() const { return w*w+x*x+y*y+z*z; }
    double Norm()  const { return std::sqrt(Norm2()); }

    Quat Normalized() const {
        double n=Norm();
        assert(!nearlyZero(n));
        return {w/n, x/n, y/n, z/n};
    }

    Quat Conjugate() const { return {w,-x,-y,-z}; }

    Quat Inverse() const {
        double n2=Norm2();
        assert(!nearlyZero(n2));
        return {w/n2, -x/n2, -y/n2, -z/n2};
    }

    // Produit de Hamilton
    Quat operator*(const Quat& o) const {
        return {
            w*o.w - x*o.x - y*o.y - z*o.z,
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w
        };
    }

    bool operator==(const Quat& o) const {
        return approxEq(w,o.w)&&approxEq(x,o.x)&&
               approxEq(y,o.y)&&approxEq(z,o.z);
    }
};

// Quaternion depuis axe-angle
inline Quat FromAxisAngle(const Vec3d& axis, double angleRad) {
    Vec3d n=axis.Normalized();
    double s=std::sin(angleRad/2.0);
    double c=std::cos(angleRad/2.0);
    return {c, n.x*s, n.y*s, n.z*s};
}

// Rotation d'un vecteur par un quaternion
// v' = q x (0,v) x q* (version optimisee)
inline Vec3d Rotate(const Quat& q, const Vec3d& v) {
    Vec3d qVec={q.x, q.y, q.z};
    Vec3d uv  =Cross(qVec, v);
    Vec3d uuv =Cross(qVec, uv);
    return v + (uv*(2.0*q.w)) + (uuv*2.0);
}

// Quat vers Mat3
inline Mat3d ToMat3(const Quat& q) {
    double xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    double xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    double wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    Mat3d R;
    R(0,0)=1-2*(yy+zz); R(0,1)=2*(xy-wz);   R(0,2)=2*(xz+wy);
    R(1,0)=2*(xy+wz);   R(1,1)=1-2*(xx+zz); R(1,2)=2*(yz-wx);
    R(2,0)=2*(xz-wy);   R(2,1)=2*(yz+wx);   R(2,2)=1-2*(xx+yy);
    return R;
}

// Mat3 vers Quat -- Methode de Shepperd
inline Quat FromMat3(const Mat3d& R) {
    double trace=R(0,0)+R(1,1)+R(2,2);
    Quat q;
    if (trace>0) {
        double s=0.5/std::sqrt(trace+1.0);
        q.w=0.25/s;
        q.x=(R(2,1)-R(1,2))*s;
        q.y=(R(0,2)-R(2,0))*s;
        q.z=(R(1,0)-R(0,1))*s;
    } else if (R(0,0)>R(1,1)&&R(0,0)>R(2,2)) {
        double s=2.0*std::sqrt(1.0+R(0,0)-R(1,1)-R(2,2));
        q.w=(R(2,1)-R(1,2))/s;
        q.x=0.25*s;
        q.y=(R(0,1)+R(1,0))/s;
        q.z=(R(0,2)+R(2,0))/s;
    } else if (R(1,1)>R(2,2)) {
        double s=2.0*std::sqrt(1.0+R(1,1)-R(0,0)-R(2,2));
        q.w=(R(0,2)-R(2,0))/s;
        q.x=(R(0,1)+R(1,0))/s;
        q.y=0.25*s;
        q.z=(R(1,2)+R(2,1))/s;
    } else {
        double s=2.0*std::sqrt(1.0+R(2,2)-R(0,0)-R(1,1));
        q.w=(R(1,0)-R(0,1))/s;
        q.x=(R(0,2)+R(2,0))/s;
        q.y=(R(1,2)+R(2,1))/s;
        q.z=0.25*s;
    }
    return q.Normalized();
}

// SLERP -- chemin geodesique sur S3
inline Quat Slerp(Quat a, Quat b, double t) {
    double cosAngle=a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;
    // Chemin court
    if (cosAngle<0.0) {
        b.w=-b.w; b.x=-b.x; b.y=-b.y; b.z=-b.z;
        cosAngle=-cosAngle;
    }
    double k0, k1;
    if (cosAngle>0.9999) {
        // LERP si angle trop petit
        k0=1.0-t; k1=t;
    } else {
        double angle=std::acos(cosAngle);
        double sinAngle=std::sin(angle);
        k0=std::sin((1.0-t)*angle)/sinAngle;
        k1=std::sin(t*angle)/sinAngle;
    }
    return Quat{
        k0*a.w+k1*b.w,
        k0*a.x+k1*b.x,
        k0*a.y+k1*b.y,
        k0*a.z+k1*b.z
    }.Normalized();
}

// LERP simple (pour comparaison avec SLERP)
inline Quat LerpQuat(const Quat& a, const Quat& b, double t) {
    return Quat{
        a.w+(b.w-a.w)*t,
        a.x+(b.x-a.x)*t,
        a.y+(b.y-a.y)*t,
        a.z+(b.z-a.z)*t
    }.Normalized();
}

// Quat vers Mat4d pour rendu
inline Mat4d QuatToMat4(const Quat& q) {
    double xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    double xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    double wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    Mat4d M=Mat4d::Identity();
    M(0,0)=1-2*(yy+zz); M(0,1)=2*(xy-wz);   M(0,2)=2*(xz+wy);
    M(1,0)=2*(xy+wz);   M(1,1)=1-2*(xx+zz); M(1,2)=2*(yz-wx);
    M(2,0)=2*(xz-wy);   M(2,1)=2*(yz+wx);   M(2,2)=1-2*(xx+yy);
    return M;
}

} // namespace NkMath