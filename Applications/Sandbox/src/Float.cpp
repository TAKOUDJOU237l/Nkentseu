#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>

namespace NkMath {

constexpr double kEps  = 1e-9;
constexpr float  kFEps = 1e-6f;

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
    printf("  mantisse = 0x%06X | bin = ",
           mantissa);
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

} // namespace NkMath

int main() {
    using namespace NkMath;

    // Tests 1-6 : inspectFloat
    inspectFloat(0.1f);
    inspectFloat(1.0f);
    inspectFloat(1.0f / 0.0f);
    inspectFloat(std::sqrt(-1.0f));
    inspectFloat(-0.0f);
    inspectFloat(+0.0f);
    printf("-0.0f == +0.0f : %s\n\n",
           (-0.0f == +0.0f) ? "VRAI" : "FAUX");
    inspectFloat(std::numeric_limits<float>::min());

    // Tests 7-8 : kahanSum vs accumulate
    const int N = 1000000;
    std::vector<float> data(N, 0.1f);
    float accum = std::accumulate(
        data.begin(), data.end(), 0.0f
    );
    float kahan = kahanSum(data.data(), N);
    printf("accumulate : %.4f erreur=%.4f\n",
           accum, std::abs(accum - 100000.0f));
    printf("kahanSum   : %.4f erreur=%.6f\n\n",
           kahan, std::abs(kahan - 100000.0f));

    // Test 9 : variance
    float path[] = {1e8f, 1e8f, 1.0f, 2.0f};
    printf("varianceNaive   : %.6f\n",
           varianceNaive(path, 4));
    printf("varianceWelford : %.6f\n\n",
           varianceWelford(path, 4));

    // Test 10 : epsilon machine
    printf("epsilon boucle  : %.10e\n",
           measureEpsilon());
    printf("epsilon C++17   : %.10e\n",
           std::numeric_limits<float>::epsilon());

    return 0;
}