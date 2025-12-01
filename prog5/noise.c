#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

// XOROSHIRO128+ PRNG state
uint64_t s[2] = {1, 2}; // Some non-zero seed, you may initialize with time or a seed function

void seed_xoro(int seed) {
    s[1] = seed;
    s[2] = seed + 1;
}

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xoroshiro128plus(void) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    s[1] = rotl(s1, 36); // c

    return result;
}

// Returns a double uniform in [0,1)
double uniform() {
    // xoroshiro128+ outputs 64-bit unsigned integer, map to [0,1)
    return (xoroshiro128plus() >> 11) * (1.0/9007199254740992.0); // 53-bit precision double
}

double gaussian_noise() {
    static int has_spare = 0;
    static double spare;

    if (has_spare)
    {
        has_spare = 0;
        return spare;
    }

    has_spare = 1;
    double u, v, s;
    do
    {
        u = uniform() * 2.0 - 1.0;
        v = uniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}
