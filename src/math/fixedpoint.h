//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MATH_FIXEDPOINT_H
#define NSHOGI_ENGINE_MATH_FIXEDPOINT_H

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace nshogi {
namespace engine {
namespace math {

class FixedPoint64 {
 public:
    constexpr FixedPoint64()
        : Data(0) {
    }

    FixedPoint64(double D) {
        assert(D >= 0 && D <= 1);

        uint64_t RawBinary;
        std::memcpy(reinterpret_cast<char*>(&RawBinary),
                    reinterpret_cast<const char*>(&D), sizeof(double));

        if (RawBinary == 0) {
            Data = 0;
            return;
        }

        const int8_t Exponent =
            static_cast<int8_t>(1023 - ((RawBinary >> 52) & 0x7FF));
        const uint64_t Mantissa =
            (RawBinary & 0xFFFFFFFFFFFFF) >> (52 - FractionBits);

        Data = (Mantissa | (1ULL << FractionBits)) >> Exponent;
    }

    FixedPoint64(float F)
        : FixedPoint64(static_cast<double>(F)) {
    }

    FixedPoint64(uint32_t Integer)
        : Data((uint64_t)Integer << FractionBits) {
    }

    void add(double D) {
        FixedPoint64 Val(D);
        Data += Val.Data;
    }

    void addOne() {
        Data += 1ULL << FractionBits;
    }

    FixedPoint64 operator+(const FixedPoint64& FP) const {
        return FixedPoint64(Data + FP.Data);
    }

    FixedPoint64 operator-(const FixedPoint64& FP) const {
        return FixedPoint64(Data - FP.Data);
    }

    FixedPoint64 operator/(const FixedPoint64& FP) const {
        return FixedPoint64(Data / FP.Data);
    }

    FixedPoint64 operator*(const FixedPoint64& FP) const {
        return FixedPoint64((Data >> (FractionBits / 2)) *
                            (FP.Data >> (FractionBits / 2)));
    }

    bool operator>(const FixedPoint64& FP) const {
        return Data > FP.Data;
    }

    double toDouble() const {
        const uint64_t Integer = Data >> FractionBits;
        const uint64_t Fraction = Data & ((1ULL << FractionBits) - 1);

        return static_cast<double>(Integer) +
               (static_cast<double>(Fraction) /
                static_cast<double>(1ULL << FractionBits));
    }

 private:
    static constexpr uint8_t FractionBits = 36; // must be even.

    FixedPoint64(uint64_t RawData)
        : Data(RawData) {
    }

    uint64_t Data;

    friend struct AtomicFixedPoint64;
};

struct AtomicFixedPoint64 {
 public:
    AtomicFixedPoint64(uint64_t V)
        : Data(V) {
    }

    void store(double Value, std::memory_order Order) {
        FixedPoint64 FP(Value);
        Data.store(FP.Data, Order);
    }

    FixedPoint64 load(std::memory_order Order) const {
        return Data.load(Order);
    }

    void fetch_add(double Value, std::memory_order Order) {
        FixedPoint64 FP(Value);
        Data.fetch_add(FP.Data, Order);
    }

 private:
    std::atomic<uint64_t> Data;
};

} // namespace math
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MATH_FIXEDPOINT_H
