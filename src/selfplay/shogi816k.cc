#include "shogi816k.h"

#include <random>

namespace nshogi {
namespace engine {
namespace selfplay {

PositionBuilderShogi816k::PositionBuilderShogi816k() {
    Sliders[core::Black][0] = core::Sq9H;
    Sliders[core::Black][1] = core::Sq8H;
    Sliders[core::Black][2] = core::Sq7H;
    Sliders[core::Black][3] = core::Sq6H;
    Sliders[core::Black][4] = core::Sq5H;
    Sliders[core::Black][5] = core::Sq4H;
    Sliders[core::Black][6] = core::Sq3H;
    Sliders[core::Black][7] = core::Sq2H;
    Sliders[core::Black][8] = core::Sq1H;
    Steps[core::Black][0] = core::Sq9I;
    Steps[core::Black][1] = core::Sq8I;
    Steps[core::Black][2] = core::Sq7I;
    Steps[core::Black][3] = core::Sq6I;
    Steps[core::Black][4] = core::Sq5I;
    Steps[core::Black][5] = core::Sq4I;
    Steps[core::Black][6] = core::Sq3I;
    Steps[core::Black][7] = core::Sq2I;
    Steps[core::Black][8] = core::Sq1I;

    Sliders[core::White][0] = core::Sq9B;
    Sliders[core::White][1] = core::Sq8B;
    Sliders[core::White][2] = core::Sq7B;
    Sliders[core::White][3] = core::Sq6B;
    Sliders[core::White][4] = core::Sq5B;
    Sliders[core::White][5] = core::Sq4B;
    Sliders[core::White][6] = core::Sq3B;
    Sliders[core::White][7] = core::Sq2B;
    Sliders[core::White][8] = core::Sq1B;
    Steps[core::White][0] = core::Sq9A;
    Steps[core::White][1] = core::Sq8A;
    Steps[core::White][2] = core::Sq7A;
    Steps[core::White][3] = core::Sq6A;
    Steps[core::White][4] = core::Sq5A;
    Steps[core::White][5] = core::Sq4A;
    Steps[core::White][6] = core::Sq3A;
    Steps[core::White][7] = core::Sq2A;
    Steps[core::White][8] = core::Sq1A;
}

core::Position PositionBuilderShogi816k::build() {
    shuffle();

    for (std::size_t I = 0; I < 81; ++I) {
        setPiece((core::Square)I, core::PK_Empty);
    }
    for (std::size_t I = 0; I < 9; ++I) {
        setPiece(core::Sq9C, core::PK_WhitePawn);
        setPiece(core::Sq8C, core::PK_WhitePawn);
        setPiece(core::Sq7C, core::PK_WhitePawn);
        setPiece(core::Sq6C, core::PK_WhitePawn);
        setPiece(core::Sq5C, core::PK_WhitePawn);
        setPiece(core::Sq4C, core::PK_WhitePawn);
        setPiece(core::Sq3C, core::PK_WhitePawn);
        setPiece(core::Sq2C, core::PK_WhitePawn);
        setPiece(core::Sq1C, core::PK_WhitePawn);
        setPiece(core::Sq9G, core::PK_BlackPawn);
        setPiece(core::Sq8G, core::PK_BlackPawn);
        setPiece(core::Sq7G, core::PK_BlackPawn);
        setPiece(core::Sq6G, core::PK_BlackPawn);
        setPiece(core::Sq5G, core::PK_BlackPawn);
        setPiece(core::Sq4G, core::PK_BlackPawn);
        setPiece(core::Sq3G, core::PK_BlackPawn);
        setPiece(core::Sq2G, core::PK_BlackPawn);
        setPiece(core::Sq1G, core::PK_BlackPawn);
    }

    setPiece(Sliders[core::Black][0], core::PK_BlackBishop);
    setPiece(Sliders[core::Black][1], core::PK_BlackRook);
    setPiece(Steps[core::Black][0], core::PK_BlackLance);
    setPiece(Steps[core::Black][1], core::PK_BlackKnight);
    setPiece(Steps[core::Black][2], core::PK_BlackSilver);
    setPiece(Steps[core::Black][3], core::PK_BlackGold);
    setPiece(Steps[core::Black][4], core::PK_BlackKing);
    setPiece(Steps[core::Black][5], core::PK_BlackGold);
    setPiece(Steps[core::Black][6], core::PK_BlackSilver);
    setPiece(Steps[core::Black][7], core::PK_BlackKnight);
    setPiece(Steps[core::Black][8], core::PK_BlackLance);

    setPiece(Sliders[core::White][0], core::PK_WhiteBishop);
    setPiece(Sliders[core::White][1], core::PK_WhiteRook);
    setPiece(Steps[core::White][0], core::PK_WhiteLance);
    setPiece(Steps[core::White][1], core::PK_WhiteKnight);
    setPiece(Steps[core::White][2], core::PK_WhiteSilver);
    setPiece(Steps[core::White][3], core::PK_WhiteGold);
    setPiece(Steps[core::White][4], core::PK_WhiteKing);
    setPiece(Steps[core::White][5], core::PK_WhiteGold);
    setPiece(Steps[core::White][6], core::PK_WhiteSilver);
    setPiece(Steps[core::White][7], core::PK_WhiteKnight);
    setPiece(Steps[core::White][8], core::PK_WhiteLance);

    return core::PositionBuilder::build();
}

void PositionBuilderShogi816k::shuffle() {
    static std::mt19937_64 MT(20240630);

    for (std::size_t I = 0; I < 8; ++I) {
        std::uniform_int_distribution<std::size_t> Dist(I, 8);

        const std::size_t R1 = Dist(MT);
        if (R1 != I) {
            std::swap(Sliders[core::Black][I], Sliders[core::Black][R1]);
        }

        const std::size_t R2 = Dist(MT);
        if (R2 != I) {
            std::swap(Steps[core::Black][I], Steps[core::Black][R2]);
        }

        const std::size_t R3 = Dist(MT);
        if (R3 != I) {
            std::swap(Sliders[core::White][I], Sliders[core::White][R3]);
        }

        const std::size_t R4 = Dist(MT);
        if (R4 != I) {
            std::swap(Steps[core::White][I], Steps[core::White][R4]);
        }
    }
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
