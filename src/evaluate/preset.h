#ifndef NSHOGI_ENGINE_EVALUATE_PRESET_H
#define NSHOGI_ENGINE_EVALUATE_PRESET_H


#include <nshogi/ml/featurestack.h>

namespace nshogi {
namespace engine {
namespace evaluate {
namespace preset {

using SimpleFeatures = ml::FeatureStackComptime<
    ml::FeatureType::FT_MyPawn,    ml::FeatureType::FT_MyLance,    ml::FeatureType::FT_MyKnight,    ml::FeatureType::FT_MySilver,    ml::FeatureType::FT_MyGold,      ml::FeatureType::FT_MyKing,    ml::FeatureType::FT_MyBishop, ml::FeatureType::FT_MyRook,
    ml::FeatureType::FT_MyProPawn, ml::FeatureType::FT_MyProLance, ml::FeatureType::FT_MyProKnight, ml::FeatureType::FT_MyProSilver, ml::FeatureType::FT_MyProBishop, ml::FeatureType::FT_MyProRook,
    ml::FeatureType::FT_OpPawn,    ml::FeatureType::FT_OpLance,    ml::FeatureType::FT_OpKnight,    ml::FeatureType::FT_OpSilver,    ml::FeatureType::FT_OpGold,      ml::FeatureType::FT_OpKing,    ml::FeatureType::FT_OpBishop, ml::FeatureType::FT_OpRook,
    ml::FeatureType::FT_OpProPawn, ml::FeatureType::FT_OpProLance, ml::FeatureType::FT_OpProKnight, ml::FeatureType::FT_OpProSilver, ml::FeatureType::FT_OpProBishop, ml::FeatureType::FT_OpProRook,

    ml::FeatureType::FT_MyStandPawn1,   ml::FeatureType::FT_MyStandPawn2,   ml::FeatureType::FT_MyStandPawn3,   ml::FeatureType::FT_MyStandPawn4,
    ml::FeatureType::FT_MyStandLance1,  ml::FeatureType::FT_MyStandLance2,  ml::FeatureType::FT_MyStandLance3,  ml::FeatureType::FT_MyStandLance4,
    ml::FeatureType::FT_MyStandKnight1, ml::FeatureType::FT_MyStandKnight2, ml::FeatureType::FT_MyStandKnight3, ml::FeatureType::FT_MyStandKnight4,
    ml::FeatureType::FT_MyStandSilver1, ml::FeatureType::FT_MyStandSilver2, ml::FeatureType::FT_MyStandSilver3, ml::FeatureType::FT_MyStandSilver4,
    ml::FeatureType::FT_MyStandGold1,   ml::FeatureType::FT_MyStandGold2,   ml::FeatureType::FT_MyStandGold3,   ml::FeatureType::FT_MyStandGold4,
    ml::FeatureType::FT_MyStandBishop1, ml::FeatureType::FT_MyStandBishop2,
    ml::FeatureType::FT_MyStandRook1,   ml::FeatureType::FT_MyStandRook2,

    ml::FeatureType::FT_OpStandPawn1,   ml::FeatureType::FT_OpStandPawn2,   ml::FeatureType::FT_OpStandPawn3,   ml::FeatureType::FT_OpStandPawn4,
    ml::FeatureType::FT_OpStandLance1,  ml::FeatureType::FT_OpStandLance2,  ml::FeatureType::FT_OpStandLance3,  ml::FeatureType::FT_OpStandLance4,
    ml::FeatureType::FT_OpStandKnight1, ml::FeatureType::FT_OpStandKnight2, ml::FeatureType::FT_OpStandKnight3, ml::FeatureType::FT_OpStandKnight4,
    ml::FeatureType::FT_OpStandSilver1, ml::FeatureType::FT_OpStandSilver2, ml::FeatureType::FT_OpStandSilver3, ml::FeatureType::FT_OpStandSilver4,
    ml::FeatureType::FT_OpStandGold1,   ml::FeatureType::FT_OpStandGold2,   ml::FeatureType::FT_OpStandGold3,   ml::FeatureType::FT_OpStandGold4,
    ml::FeatureType::FT_OpStandBishop1, ml::FeatureType::FT_OpStandBishop2,
    ml::FeatureType::FT_OpStandRook1,   ml::FeatureType::FT_OpStandRook2,

    ml::FeatureType::FT_Black, ml::FeatureType::FT_White>;

using CustomFeaturesV1 = ml::FeatureStackComptime<
    ml::FeatureType::FT_MyPawn,    ml::FeatureType::FT_MyLance,    ml::FeatureType::FT_MyKnight,    ml::FeatureType::FT_MySilver,    ml::FeatureType::FT_MyGold,      ml::FeatureType::FT_MyKing,    ml::FeatureType::FT_MyBishop, ml::FeatureType::FT_MyRook,
    ml::FeatureType::FT_MyProPawn, ml::FeatureType::FT_MyProLance, ml::FeatureType::FT_MyProKnight, ml::FeatureType::FT_MyProSilver, ml::FeatureType::FT_MyProBishop, ml::FeatureType::FT_MyProRook,
    ml::FeatureType::FT_OpPawn,    ml::FeatureType::FT_OpLance,    ml::FeatureType::FT_OpKnight,    ml::FeatureType::FT_OpSilver,    ml::FeatureType::FT_OpGold,      ml::FeatureType::FT_OpKing,    ml::FeatureType::FT_OpBishop, ml::FeatureType::FT_OpRook,
    ml::FeatureType::FT_OpProPawn, ml::FeatureType::FT_OpProLance, ml::FeatureType::FT_OpProKnight, ml::FeatureType::FT_OpProSilver, ml::FeatureType::FT_OpProBishop, ml::FeatureType::FT_OpProRook,

    ml::FeatureType::FT_MyStandPawn1,   ml::FeatureType::FT_MyStandPawn2,   ml::FeatureType::FT_MyStandPawn3,   ml::FeatureType::FT_MyStandPawn4, ml::FeatureType::FT_MyStandPawn5, ml::FeatureType::FT_MyStandPawn6,
    ml::FeatureType::FT_MyStandLance1,  ml::FeatureType::FT_MyStandLance2,  ml::FeatureType::FT_MyStandLance3,  ml::FeatureType::FT_MyStandLance4,
    ml::FeatureType::FT_MyStandKnight1, ml::FeatureType::FT_MyStandKnight2, ml::FeatureType::FT_MyStandKnight3, ml::FeatureType::FT_MyStandKnight4,
    ml::FeatureType::FT_MyStandSilver1, ml::FeatureType::FT_MyStandSilver2, ml::FeatureType::FT_MyStandSilver3, ml::FeatureType::FT_MyStandSilver4,
    ml::FeatureType::FT_MyStandGold1,   ml::FeatureType::FT_MyStandGold2,   ml::FeatureType::FT_MyStandGold3,   ml::FeatureType::FT_MyStandGold4,
    ml::FeatureType::FT_MyStandBishop1, ml::FeatureType::FT_MyStandBishop2,
    ml::FeatureType::FT_MyStandRook1,   ml::FeatureType::FT_MyStandRook2,

    ml::FeatureType::FT_OpStandPawn1,   ml::FeatureType::FT_OpStandPawn2,   ml::FeatureType::FT_OpStandPawn3,   ml::FeatureType::FT_OpStandPawn4, ml::FeatureType::FT_OpStandPawn5, ml::FeatureType::FT_OpStandPawn6,
    ml::FeatureType::FT_OpStandLance1,  ml::FeatureType::FT_OpStandLance2,  ml::FeatureType::FT_OpStandLance3,  ml::FeatureType::FT_OpStandLance4,
    ml::FeatureType::FT_OpStandKnight1, ml::FeatureType::FT_OpStandKnight2, ml::FeatureType::FT_OpStandKnight3, ml::FeatureType::FT_OpStandKnight4,
    ml::FeatureType::FT_OpStandSilver1, ml::FeatureType::FT_OpStandSilver2, ml::FeatureType::FT_OpStandSilver3, ml::FeatureType::FT_OpStandSilver4,
    ml::FeatureType::FT_OpStandGold1,   ml::FeatureType::FT_OpStandGold2,   ml::FeatureType::FT_OpStandGold3,   ml::FeatureType::FT_OpStandGold4,
    ml::FeatureType::FT_OpStandBishop1, ml::FeatureType::FT_OpStandBishop2,
    ml::FeatureType::FT_OpStandRook1,   ml::FeatureType::FT_OpStandRook2,

    ml::FeatureType::FT_Black, ml::FeatureType::FT_White,

    ml::FeatureType::FT_Check,
    ml::FeatureType::FT_NoMyPawnFile, ml::FeatureType::FT_NoOpPawnFile, ml::FeatureType::FT_Progress, ml::FeatureType::FT_ProgressUnit,

    ml::FeatureType::FT_MyDrawValue, ml::FeatureType::FT_OpDrawValue,

    ml::FeatureType::FT_MyDeclarationScore, ml::FeatureType::FT_OpDeclarationScore,
    ml::FeatureType::FT_MyPieceScore, ml::FeatureType::FT_OpPieceScore>;


} // namespace preset
} // namespace evaluate
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_EVALUATE_PRESET_H
