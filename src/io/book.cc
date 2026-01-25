//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "book.h"
#include "../book/bookmaker.h"
#include "../book/node.h"

#include <cinttypes>
#include <cmath>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace io {
namespace book {
//
// namespace {
//
// int convertWinRateToScoreCP(double WinRate) {
//     constexpr double Constant = 600;
//
//     if (WinRate == 0.0) {
//         return -30000;
//     } else if (WinRate == 1.0) {
//         return 30000;
//     }
//
//     return (int)(-Constant * std::log(1.0 / WinRate - 1));
// }
//
// } // namespace
//
// void save(const engine::book::Book& Book, std::ofstream& Ofs,
//           Format OutputFormat) {
//     if (OutputFormat == Format::NShogi) {
//         for (const auto& [Sfen, Index] : Book.Dictionary) {
//             const auto& Entry = Book.Entries[Index];
//
//             const std::size_t Size = Sfen.size() + 1;
//             Ofs.write(reinterpret_cast<const char*>(&Size),
//                       sizeof(std::size_t));
//             Ofs.write(Sfen.c_str(), (long)Size);
//
//             const uint32_t MoveValue = Entry.BestMove.value();
//             Ofs.write(reinterpret_cast<const char*>(&Entry.WinRate),
//                       sizeof(double));
//             Ofs.write(reinterpret_cast<const char*>(&Entry.DrawRate),
//                       sizeof(double));
//             Ofs.write(reinterpret_cast<const char*>(&MoveValue),
//                       sizeof(uint32_t));
//         }
//     } else if (OutputFormat == Format::YaneuraOu) {
//         for (const auto& [Sfen, Index] : Book.Dictionary) {
//             const auto& Entry = Book.Entries[Index];
//
//             Ofs << "sfen " << Sfen << std::endl;
//
//             Ofs << nshogi::io::sfen::move32ToSfen(Entry.BestMove) << " ";
//             Ofs << "none ";
//             Ofs << convertWinRateToScoreCP(Entry.WinRate) << " ";
//             Ofs << "1 1" << std::endl;
//         }
//     }
// }
//
// void load(engine::book::Book& Book, std::ifstream& Ifs) {
//     std::vector<char> Buffer(1024);
//
//     Book.Dictionary.clear();
//     Book.Entries.clear();
//
//     while (true) {
//         std::size_t Size = 0;
//         Ifs.read(reinterpret_cast<char*>(&Size), sizeof(std::size_t));
//
//         if (Ifs.eof()) {
//             break;
//         }
//
//         Ifs.read(Buffer.data(), (long)Size);
//
//         engine::book::BookEntry Entry;
//         uint32_t MoveValue = 0;
//         Ifs.read(reinterpret_cast<char*>(&Entry.WinRate), sizeof(double));
//         Ifs.read(reinterpret_cast<char*>(&Entry.DrawRate), sizeof(double));
//         Ifs.read(reinterpret_cast<char*>(&MoveValue), sizeof(uint32_t));
//         Entry.BestMove = core::Move32::fromValue(MoveValue);
//
//         Book.Entries.push_back(Entry);
//         const std::string Sfen = std::string(Buffer.data());
//         Book.Dictionary[Sfen] = Book.Entries.size() - 1;
//     }
// }

void save(const engine::book::BookMaker& Maker, std::ofstream& IndexOfs, std::ofstream& DataOfs) {
    const std::size_t NumIndices = Maker.NodeIndices.size();
    IndexOfs.write(reinterpret_cast<const char*>(&NumIndices),
                  sizeof(std::size_t));

    for (const auto& [Sfen, Index] : Maker.NodeIndices) {
        const std::size_t Size = Sfen.size() + 1;
        IndexOfs.write(reinterpret_cast<const char*>(&Size),
                      sizeof(std::size_t));
        IndexOfs.write(Sfen.c_str(), (long)Size);
        IndexOfs.write(reinterpret_cast<const char*>(&Index),
                      sizeof(engine::book::NodeIndex));
    }

    const std::size_t NumNodes = Maker.Nodes.size();
    DataOfs.write(reinterpret_cast<const char*>(&NumNodes),
                  sizeof(std::size_t));

    for (const auto& Node : Maker.Nodes) {
        DataOfs.write(reinterpret_cast<const char*>(&Node.Index),
                      sizeof(engine::book::NodeIndex));

        const std::size_t PolicySize = Node.PolicyRaw.size();
        DataOfs.write(reinterpret_cast<const char*>(&PolicySize),
                      sizeof(std::size_t));
        DataOfs.write(reinterpret_cast<const char*>(Node.PolicyRaw.data()),
                      (long)(sizeof(float) * PolicySize));

        DataOfs.write(reinterpret_cast<const char*>(&Node.WinRateRaw),
                      sizeof(float));
        DataOfs.write(reinterpret_cast<const char*>(&Node.DrawRateRaw),
                      sizeof(float));

        const std::size_t MovesSize = Node.Moves.size();
        DataOfs.write(reinterpret_cast<const char*>(&MovesSize),
                      sizeof(std::size_t));
        DataOfs.write(reinterpret_cast<const char*>(Node.Moves.data()),
                      (long)(sizeof(core::Move32) * MovesSize));

        const std::size_t VisitCountsSize = Node.VisitCounts.size();
        DataOfs.write(reinterpret_cast<const char*>(&VisitCountsSize),
                        sizeof(std::size_t));
        DataOfs.write(reinterpret_cast<const char*>(Node.VisitCounts.data()),
                        (long)(sizeof(uint64_t) * VisitCountsSize));
        const std::size_t WinRateAccumulatedsSize = Node.WinRateAccumulateds.size();
        DataOfs.write(reinterpret_cast<const char*>(&WinRateAccumulatedsSize),
                        sizeof(std::size_t));
        DataOfs.write(reinterpret_cast<const char*>(Node.WinRateAccumulateds.data()),
                        (long)(sizeof(double) * WinRateAccumulatedsSize));
        const std::size_t DrawRateAccumulatedsSize = Node.DrawRateAccumulateds.size();
        DataOfs.write(reinterpret_cast<const char*>(&DrawRateAccumulatedsSize),
                        sizeof(std::size_t));
        DataOfs.write(reinterpret_cast<const char*>(Node.DrawRateAccumulateds .data()),
                        (long)(sizeof(double) * DrawRateAccumulatedsSize));
        const std::size_t ChildrenSize = Node.Children.size();
        DataOfs.write(reinterpret_cast<const char*>(&ChildrenSize),
                        sizeof(std::size_t));
        DataOfs.write(reinterpret_cast<const char*>(Node.Children.data()),
                        (long)(sizeof(engine::book::NodeIndex) * ChildrenSize));
    }
}

void load(engine::book::BookMaker* Maker, std::ifstream& IndexIfs, std::ifstream& DataIfs) {
    std::vector<char> Buffer(10240);

    Maker->NodeIndices.clear();
    Maker->Nodes.clear();

    std::size_t NumIndices = 0;
    IndexIfs.read(reinterpret_cast<char*>(&NumIndices),
                    sizeof(std::size_t));
    for (std::size_t I = 0; I < NumIndices; ++I) {
        std::size_t Size = 0;
        IndexIfs.read(reinterpret_cast<char*>(&Size),
                        sizeof(std::size_t));
        IndexIfs.read(Buffer.data(), (long)Size);
        engine::book::NodeIndex Index;
        IndexIfs.read(reinterpret_cast<char*>(&Index),
                        sizeof(engine::book::NodeIndex));
        const std::string Sfen = std::string(Buffer.data());
        Maker->NodeIndices[Sfen] = Index;
    }

    std::size_t NumNodes = 0;
    DataIfs.read(reinterpret_cast<char*>(&NumNodes),
                    sizeof(std::size_t));
    for (std::size_t I = 0; I < NumNodes; ++I) {
        engine::book::Node Node(engine::book::NI_Null);
        DataIfs.read(reinterpret_cast<char*>(&Node.Index),
                      sizeof(engine::book::NodeIndex));
        std::size_t PolicySize = 0;
        DataIfs.read(reinterpret_cast<char*>(&PolicySize),
                        sizeof(std::size_t));
        Node.PolicyRaw.resize(PolicySize);
        DataIfs.read(reinterpret_cast<char*>(Node.PolicyRaw.data()),
                        (long)(sizeof(float) * PolicySize));
        DataIfs.read(reinterpret_cast<char*>(&Node.WinRateRaw),
                        sizeof(float));
        DataIfs.read(reinterpret_cast<char*>(&Node.DrawRateRaw),
                        sizeof(float));
        std::size_t MovesSize = 0;
        DataIfs.read(reinterpret_cast<char*>(&MovesSize),
                        sizeof(std::size_t));
        Node.Moves.resize(MovesSize);
        DataIfs.read(reinterpret_cast<char*>(Node.Moves.data()),
                        (long)(sizeof(core::Move32) * MovesSize));
        std::size_t VisitCountsSize = 0;
        DataIfs.read(reinterpret_cast<char*>(&VisitCountsSize),
                        sizeof(std::size_t));
        Node.VisitCounts.resize(VisitCountsSize);
        DataIfs.read(reinterpret_cast<char*>(Node.VisitCounts.data()),
                        (long)(sizeof(uint64_t) * VisitCountsSize));
        std::size_t WinRateAccumulatedsSize = 0;
        DataIfs.read(reinterpret_cast<char*>(&WinRateAccumulatedsSize),
                        sizeof(std::size_t));
        Node.WinRateAccumulateds.resize(WinRateAccumulatedsSize);
        DataIfs.read(reinterpret_cast<char*>(Node.WinRateAccumulateds.data()),
                        (long)(sizeof(double) * WinRateAccumulatedsSize));
        std::size_t DrawRateAccumulatedsSize = 0;
        DataIfs.read(reinterpret_cast<char*>(&DrawRateAccumulatedsSize),
                        sizeof(std::size_t));
        Node.DrawRateAccumulateds.resize(DrawRateAccumulatedsSize);
        DataIfs.read(reinterpret_cast<char*>(Node.DrawRateAccumulateds.data()),
                        (long)(sizeof(double) * DrawRateAccumulatedsSize));
        std::size_t ChildrenSize = 0;
        DataIfs.read(reinterpret_cast<char*>(&ChildrenSize),
                        sizeof(std::size_t));
        Node.Children.resize(ChildrenSize);
        DataIfs.read(reinterpret_cast<char*>(Node.Children.data()),
                        (long)(sizeof(engine::book::NodeIndex) * ChildrenSize));
        Maker->Nodes.push_back(Node);
    }
}

void save(const std::vector<engine::book::BookEntry>& Book, Format OutputFormat, std::ofstream& Ofs) {
    if (OutputFormat == Format::YaneuraOu) {
        Ofs << "#YANEURAOU-DB2016 1.00" << std::endl;

        for (const auto& Entry : Book) {
            Ofs << "sfen " << Entry.Sfen << std::endl;

            Ofs << nshogi::io::sfen::move32ToSfen(Entry.Move) << " ";
            Ofs << "none ";
            const double WinRate = std::clamp(static_cast<double>(Entry.WinRate), 0.1, 0.9);
            Ofs << (int)(-600.0 * std::log(1.0 / WinRate - 1.0)) << " ";
            Ofs << "1 1" << std::endl;
        }
    }
}

} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi
