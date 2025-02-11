//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_ARGPARSER_H
#define NSHOGI_ENGINE_ARGPARSER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nshogi {
namespace engine {

class ArgParser {
 public:
    ArgParser();

    void addOption(char NameShort, const char* NameLong, const char* DefaultValue, const char* Description);
    void addOption(const char* NameLong, const char* DefaultValue, const char* Description);
    void addOption(char NameShort, const char* NameLong, const char* Description);
    void addOption(const char* NameLong, const char* Description);

    const std::string& getOption(char NameShort) const;
    const std::string& getOption(const char* NameLong) const;

    const std::vector<std::string>& getUnnamedOptions() const;

    void parse(int Argc, char* Argv[], bool AllowUnknownOption = false);

    void showHelp() const;

    bool isSpecified(char ShortName) const;
    bool isSpecified(const char* LongName) const;

 private:
    bool exists(char ShortName) const;
    bool exists(const char* LongName) const;

    std::unordered_map<char, std::string> MapShortToLong;
    std::unordered_map<std::string, std::string> NamedOptions;
    std::vector<std::string> UnnamedOptions;
    std::unordered_map<std::string, std::string> Descriptions;

    std::unordered_set<std::string> Specifieds;
};

} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ARGPARSER_H
