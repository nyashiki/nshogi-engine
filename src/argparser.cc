#include "argparser.h"

#include <iostream>
#include <stdexcept>

namespace nshogi {
namespace engine {

ArgParser::ArgParser() {
    addOption("help", "", "Show this help.");
}

void ArgParser::addOption(char NameShort, const char* NameLong, const char* DefaultValue, const char* Description) {
    NamedOptions.emplace(NameLong, DefaultValue);
    Descriptions.emplace(NameLong, Description);

    if (NameShort != 0) {
        MapShortToLong.emplace(NameShort, NameLong);
    }
}

void ArgParser::addOption(const char* NameLong, const char* DefaultValue, const char* Description) {
    addOption(0, NameLong,  DefaultValue, Description);
}

void ArgParser::addOption(char NameShort, const char* NameLong, const char* Description) {
    addOption(NameShort, NameLong, "", Description);
}

void ArgParser::addOption(const char* NameLong, const char* Description) {
    addOption(0, NameLong, "", Description);
}

const std::string& ArgParser::getOption(char NameShort) const {
    using std::literals::string_literals::operator""s;

    if (!exists(NameShort)) {
        throw std::runtime_error(
                "Unknown option `"s + NameShort + "` is given."s);
    }

    return NamedOptions.at(MapShortToLong.at(NameShort));
}

const std::string& ArgParser::getOption(const char* NameLong) const {
    using std::literals::string_literals::operator""s;

    if (!exists(NameLong)) {
        throw std::runtime_error(
                "Unknown option `"s + NameLong + "` is given."s);
    }

    return NamedOptions.at(NameLong);
}

const std::vector<std::string>& ArgParser::getUnnamedOptions() const {
    return UnnamedOptions;
}

void ArgParser::parse(int Argc, char* Argv[], bool AllowUnknownOption) {
    using std::literals::string_literals::operator""s;

    std::string OptionName = "";

    for (int I = 1; I < Argc; ++I) {
        const auto Element = std::string(Argv[I]);

        if (Element.size() >= 3 &&
                Element[0] == '-' &&
                Element[1] == '-' &&
                Element[2] == '-') {

            throw std::runtime_error(
                    "option starts with more than or equal to three hyphenations.");
        } else if (Element.size() >= 2 &&
                Element[0] == '-' && Element[1] == '-') {
            if (Element.size() == 2) {
                throw std::runtime_error(
                        "Given two hyphenations but no name is specified.");
            }

            if (OptionName != "") {
                Specifieds.emplace(OptionName);
            }

            OptionName = Element.substr(2);

            if (!AllowUnknownOption && !exists(OptionName.data())) {
                throw std::runtime_error(
                        "Unknown option `"s + OptionName + "` is given."s);
            }

            continue;
        } else if (Element.size() >= 1 && Element[0] == '-') {
            if (Element.size() == 1) {
                throw std::runtime_error(
                        "Given a hyphenation but no name is specified.");
            }
            if (Element.size() >= 3) {
                throw std::runtime_error(
                        "Only one character can be accepted for short name options.");
            }

            if (!exists(Element[1])) {
                throw std::runtime_error(
                        "Unknown option `"s + Element[1] + "` is given."s);
            }

            if (OptionName != "") {
                Specifieds.emplace(OptionName);
            }

            OptionName = MapShortToLong.at(Element[1]);
            continue;
        }

        if (OptionName == "") {
            UnnamedOptions.emplace_back(Element);
        } else {
            auto it = NamedOptions.find(OptionName);
            if (it != NamedOptions.end()) {
                it->second = Element;
            } else {
                NamedOptions.emplace(OptionName, Element);
            }

            Specifieds.emplace(OptionName);
            OptionName = "";
        }
    }

    if (OptionName != "") {
        Specifieds.emplace(OptionName);
    }
}

bool ArgParser::exists(char ShortName) const {
    return MapShortToLong.find(ShortName) != MapShortToLong.end();
}

bool ArgParser::exists(const char* LongName) const {
    return NamedOptions.find(LongName) != NamedOptions.end();
}

bool ArgParser::isSpecified(char ShortName) const {
    return Specifieds.contains(MapShortToLong.at(ShortName));
}

bool ArgParser::isSpecified(const char* LongName) const {
    return Specifieds.contains(LongName);
}

void ArgParser::showHelp() const {
    std::cout << "OPTIONS:" << std::endl;

    std::unordered_set<std::string> Shown;

    for (const auto& [k, v] : MapShortToLong) {
        const auto& Description = Descriptions.at(v);

        std::cout << "\t-" << k << ", --" << v << ": " <<
            Description << std::endl;

        std::cout << std::endl;

        Shown.emplace(v);
    }

    for (const auto& [v, d] : NamedOptions) {
        if (Shown.contains(v)) {
            continue;
        }

        const auto& Description = Descriptions.at(v);
        std::cout << "\t--" << v << ": " <<
            Description << std::endl;
        std::cout << std::endl;

    }
}

} // namespace engine
} // namespace nshogi
