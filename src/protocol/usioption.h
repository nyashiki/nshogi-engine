#ifndef NSHOGI_ENGINE_PROTOCOL_USIOPTION_H
#define NSHOGI_ENGINE_PROTOCOL_USIOPTION_H

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <cinttypes>
#include <cstring>

namespace nshogi {
namespace engine {
namespace protocol {

class USIOption {
 public:
    bool addOption(const char* Key, bool DefaultValue) {
        if (exists(Key)) {
            return false;
        }

        Items.push_back(std::make_unique<BoolItem>(Key, DefaultValue));
        return true;
    }

    bool addIntOption(const char* Key, int64_t DefaultValue, int64_t Min, int64_t Max) {
        if (exists(Key)) {
            return false;
        }

        Items.push_back(std::make_unique<IntItem>(Key, DefaultValue, Min, Max));
        return true;
    }

    bool addBoolOption(const char* Key, bool DefaultValue) {
        if (exists(Key)) {
            return false;
        }

        Items.push_back(std::make_unique<BoolItem>(Key, DefaultValue));
        return true;
    }

    bool addStringOption(const char* Key, const char* DefaultValue) {
        if (exists(Key)) {
            return false;
        }

        Items.push_back(std::make_unique<StringItem>(Key, DefaultValue));
        return true;
    }

    bool addFileNameOption(const char* Key, const char* DefaultValue) {
        if (exists(Key)) {
            return false;
        }

        Items.push_back(std::make_unique<FileNameItem>(Key, DefaultValue));
        return true;
    }

    int64_t getIntOption(const char* Key) const {
        const void* Value = getOption(Key);
        return *((const int64_t*)(Value));
    }

    bool getBoolOption(const char* Key) const {
        const void* Value = getOption(Key);
        return *((const bool*)(Value));
    }

    const char* getStringOption(const char* Key) const {
        const void* Value = getOption(Key);
        return (const char*)Value;
    }

    const char* getFileNameOption(const char* Key) const {
        const void* Value = getOption(Key);
        return (const char*)Value;
    }

    bool setOptionValue(const char* Key, const char* Value) {
        for (auto& I : Items) {
            if (I->Key == Key) {
                I->set(Value);
                return true;
            }
        }

        return false;
    }

    void showOption() const {
        for (const auto& I : Items) {
            I->print();
        }
    }

 private:
    struct Item {
     public:
        Item(const std::string& Key_)
            : Key(Key_) {
        }

        virtual ~Item() {
        }

        virtual void print() const = 0;
        virtual void set(const char* Value) = 0;
        virtual void* get() const = 0;

        std::string Key;
    };

    struct BoolItem : public Item {
     public:
        bool Value;

        BoolItem(const std::string& Key_, bool Value_)
            : Item(Key_)
            , Value(Value_) {
        }

        void print() const {
            std::cout << "option name " << Key << " type check default " << (Value ? "true" : "false") << std::endl;
        }

        void set(const char* Value_) {
            if (std::strcmp(Value_, "0") == 0 ||
                    std::strcmp(Value_, "false") == 0) {
                Value = false;
            } else {
                Value = true;
            }
        }

        void* get() const {
            return (void*)&Value;
        }
    };

    struct FileNameItem : public Item {
     public:
        std::string Value;

        FileNameItem(const std::string& Key_, const std::string& Value_)
            : Item(Key_)
            , Value(Value_) {
        }

        void print() const {
            std::cout << "option name " << Key << " type filename default " << Value << std::endl;
        }

        void set(const char* Value_) {
            Value = Value_;
        }

        void* get() const {
            return (void*)Value.c_str();
        }
    };

    struct StringItem : public Item {
     public:
        std::string Value;

        StringItem(const std::string& Key_, const std::string& Value_)
            : Item(Key_)
            , Value(Value_) {
        }

        void print() const {
            std::cout << "option name " << Key << " type string default " << Value << std::endl;
        }

        void set(const char* Value_) {
            Value = Value_;
        }

        void* get() const {
            return (void*)Value.c_str();
        }
    };

    struct IntItem : public Item {
     public:
        int64_t Value;
        int64_t Min, Max;

        IntItem(const std::string& Key_, int64_t Value_, int64_t Min_, int64_t Max_)
            : Item(Key_)
            , Value(Value_)
            , Min(Min_)
            , Max(Max_) {
        }

        void print() const {
            std::cout << "option name " << Key << " type spin default " << Value <<
                " min " << Min << " max " << Max << std::endl;
        }

        void set(const char* Value_) {
            Value = std::stoll(Value_);
        }

        void* get() const {
            return (void*)&Value;
        }
    };

    void* getOption(const std::string& Key) const {
        for (const auto& I : Items) {
            if (I->Key == Key) {
                return I->get();
            }
        }

        throw std::runtime_error("Option item not found.");
    }

    bool exists(const std::string& Key) const {
        for (const auto& I : Items) {
            if (I->Key == Key) {
                return true;
            }
        }

        return false;
    }

    std::vector<std::unique_ptr<Item>> Items;
};

} // namespace protocol
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_PROTOCOL_USIOPTION_H
