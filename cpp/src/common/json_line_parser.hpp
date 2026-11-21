// Parser for files with one JSON object per line (the Yelp files). Top-level strings, numbers, booleans and null are
// decoded; nested objects and arrays are kept as raw text, which can be parsed again with JsonLineParser.
#pragma once

#include <map>
#include <string>

namespace colocation {

struct JsonValue {
    enum class Kind { Null, Bool, Number, String, Raw };
    Kind kind = Kind::Null;
    double number = 0.0;
    bool boolean = false;
    std::string text;   // the string value, or the raw text of a nested object or array

    bool isString() const { return kind == Kind::String; }
    bool isNumber() const { return kind == Kind::Number; }
};

using JsonObject = std::map<std::string, JsonValue>;

class JsonLineParser {
public:
    explicit JsonLineParser(const std::string& text) : text_(text) {}
    JsonObject parseObject();

private:
    const std::string& text_;
    std::size_t pos_ = 0;

    char peek() const;
    char next();
    void expect(char c);
    void skipWhitespace();
    unsigned readHex4();
    std::string readString();
    void skipNested();
    JsonValue readValue();
    static void appendUtf8(std::string& out, unsigned codePoint);
};

}  // namespace colocation
