#include "common/json_line_parser.hpp"

#include <cstdlib>
#include <stdexcept>

namespace colocation {

char JsonLineParser::peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

char JsonLineParser::next() {
    if (pos_ >= text_.size()) throw std::runtime_error("json: unexpected end of input");
    return text_[pos_++];
}

void JsonLineParser::expect(char c) {
    if (next() != c) throw std::runtime_error(std::string("json: expected ") + c);
}

void JsonLineParser::skipWhitespace() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' || text_[pos_] == '\r')) ++pos_;
}

void JsonLineParser::appendUtf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

unsigned JsonLineParser::readHex4() {
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = next();
        value <<= 4;
        if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
        else throw std::runtime_error("json: bad \\u escape");
    }
    return value;
}

std::string JsonLineParser::readString() {
    expect('"');
    std::string out;
    while (true) {
        const char c = next();
        if (c == '"') break;
        if (c != '\\') {
            out += c;
            continue;
        }
        const char e = next();
        switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                unsigned cp = readHex4();
                const bool highSurrogate = cp >= 0xD800 && cp < 0xDC00;
                if (highSurrogate && peek() == '\\') {
                    const std::size_t save = pos_;
                    ++pos_;
                    if (next() == 'u') {
                        const unsigned low = readHex4();
                        if (low >= 0xDC00 && low < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        else pos_ = save;
                    } else {
                        pos_ = save;
                    }
                }
                appendUtf8(out, cp);
                break;
            }
            default: throw std::runtime_error("json: bad escape");
        }
    }
    return out;
}

void JsonLineParser::skipNested() {
    int depth = 0;
    do {
        const char c = next();
        if (c == '"') {
            --pos_;
            readString();
            continue;
        }
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
    } while (depth > 0);
}

JsonValue JsonLineParser::readValue() {
    JsonValue v;
    const char c = peek();
    if (c == '"') {
        v.kind = JsonValue::Kind::String;
        v.text = readString();
    } else if (c == '{' || c == '[') {
        const std::size_t start = pos_;
        skipNested();
        v.kind = JsonValue::Kind::Raw;
        v.text = text_.substr(start, pos_ - start);
    } else if (text_.compare(pos_, 4, "null") == 0) {
        pos_ += 4;
    } else if (text_.compare(pos_, 4, "true") == 0) {
        pos_ += 4;
        v.kind = JsonValue::Kind::Bool;
        v.boolean = true;
    } else if (text_.compare(pos_, 5, "false") == 0) {
        pos_ += 5;
        v.kind = JsonValue::Kind::Bool;
    } else {
        const char* start = text_.c_str() + pos_;
        char* end = nullptr;
        v.number = std::strtod(start, &end);
        if (end == start) throw std::runtime_error("json: bad value");
        pos_ += static_cast<std::size_t>(end - start);
        v.kind = JsonValue::Kind::Number;
    }
    return v;
}

JsonObject JsonLineParser::parseObject() {
    JsonObject out;
    skipWhitespace();
    expect('{');
    skipWhitespace();
    if (peek() == '}') {
        ++pos_;
        return out;
    }
    while (true) {
        skipWhitespace();
        const std::string key = readString();
        skipWhitespace();
        expect(':');
        skipWhitespace();
        out[key] = readValue();
        skipWhitespace();
        const char c = next();
        if (c == '}') break;
        if (c != ',') throw std::runtime_error("json: expected , or }");
    }
    return out;
}

}  // namespace colocation
