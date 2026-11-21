// Minimal parser for one JSON object per line (Yelp files): top-level strings, numbers, booleans and null are kept,
// nested objects and arrays are kept as raw text (an object of numbers can be read again with parse_object).
#pragma once
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

namespace uhucpm {

struct JVal {
    enum Kind { Null, Bool, Num, Str, Raw } kind = Null;
    double num = 0;
    bool b = false;
    std::string str;   // string value, or raw text of a nested object/array
};

class JsonLine {
public:
    explicit JsonLine(const std::string& s) : s_(s) {}
    std::map<std::string, JVal> parse_object() {
        std::map<std::string, JVal> out;
        ws();
        expect('{');
        ws();
        if (peek() == '}') { ++p_; return out; }
        while (true) {
            ws();
            std::string key = str();
            ws();
            expect(':');
            ws();
            out[key] = value();
            ws();
            char c = get();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("json: expected , or }");
        }
        return out;
    }

private:
    const std::string& s_;
    size_t p_ = 0;
    char peek() const { return p_ < s_.size() ? s_[p_] : '\0'; }
    char get() { if (p_ >= s_.size()) throw std::runtime_error("json: end of input"); return s_[p_++]; }
    void expect(char c) { if (get() != c) throw std::runtime_error(std::string("json: expected ") + c); }
    void ws() { while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) ++p_; }
    static void utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    }
    unsigned hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = get();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else throw std::runtime_error("json: bad \\u escape");
        }
        return v;
    }
    std::string str() {
        expect('"');
        std::string o;
        while (true) {
            char c = get();
            if (c == '"') break;
            if (c != '\\') { o += c; continue; }
            char e = get();
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && peek() == '\\') {
                        size_t save = p_;
                        ++p_;
                        if (get() == 'u') {
                            unsigned lo = hex4();
                            if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            else p_ = save;
                        } else p_ = save;
                    }
                    utf8(o, cp);
                    break;
                }
                default: throw std::runtime_error("json: bad escape");
            }
        }
        return o;
    }
    void skip_nested() {   // object or array, strings respected
        int depth = 0;
        do {
            char c = get();
            if (c == '"') { --p_; str(); continue; }
            if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') --depth;
        } while (depth > 0);
    }
    JVal value() {
        JVal v;
        char c = peek();
        if (c == '"') { v.kind = JVal::Str; v.str = str(); }
        else if (c == '{' || c == '[') { size_t a = p_; skip_nested(); v.kind = JVal::Raw; v.str = s_.substr(a, p_ - a); }
        else if (s_.compare(p_, 4, "null") == 0) { p_ += 4; v.kind = JVal::Null; }
        else if (s_.compare(p_, 4, "true") == 0) { p_ += 4; v.kind = JVal::Bool; v.b = true; }
        else if (s_.compare(p_, 5, "false") == 0) { p_ += 5; v.kind = JVal::Bool; v.b = false; }
        else {
            const char* st = s_.c_str() + p_;
            char* en = nullptr;
            v.num = std::strtod(st, &en);
            if (en == st) throw std::runtime_error("json: bad value");
            p_ += (size_t)(en - st);
            v.kind = JVal::Num;
        }
        return v;
    }
};

}  // namespace uhucpm
