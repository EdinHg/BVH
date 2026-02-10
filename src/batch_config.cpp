#include "../include/batch_config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

// Simple JSON parser for our specific config format
// No external dependencies - hand-written parser

class SimpleJsonParser {
private:
    std::string content;
    size_t pos;
    
    void skipWhitespace() {
        while (pos < content.length() && std::isspace(content[pos])) {
            ++pos;
        }
    }
    
    char peek() {
        skipWhitespace();
        return pos < content.length() ? content[pos] : '\0';
    }
    
    char consume() {
        char c = peek();
        ++pos;
        return c;
    }
    
    std::string parseString() {
        skipWhitespace();
        if (consume() != '"') throw std::runtime_error("Expected '\"' at position " + std::to_string(pos));
        
        std::string result;
        while (pos < content.length() && content[pos] != '"') {
            if (content[pos] == '\\' && pos + 1 < content.length()) {
                ++pos;
                switch (content[pos]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: result += content[pos];
                }
            } else {
                result += content[pos];
            }
            ++pos;
        }
        
        if (consume() != '"') throw std::runtime_error("Expected closing '\"' at position " + std::to_string(pos));
        return result;
    }
    
    int parseNumber() {
        skipWhitespace();
        std::string numStr;
        if (peek() == '-') numStr += consume();
        while (pos < content.length() && std::isdigit(content[pos])) {
            numStr += consume();
        }
        return std::stoi(numStr);
    }
    
    float parseFloat() {
        skipWhitespace();
        std::string numStr;
        if (peek() == '-') numStr += consume();
        while (pos < content.length() && (std::isdigit(content[pos]) || content[pos] == '.')) {
            numStr += consume();
        }
        return std::stof(numStr);
    }
    
    bool parseBoolean() {
        skipWhitespace();
        if (content.substr(pos, 4) == "true") {
            pos += 4;
            return true;
        } else if (content.substr(pos, 5) == "false") {
            pos += 5;
            return false;
        }
        throw std::runtime_error("Expected boolean at position " + std::to_string(pos));
    }

public:
    SimpleJsonParser(const std::string& json) : content(json), pos(0) {}
    
    std::string getString(const std::string& key) {
        size_t keyPos = content.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return "";
        
        pos = keyPos + key.length() + 2;
        skipWhitespace();
        if (consume() != ':') return "";
        
        return parseString();
    }
    
    int getInt(const std::string& key, int defaultVal = 0) {
        size_t keyPos = content.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return defaultVal;
        
        pos = keyPos + key.length() + 2;
        skipWhitespace();
        if (consume() != ':') return defaultVal;
        
        try {
            return parseNumber();
        } catch (...) {
            return defaultVal;
        }
    }
    
    float getFloat(const std::string& key, float defaultVal = 0.0f) {
        size_t keyPos = content.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return defaultVal;
        
        pos = keyPos + key.length() + 2;
        skipWhitespace();
        if (consume() != ':') return defaultVal;
        
        try {
            return parseFloat();
        } catch (...) {
            return defaultVal;
        }
    }
    
    bool getBoolean(const std::string& key, bool defaultVal = false) {
        size_t keyPos = content.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return defaultVal;
        
        pos = keyPos + key.length() + 2;
        skipWhitespace();
        if (consume() != ':') return defaultVal;
        
        try {
            return parseBoolean();
        } catch (...) {
            return defaultVal;
        }
    }
    
    std::vector<std::string> getStringArray(const std::string& key) {
        std::vector<std::string> result;
        size_t keyPos = content.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return result;
        
        pos = keyPos + key.length() + 2;
        skipWhitespace();
        if (consume() != ':') return result;
        skipWhitespace();
        if (consume() != '[') return result;
        
        while (peek() != ']' && pos < content.length()) {
            skipWhitespace();
            if (peek() == '"') {
                result.push_back(parseString());
            }
            skipWhitespace();
            if (peek() == ',') consume();
        }
        
        if (consume() != ']') return result;
        return result;
    }
    
    std::vector<int> getIntArray(const std::string& key) {
        std::vector<int> result;
        size_t keyPos = content.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return result;
        
        pos = keyPos + key.length() + 2;
        skipWhitespace();
        if (consume() != ':') return result;
        skipWhitespace();
        if (consume() != '[') return result;
        
        while (peek() != ']' && pos < content.length()) {
            skipWhitespace();
            if (std::isdigit(peek()) || peek() == '-') {
                result.push_back(parseNumber());
            }
            skipWhitespace();
            if (peek() == ',') consume();
        }
        
        if (consume() != ']') return result;
        return result;
    }
    
    std::vector<std::string> getObjectArray(const std::string& key) {
        std::vector<std::string> result;
        size_t keyPos = content.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return result;
        
        pos = keyPos + key.length() + 2;
        skipWhitespace();
        if (consume() != ':') return result;
        skipWhitespace();
        if (consume() != '[') return result;
        
        while (peek() != ']' && pos < content.length()) {
            skipWhitespace();
            if (peek() == '{') {
                int braceCount = 0;
                size_t startPos = pos;
                do {
                    if (content[pos] == '{') ++braceCount;
                    else if (content[pos] == '}') --braceCount;
                    ++pos;
                } while (braceCount > 0 && pos < content.length());
                
                result.push_back(content.substr(startPos, pos - startPos));
            }
            skipWhitespace();
            if (peek() == ',') consume();
        }
        
        if (consume() != ']') return result;
        return result;
    }
};

BatchConfig loadBatchConfig(const std::string& filename) {
    BatchConfig config;
    
    // Read file
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonContent = buffer.str();
    file.close();
    
    // Parse JSON
    SimpleJsonParser parser(jsonContent);
    
    // Load simple fields
    config.iterations = parser.getInt("iterations", 5);
    config.warmup = parser.getBoolean("warmup", true);
    config.outputFile = parser.getString("output");
    config.quiet = parser.getBoolean("quiet", false);
    
    // Load algorithms
    std::vector<std::string> algoList = parser.getStringArray("algorithms");
    if (!algoList.empty()) {
        config.algorithms = algoList;
    } else {
        config.algorithms.push_back("all");
    }
    
    // Load PLOC radius values
    config.plocRadius = parser.getIntArray("ploc_radius");
    if (config.plocRadius.empty()) {
        config.plocRadius = {10, 25, 100};
    }
    
    // Load render configuration
    if (jsonContent.find("\"render\"") != std::string::npos) {
        config.render.enabled = parser.getBoolean("render_enabled", false);
        config.render.prefix = parser.getString("render_prefix");
        config.render.size = parser.getString("render_size");
        config.render.shading = parser.getString("render_shading");
        config.render.camera = parser.getString("render_camera");
        config.render.cameraUp = parser.getString("render_camera_up");
        config.render.fov = parser.getFloat("render_fov", 0.0f);
    }
    
    // Load models
    std::vector<std::string> modelObjStrs = parser.getObjectArray("models");
    for (const auto& objStr : modelObjStrs) {
        SimpleJsonParser modelParser(objStr);
        
        ModelConfig model;
        model.type = modelParser.getString("type");
        model.path = modelParser.getString("path");
        model.triangles = modelParser.getInt("triangles");
        model.name = modelParser.getString("name");
        
        // Generate name if not provided
        if (model.name.empty()) {
            if (model.type == "obj") {
                model.name = model.path;
                // Extract just the filename
                size_t lastSlash = model.name.find_last_of("/\\");
                if (lastSlash != std::string::npos) {
                    model.name = model.name.substr(lastSlash + 1);
                }
            } else if (model.type == "random") {
                if (model.triangles >= 1000000) {
                    model.name = "random_" + std::to_string(model.triangles / 1000000) + "M";
                } else if (model.triangles >= 1000) {
                    model.name = "random_" + std::to_string(model.triangles / 1000) + "K";
                } else {
                    model.name = "random_" + std::to_string(model.triangles);
                }
            }
        }
        
        if (!model.type.empty()) {
            config.models.push_back(model);
        }
    }
    
    return config;
}

bool validateBatchConfig(const BatchConfig& config, std::string& errorMsg) {
    if (config.iterations < 1) {
        errorMsg = "iterations must be >= 1";
        return false;
    }
    
    if (config.algorithms.empty()) {
        errorMsg = "No algorithms specified";
        return false;
    }
    
    if (config.models.empty()) {
        errorMsg = "No models specified";
        return false;
    }
    
    for (const auto& model : config.models) {
        if (model.type != "obj" && model.type != "random") {
            errorMsg = "Invalid model type: " + model.type + " (expected 'obj' or 'random')";
            return false;
        }
        
        if (model.type == "obj" && model.path.empty()) {
            errorMsg = "OBJ model must have 'path' field";
            return false;
        }
        
        if (model.type == "random" && model.triangles <= 0) {
            errorMsg = "Random model must have triangles > 0";
            return false;
        }
    }
    
    for (const auto& algo : config.algorithms) {
        if (algo != "lbvh" && algo != "lbvh+" && algo != "ploc" && algo != "all") {
            errorMsg = "Invalid algorithm: " + algo + " (expected 'lbvh', 'lbvh+', 'ploc', or 'all')";
            return false;
        }
    }
    
    if (config.plocRadius.empty()) {
        errorMsg = "ploc_radius list is empty";
        return false;
    }
    
    for (int r : config.plocRadius) {
        if (r <= 0) {
            errorMsg = "PLOC radius must be > 0";
            return false;
        }
    }
    
    if (config.render.enabled && config.render.prefix.empty()) {
        errorMsg = "render_prefix must be specified if rendering is enabled";
        return false;
    }
    
    if (config.outputFile.empty()) {
        errorMsg = "output file path must be specified";
        return false;
    }
    
    return true;
}
