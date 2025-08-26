#include "pch.h"
#include "ConfigParser.h"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <iostream>

ModelConfig::ModelConfig() {
    setDefaults();
}

void ModelConfig::setDefaults() {
    num_classes = 3;
    num_input_channels = 1;
    
    patch_size.clear();
    patch_size.push_back(128);
    patch_size.push_back(128);
    patch_size.push_back(128);
    
    target_spacing.clear();
    target_spacing.push_back(1.0f);
    target_spacing.push_back(1.0f);
    target_spacing.push_back(1.0f);
    
    transpose_forward.clear();
    transpose_forward.push_back(0);
    transpose_forward.push_back(1);
    transpose_forward.push_back(2);
    
    transpose_backward.clear();
    transpose_backward.push_back(0);
    transpose_backward.push_back(1);
    transpose_backward.push_back(2);
    
    mean = 0.0f;
    std = 1.0f;
    min_val = 0.0f;
    max_val = 1.0f;
    percentile_00_5 = 0.0f;
    percentile_99_5 = 1.0f;
    normalization_scheme = "CTNormalization";
    use_mask_for_norm = false;
    use_tta = false;
}

ConfigParser::ConfigParser() {
}

ConfigParser::~ConfigParser() {
}

const ModelConfig& ConfigParser::getConfig() const {
    return currentConfig;
}

bool ConfigParser::parseJsonString(const std::string& jsonContent, ModelConfig& config) {
    try {
        currentConfig = ModelConfig();
        
        parseIntValue(jsonContent, "num_classes", config.num_classes);
        parseIntValue(jsonContent, "num_input_channels", config.num_input_channels);
        parseIntArray(jsonContent, "patch_size", config.patch_size);
        parseFloatArray(jsonContent, "target_spacing", config.target_spacing);
        parseIntArray(jsonContent, "transpose_forward", config.transpose_forward);
        parseIntArray(jsonContent, "transpose_backward", config.transpose_backward);
        parseIntensityProperties(jsonContent, config);
        
        size_t pos = jsonContent.find("\"normalization_schemes\":");
        if (pos != std::string::npos) {
            pos = jsonContent.find("[", pos);
            size_t end = jsonContent.find("]", pos);
            std::string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            size_t quoteStart = arrayStr.find("\"");
            if (quoteStart != std::string::npos) {
                size_t quoteEnd = arrayStr.find("\"", quoteStart + 1);
                config.normalization_scheme = arrayStr.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
            }
        }
        
        parseBoolValue(jsonContent, "use_tta", config.use_tta);
        
        // 解析 use_mask_for_norm（注意：它是一个数组）
        size_t maskPos = jsonContent.find("\"use_mask_for_norm\":");
        if (maskPos != std::string::npos) {
            size_t arrayStart = jsonContent.find("[", maskPos);
            size_t arrayEnd = jsonContent.find("]", arrayStart);
            if (arrayStart != std::string::npos && arrayEnd != std::string::npos) {
                std::string arrayContent = jsonContent.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
                // 移除空白字符
                arrayContent.erase(std::remove_if(arrayContent.begin(), arrayContent.end(), ::isspace), arrayContent.end());
                // 如果数组包含 "true"，则设置为 true
                config.use_mask_for_norm = (arrayContent.find("true") != std::string::npos);
                std::cout << "[DEBUG] Parsed use_mask_for_norm: " << (config.use_mask_for_norm ? "true" : "false") << std::endl;
            }
        }
        
        currentConfig = config;
        
        return true;
        
    } catch (...) {
        return false;
    }
}

bool ConfigParser::parseIntValue(const std::string& jsonContent, const std::string& key, int& value) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos != std::string::npos) {
        pos = jsonContent.find(":", pos) + 1;
        size_t end = jsonContent.find(",", pos);
        if (end == std::string::npos) end = jsonContent.find("}", pos);
        std::string valueStr = jsonContent.substr(pos, end - pos);
        valueStr.erase(std::remove_if(valueStr.begin(), valueStr.end(), ::isspace), valueStr.end());
        try {
            value = std::stoi(valueStr);
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool ConfigParser::parseFloatValue(const std::string& jsonContent, const std::string& key, float& value) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos != std::string::npos) {
        pos = jsonContent.find(":", pos) + 1;
        size_t end = jsonContent.find(",", pos);
        if (end == std::string::npos) end = jsonContent.find("}", pos);
        std::string valueStr = jsonContent.substr(pos, end - pos);
        valueStr.erase(std::remove_if(valueStr.begin(), valueStr.end(), ::isspace), valueStr.end());
        try {
            value = std::stof(valueStr);
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool ConfigParser::parseStringValue(const std::string& jsonContent, const std::string& key, std::string& value) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos != std::string::npos) {
        pos = jsonContent.find("\"", pos + searchKey.length());
        if (pos != std::string::npos) {
            size_t end = jsonContent.find("\"", pos + 1);
            if (end != std::string::npos) {
                value = jsonContent.substr(pos + 1, end - pos - 1);
                return true;
            }
        }
    }
    return false;
}

bool ConfigParser::parseBoolValue(const std::string& jsonContent, const std::string& key, bool& value) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos != std::string::npos) {
        pos = jsonContent.find(":", pos) + 1;
        size_t end = jsonContent.find(",", pos);
        if (end == std::string::npos) end = jsonContent.find("}", pos);
        std::string valueStr = jsonContent.substr(pos, end - pos);
        valueStr.erase(std::remove_if(valueStr.begin(), valueStr.end(), ::isspace), valueStr.end());
        value = (valueStr == "true");
        return true;
    }
    return false;
}

bool ConfigParser::parseIntArray(const std::string& jsonContent, const std::string& key, std::vector<int>& array) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos != std::string::npos) {
        pos = jsonContent.find("[", pos);
        size_t end = jsonContent.find("]", pos);
        if (pos != std::string::npos && end != std::string::npos) {
            std::string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            std::stringstream ss(arrayStr);
            std::string item;
            array.clear();
            while (std::getline(ss, item, ',')) {
                item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
                if (!item.empty()) {
                    try {
                        array.push_back(std::stoi(item));
                    } catch (...) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
    return false;
}

bool ConfigParser::parseFloatArray(const std::string& jsonContent, const std::string& key, std::vector<float>& array) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos != std::string::npos) {
        pos = jsonContent.find("[", pos);
        size_t end = jsonContent.find("]", pos);
        if (pos != std::string::npos && end != std::string::npos) {
            std::string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            std::stringstream ss(arrayStr);
            std::string item;
            array.clear();
            while (std::getline(ss, item, ',')) {
                item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
                if (!item.empty()) {
                    try {
                        array.push_back(std::stof(item));
                    } catch (...) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
    return false;
}

bool ConfigParser::parseIntensityProperties(const std::string& jsonContent, ModelConfig& config) {
    size_t pos = jsonContent.find("\"intensity_properties\":");
    if (pos != std::string::npos) {
        pos = jsonContent.find("\"0\":", pos);
        if (pos != std::string::npos) {
            size_t objStart = jsonContent.find("{", pos);
            size_t objEnd = jsonContent.find("}", objStart);
            if (objStart != std::string::npos && objEnd != std::string::npos) {
                std::string objStr = jsonContent.substr(objStart, objEnd - objStart + 1);
                
                parseFloatValue(objStr, "mean", config.mean);
                parseFloatValue(objStr, "std", config.std);
                parseFloatValue(objStr, "min", config.min_val);
                parseFloatValue(objStr, "max", config.max_val);
                parseFloatValue(objStr, "percentile_00_5", config.percentile_00_5);
                parseFloatValue(objStr, "percentile_99_5", config.percentile_99_5);
                
                return true;
            }
        }
    }
    return false;
}