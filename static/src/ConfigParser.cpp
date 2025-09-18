#include "pch.h"
#include "ConfigParser.h"
#include "UnetMain.h"  // 需要包含以获取nnUNetConfig定义
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
    
    mean.assign(1, 0.0f);
    std.assign(1, 1.0f);
    min_val.assign(1, 0.0f);
    max_val.assign(1, 1.0f);
    percentile_00_5.assign(1, 0.0f);
    percentile_99_5.assign(1, 1.0f);
    
    normalization_scheme.assign(1, "CTNormalization");
    use_mask_for_norm.assign(1, false);
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
        parseStringArray(jsonContent, "normalization_schemes", config.normalization_scheme);
        parseBoolArray(jsonContent, "use_mask_for_norm", config.use_mask_for_norm);

        parseBoolValue(jsonContent, "use_tta", config.use_tta);
        
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

// 新增函数：解析字符串数组
bool ConfigParser::parseStringArray(const std::string& jsonContent, const std::string& key, std::vector<std::string>& array) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos == std::string::npos) return false;

    pos = jsonContent.find("[", pos);
    size_t end = jsonContent.find("]", pos);
    if (pos == std::string::npos || end == std::string::npos) return false;

    std::string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
    array.clear();

    size_t current_pos = 0;
    while (current_pos < arrayStr.length()) {
        size_t quoteStart = arrayStr.find("\"", current_pos);
        if (quoteStart == std::string::npos) break;
        size_t quoteEnd = arrayStr.find("\"", quoteStart + 1);
        if (quoteEnd == std::string::npos) break;

        std::string item = arrayStr.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
        array.push_back(item);
        current_pos = quoteEnd + 1;
    }
    return true;
}

// 新增函数：解析布尔数组
bool ConfigParser::parseBoolArray(const std::string& jsonContent, const std::string& key, std::vector<bool>& array) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = jsonContent.find(searchKey);
    if (pos == std::string::npos) return false;

    pos = jsonContent.find("[", pos);
    size_t end = jsonContent.find("]", pos);
    if (pos == std::string::npos || end == std::string::npos) return false;

    std::string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
    std::stringstream ss(arrayStr);
    std::string item;
    array.clear();
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) {
            array.push_back(item == "true");
        }
    }
    return true;
}

bool ConfigParser::parseIntensityProperties(const std::string& jsonContent, ModelConfig& config) {
    std::string searchKey = "\"intensity_properties\":";
    size_t intensity_pos = jsonContent.find(searchKey);
    if (intensity_pos == std::string::npos) return false;
    size_t obj_start = jsonContent.find("{", intensity_pos + searchKey.length());
    if (obj_start == std::string::npos) return false;
    // --- 查找匹配的 '}' ---
    size_t obj_end = std::string::npos;
    int brace_count = 0;
    for (size_t i = obj_start; i < jsonContent.length(); ++i) {
        if (jsonContent[i] == '{') {
            brace_count++;
        } else if (jsonContent[i] == '}') {
            brace_count--;
            if (brace_count == 0) {
                obj_end = i;
                break;
            }
        }
    }
    // --- 查找结束 ---
    if (obj_end == std::string::npos) return false; // 没有找到匹配的 '}'
    // 提取整个 intensity_properties 对象的完整内容
    std::string intensity_obj_str = jsonContent.substr(obj_start, obj_end - obj_start + 1);
    // 清空现有的配置
    config.mean.clear();
    config.std.clear();
    config.min_val.clear();
    config.max_val.clear();
    config.percentile_00_5.clear();
    config.percentile_99_5.clear();
    int channel_index = 0;
    while (true) {
        std::string channel_key = "\"" + std::to_string(channel_index) + "\":";
        size_t channel_pos = intensity_obj_str.find(channel_key);
        if (channel_pos == std::string::npos) {
            break; // 找不到更多的通道了
        }
        size_t prop_start = intensity_obj_str.find("{", channel_pos);
        size_t prop_end = intensity_obj_str.find("}", prop_start); 
        if (prop_start == std::string::npos || prop_end == std::string::npos) {
            break; 
        }
        std::string prop_str = intensity_obj_str.substr(prop_start, prop_end - prop_start + 1);
        float mean_val, std_dev, min_v, max_v, p005, p995;
        parseFloatValue(prop_str, "mean", mean_val = 0.0f);
        parseFloatValue(prop_str, "std", std_dev = 1.0f);
        parseFloatValue(prop_str, "min", min_v = 0.0f);
        parseFloatValue(prop_str, "max", max_v = 1.0f);
        parseFloatValue(prop_str, "percentile_00_5", p005 = 0.0f);
        parseFloatValue(prop_str, "percentile_99_5", p995 = 1.0f);
        config.mean.push_back(mean_val);
        config.std.push_back(std_dev);
        config.min_val.push_back(min_v);
        config.max_val.push_back(max_v);
        config.percentile_00_5.push_back(p005);
        config.percentile_99_5.push_back(p995);

        channel_index++;
    }
    int num_channels_from_json = 0;
    if (parseIntValue(jsonContent, "num_input_channels", num_channels_from_json)) {
        config.num_input_channels = num_channels_from_json;
        if (channel_index > 0 && channel_index != num_channels_from_json) {
            std::cerr << "Warning: Number of channels in 'intensity_properties' (" << channel_index 
                      << ") does not match 'num_input_channels' (" << num_channels_from_json << ")." << std::endl;
        }
    } else if (channel_index > 0) {
        config.num_input_channels = channel_index;
    }
    return !config.mean.empty(); // 如果成功解析了至少一个通道的mean，就返回true
}

void ConfigParser::applyConfigToUnet(const ModelConfig& modelConfig, nnUNetConfig& unetConfig) {
    // Apply basic parameters
    unetConfig.num_classes = modelConfig.num_classes;
    unetConfig.input_channels = modelConfig.num_input_channels;
    
    // Apply patch size
    unetConfig.patch_size.clear();
    for (int val : modelConfig.patch_size) {
        unetConfig.patch_size.push_back(static_cast<int64_t>(val));
    }
    
    // Apply target spacing
    unetConfig.voxel_spacing = modelConfig.target_spacing;
    
    // Apply transpose settings
    unetConfig.transpose_forward = modelConfig.transpose_forward;
    unetConfig.transpose_backward = modelConfig.transpose_backward;
    
    // Build transpose strings for CImg
    std::string forward_str, backward_str;
    for (size_t i = 0; i < modelConfig.transpose_forward.size(); i++) {
        if (modelConfig.transpose_forward[i] == 0) forward_str += 'x';
        else if (modelConfig.transpose_forward[i] == 1) forward_str += 'y';
        else if (modelConfig.transpose_forward[i] == 2) forward_str += 'z';
    }
    for (size_t i = 0; i < modelConfig.transpose_backward.size(); i++) {
        if (modelConfig.transpose_backward[i] == 0) backward_str += 'x';
        else if (modelConfig.transpose_backward[i] == 1) backward_str += 'y';
        else if (modelConfig.transpose_backward[i] == 2) backward_str += 'z';
    }
    // Note: cimg_transpose_forward/backward are const char* and should be set elsewhere
    
    // Apply intensity properties with double precision
    unetConfig.means.clear();
    unetConfig.stds.clear();
    unetConfig.percentile_00_5s.clear();
    unetConfig.percentile_99_5s.clear();
    
    unetConfig.means.assign(modelConfig.mean.begin(), modelConfig.mean.end());
    unetConfig.stds.assign(modelConfig.std.begin(), modelConfig.std.end());
    unetConfig.percentile_00_5s.assign(modelConfig.percentile_00_5.begin(), modelConfig.percentile_00_5.end());
    unetConfig.percentile_99_5s.assign(modelConfig.percentile_99_5.begin(), modelConfig.percentile_99_5.end());
    // 应用 normalization settings
    unetConfig.normalization_schemes = modelConfig.normalization_scheme;
    unetConfig.use_mask_for_norm = modelConfig.use_mask_for_norm;
    unetConfig.use_mirroring = modelConfig.use_tta;
}