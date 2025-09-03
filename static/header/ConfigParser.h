#pragma once

#include <vector>
#include <string>

// Forward declaration
struct nnUNetConfig;

// JSON配置结构
class ModelConfig {
public:
    std::vector<int> all_labels;
    int num_classes;
    int num_input_channels;
    std::vector<int> patch_size;
    std::vector<float> target_spacing;
    std::vector<int> transpose_forward;
    std::vector<int> transpose_backward;
    float mean;
    float std;
    float min_val;
    float max_val;
    float percentile_00_5;
    float percentile_99_5;
    std::string normalization_scheme;
    bool use_mask_for_norm;
    bool use_tta;
    
    ModelConfig();
    void setDefaults();
};

class ConfigParser {
public:
    ConfigParser();
    ~ConfigParser();

    bool parseJsonString(const std::string& jsonContent, ModelConfig& config);
    const ModelConfig& getConfig() const;
    
    // 静态方法：将ModelConfig应用到nnUNetConfig
    static void applyConfigToUnet(const ModelConfig& modelConfig, nnUNetConfig& unetConfig);

private:
    ModelConfig currentConfig;
    
    bool parseIntValue(const std::string& jsonContent, const std::string& key, int& value);
    bool parseFloatValue(const std::string& jsonContent, const std::string& key, float& value);
    bool parseStringValue(const std::string& jsonContent, const std::string& key, std::string& value);
    bool parseBoolValue(const std::string& jsonContent, const std::string& key, bool& value);
    bool parseIntArray(const std::string& jsonContent, const std::string& key, std::vector<int>& array);
    bool parseFloatArray(const std::string& jsonContent, const std::string& key, std::vector<float>& array);
    bool parseIntensityProperties(const std::string& jsonContent, ModelConfig& config);
};