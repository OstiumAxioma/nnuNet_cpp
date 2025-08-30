#include "UnetSegAI_API.h"
#include "UnetMain.h"
#include <string>
#include <cstring>
#include <cwchar>


UnetSegAI_API AI_HANDLE    UnetSegAI_CreateObj()
{
	
	AI_HANDLE AI_Hdl = NULL;

	UnetMain *pAIObj = NULL;
	pAIObj = UnetMain::CreateUnetMain();

	if (pAIObj == NULL)
		return NULL;

	AI_Hdl = reinterpret_cast<void *>(pAIObj);

	return AI_Hdl;
}


UnetSegAI_API AI_INT    UnetSegAI_SetModelPath(AI_HANDLE AI_Hdl, AI_STRING fn)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	if (fn == NULL)
		return UnetSegAI_STATUS_FAIED;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);

	// AI_STRING is already wchar_t* in static library header
	pAIObj->setModelFns(fn);

	return UnetSegAI_STATUS_SUCCESS;
}


UnetSegAI_API AI_INT       UnetSegAI_SetTileStepRatio(AI_HANDLE AI_Hdl, AI_FLOAT ratio)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);

	pAIObj->setStepSizeRatio(ratio);

	return UnetSegAI_STATUS_SUCCESS;
}


UnetSegAI_API AI_INT       UnetSegAI_Infer(AI_HANDLE AI_Hdl, AI_DataInfo *srcData)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);

	AI_INT AIWorkWell = pAIObj->performInference(srcData);

	return AIWorkWell;
}


UnetSegAI_API AI_INT       UnetSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->getSegMask(dstData);

	return UnetSegAI_STATUS_SUCCESS;
}


UnetSegAI_API AI_INT       UnetSegAI_SetOutputPaths(AI_HANDLE AI_Hdl, 
                                                                 AI_STRING preprocessPath, 
                                                                 AI_STRING modelOutputPath, 
                                                                 AI_STRING postprocessPath)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	
	// AI_STRING is already wchar_t* in static library header
	pAIObj->setOutputPaths(preprocessPath, modelOutputPath, postprocessPath);

	return UnetSegAI_STATUS_SUCCESS;
}

// 参数设置相关接口的实现
UnetSegAI_API AI_INT UnetSegAI_SetPatchSize(AI_HANDLE AI_Hdl, AI_INT x, AI_INT y, AI_INT z)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setPatchSize(x, y, z);
	return UnetSegAI_STATUS_SUCCESS;
}

UnetSegAI_API AI_INT UnetSegAI_SetNumClasses(AI_HANDLE AI_Hdl, AI_INT classes)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setNumClasses(classes);
	return UnetSegAI_STATUS_SUCCESS;
}

UnetSegAI_API AI_INT UnetSegAI_SetInputChannels(AI_HANDLE AI_Hdl, AI_INT channels)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setInputChannels(channels);
	return UnetSegAI_STATUS_SUCCESS;
}

UnetSegAI_API AI_INT UnetSegAI_SetTargetSpacing(AI_HANDLE AI_Hdl, AI_FLOAT x, AI_FLOAT y, AI_FLOAT z)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setTargetSpacing(x, y, z);
	return UnetSegAI_STATUS_SUCCESS;
}

UnetSegAI_API AI_INT UnetSegAI_SetTransposeSettings(AI_HANDLE AI_Hdl, 
                                                                AI_INT forward_x, AI_INT forward_y, AI_INT forward_z,
                                                                AI_INT backward_x, AI_INT backward_y, AI_INT backward_z)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setTransposeSettings(forward_x, forward_y, forward_z, backward_x, backward_y, backward_z);
	return UnetSegAI_STATUS_SUCCESS;
}

UnetSegAI_API AI_INT UnetSegAI_SetNormalizationType(AI_HANDLE AI_Hdl, const char* type)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setNormalizationType(type);
	return UnetSegAI_STATUS_SUCCESS;
}

UnetSegAI_API AI_INT UnetSegAI_SetIntensityProperties(AI_HANDLE AI_Hdl, 
                                                                  AI_FLOAT mean, AI_FLOAT std, 
                                                                  AI_FLOAT min_val, AI_FLOAT max_val,
                                                                  AI_FLOAT percentile_00_5, AI_FLOAT percentile_99_5)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setIntensityProperties(mean, std, min_val, max_val, percentile_00_5, percentile_99_5);
	return UnetSegAI_STATUS_SUCCESS;
}

UnetSegAI_API AI_INT UnetSegAI_SetUseMirroring(AI_HANDLE AI_Hdl, AI_BOOL use_mirroring)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	pAIObj->setUseMirroring(use_mirroring != 0);
	return UnetSegAI_STATUS_SUCCESS;
}

// 新增：JSON配置接口实现
UnetSegAI_API AI_INT UnetSegAI_SetConfigFromJson(AI_HANDLE AI_Hdl, const char* jsonContent)
{
	if (AI_Hdl == NULL)
		return UnetSegAI_STATUS_HANDLE_NULL;
		
	if (jsonContent == NULL)
		return UnetSegAI_STATUS_FAIED;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);
	bool result = pAIObj->setConfigFromJsonString(jsonContent);
	
	return result ? UnetSegAI_STATUS_SUCCESS : UnetSegAI_STATUS_FAIED;
}


UnetSegAI_API AI_VOID      UnetSegAI_ReleseObj(AI_HANDLE AI_Hdl)
{
	if (AI_Hdl == NULL)
		return;

	UnetMain *pAIObj = reinterpret_cast<UnetMain *>(AI_Hdl);

	delete pAIObj;
}