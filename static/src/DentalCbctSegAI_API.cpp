#include "DentalCbctSegAI_API.h"
#include "DentalUnet.h"
#include <string>
#include <cstring>
#include <cwchar>


DentalCbctSegAI_API AI_HANDLE    DentalCbctSegAI_CreateObj()
{
	
	AI_HANDLE AI_Hdl = NULL;

	DentalUnet *pAIObj = NULL;
	pAIObj = DentalUnet::CreateDentalUnet();

	if (pAIObj == NULL)
		return NULL;

	AI_Hdl = reinterpret_cast<void *>(pAIObj);

	return AI_Hdl;
}


DentalCbctSegAI_API AI_INT    DentalCbctSegAI_SetModelPath(AI_HANDLE AI_Hdl, AI_STRING fn)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	if (fn == NULL)
		return DentalCbctSegAI_STATUS_FAIED;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);

	// AI_STRING is already wchar_t* in static library header
	pAIObj->setModelFns(fn);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


DentalCbctSegAI_API AI_INT       DentalCbctSegAI_SetTileStepRatio(AI_HANDLE AI_Hdl, AI_FLOAT ratio)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);

	pAIObj->setStepSizeRatio(ratio);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


DentalCbctSegAI_API AI_INT       DentalCbctSegAI_Infer(AI_HANDLE AI_Hdl, AI_DataInfo *srcData)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);

	AI_INT AIWorkWell = pAIObj->performInference(srcData);

	return AIWorkWell;
}


DentalCbctSegAI_API AI_INT       DentalCbctSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->getSegMask(dstData);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


DentalCbctSegAI_API AI_INT       DentalCbctSegAI_SetOutputPaths(AI_HANDLE AI_Hdl, 
                                                                 AI_STRING preprocessPath, 
                                                                 AI_STRING modelOutputPath, 
                                                                 AI_STRING postprocessPath)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	
	// AI_STRING is already wchar_t* in static library header
	pAIObj->setOutputPaths(preprocessPath, modelOutputPath, postprocessPath);

	return DentalCbctSegAI_STATUS_SUCCESS;
}

// ���Ӳ���������ؽӿ���ʵ��
DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetPatchSize(AI_HANDLE AI_Hdl, AI_INT x, AI_INT y, AI_INT z)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setPatchSize(x, y, z);
	return DentalCbctSegAI_STATUS_SUCCESS;
}

DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetNumClasses(AI_HANDLE AI_Hdl, AI_INT classes)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setNumClasses(classes);
	return DentalCbctSegAI_STATUS_SUCCESS;
}

DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetInputChannels(AI_HANDLE AI_Hdl, AI_INT channels)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setInputChannels(channels);
	return DentalCbctSegAI_STATUS_SUCCESS;
}

DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetTargetSpacing(AI_HANDLE AI_Hdl, AI_FLOAT x, AI_FLOAT y, AI_FLOAT z)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setTargetSpacing(x, y, z);
	return DentalCbctSegAI_STATUS_SUCCESS;
}

DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetTransposeSettings(AI_HANDLE AI_Hdl, 
                                                                AI_INT forward_x, AI_INT forward_y, AI_INT forward_z,
                                                                AI_INT backward_x, AI_INT backward_y, AI_INT backward_z)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setTransposeSettings(forward_x, forward_y, forward_z, backward_x, backward_y, backward_z);
	return DentalCbctSegAI_STATUS_SUCCESS;
}

DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetNormalizationType(AI_HANDLE AI_Hdl, const char* type)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setNormalizationType(type);
	return DentalCbctSegAI_STATUS_SUCCESS;
}

DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetIntensityProperties(AI_HANDLE AI_Hdl, 
                                                                  AI_FLOAT mean, AI_FLOAT std, 
                                                                  AI_FLOAT min_val, AI_FLOAT max_val,
                                                                  AI_FLOAT percentile_00_5, AI_FLOAT percentile_99_5)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setIntensityProperties(mean, std, min_val, max_val, percentile_00_5, percentile_99_5);
	return DentalCbctSegAI_STATUS_SUCCESS;
}

DentalCbctSegAI_API AI_INT DentalCbctSegAI_SetUseMirroring(AI_HANDLE AI_Hdl, AI_BOOL use_mirroring)
{
	if (AI_Hdl == NULL)
		return DentalCbctSegAI_STATUS_HANDLE_NULL;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);
	pAIObj->setUseMirroring(use_mirroring != 0);
	return DentalCbctSegAI_STATUS_SUCCESS;
}


DentalCbctSegAI_API AI_VOID      DentalCbctSegAI_ReleseObj(AI_HANDLE AI_Hdl)
{
	if (AI_Hdl == NULL)
		return;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);

	delete pAIObj;
}