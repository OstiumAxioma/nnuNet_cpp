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


DentalCbctSegAI_API AI_VOID      DentalCbctSegAI_ReleseObj(AI_HANDLE AI_Hdl)
{
	if (AI_Hdl == NULL)
		return;

	DentalUnet *pAIObj = reinterpret_cast<DentalUnet *>(AI_Hdl);

	delete pAIObj;
}