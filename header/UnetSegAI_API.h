#ifndef _UnetSegAI_API__h
#define _UnetSegAI_API__h


// ////////////////////////////////////////////////////////////////////////////
// �ļ���UnetSegAI_API.h
// ���ߣ���ά
// ˵�������� ��ǻCBCT�ṹ�ָ� �ӿ�
//
// �������ڣ�2025-4-30

// ////////////////////////////////////////////////////////////////////////////

#define UnetSegAI_API  extern "C" __declspec(dllexport)


#define UnetSegAI_STATUS_SUCCESS          0   // �ɹ�
#define UnetSegAI_STATUS_HANDLE_NULL      1   // �վ���������ȵ��� UnetSegAI_CreateObj() �����������
#define UnetSegAI_STATUS_VOLUME_SMALL     2   // ���������ݹ�С
#define UnetSegAI_STATUS_VOLUME_LARGE     3   // ���������ݹ���
#define UnetSegAI_STATUS_CROP_FAIED       4   // ��λ��������ʧ��
#define UnetSegAI_STATUS_FAIED            5   // �ָ�����ʧ��
#define UnetSegAI_LOADING_FAIED           6   // ����AIģ������ʧ��

// --------------------------------------------------------------------
//            Ԥ��������
// --------------------------------------------------------------------
typedef unsigned char    AI_UCHAR;
typedef unsigned short   AI_USHORT;
typedef short            AI_SHORT;
typedef int              AI_INT;
typedef float            AI_FLOAT;
typedef void             AI_VOID;
typedef void*            AI_HANDLE;
typedef wchar_t*         AI_STRING;
typedef int              AI_BOOL;


//�����ݽṹ
typedef struct
{
	AI_SHORT    *ptr_Data;      // ����ָ��
	AI_INT       Width;         // ������
	AI_INT       Height;        // ������
	AI_INT       Depth;         // �������
	AI_INT       Channels;
	AI_FLOAT     VoxelSpacing;  // ���ش�С����λ��mm
	AI_FLOAT     VoxelSpacingX;  // ���ش�С����λ��mm
	AI_FLOAT     VoxelSpacingY;  // ���ش�С����λ��mm
	AI_FLOAT     VoxelSpacingZ;  // ���ش�С����λ��mm
	// ���ӣ�ԭʼspacing�ֶΣ����ļ���ȡ����ʵ����spacing��
	AI_FLOAT     OriginalVoxelSpacingX;  // ԭʼ���ش�С����λ��mm
	AI_FLOAT     OriginalVoxelSpacingY;  // ԭʼ���ش�С����λ��mm
	AI_FLOAT     OriginalVoxelSpacingZ;  // ԭʼ���ش�С����λ��mm
	// ���ӣ�origin�ֶΣ��������ҽѧͼ���ѧԭ��
	AI_FLOAT     OriginX;  // ԭ��X����
	AI_FLOAT     OriginY;  // ԭ��Y����
	AI_FLOAT     OriginZ;  // ԭ��Z����
} AI_DataInfo;


// �������
// ���г�ʼ������ȡ�㷨��Ҫ�Ĳ�����ģ�壩
// ��ʼ��ʧ�ܣ�����NULL���ǿձ�ʾ��ʼ���ɹ�
UnetSegAI_API AI_HANDLE    UnetSegAI_CreateObj();

//���÷ָ�ģ���ļ�·��
UnetSegAI_API AI_INT       UnetSegAI_SetModelPath(AI_HANDLE AI_Hdl, AI_STRING fn);

//���û�������������
UnetSegAI_API AI_INT       UnetSegAI_SetTileStepRatio(AI_HANDLE AI_Hdl, AI_FLOAT ratio);

// �ָ��ǻCBCT��CPU�����Լ1���ӣ�
// AI_Hdl: ��ʼ�������ľ����
// srcData: �����ǻCBCTͼ������
UnetSegAI_API AI_INT       UnetSegAI_Infer(AI_HANDLE AI_Hdl, AI_DataInfo *srcData);

// ��ȡ�ָ���
// AI_Hdl: ��ʼ�������ľ����
//�ָ�Mask��ǩ˵����
//1�����ǣ�2�����ǣ�3�����񼣻4������񾭹ܣ�5��������6�������� 0������
//UnetSegAI_API AI_INT       UnetSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData, AI_INT &totalToothNumber, AI_INT &upperToothNumber, AI_INT &lowerToothNumber);
UnetSegAI_API AI_INT       UnetSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData);

// ���ӣ��趨������·������Ҫ���������������ļ���
UnetSegAI_API AI_INT       UnetSegAI_SetOutputPaths(AI_HANDLE AI_Hdl, 
                                                                 AI_STRING preprocessPath, 
                                                                 AI_STRING modelOutputPath, 
                                                                 AI_STRING postprocessPath);

// ���Ӳ���������ؽӿ�
UnetSegAI_API AI_INT       UnetSegAI_SetPatchSize(AI_HANDLE AI_Hdl, AI_INT x, AI_INT y, AI_INT z);
UnetSegAI_API AI_INT       UnetSegAI_SetNumClasses(AI_HANDLE AI_Hdl, AI_INT classes);
UnetSegAI_API AI_INT       UnetSegAI_SetInputChannels(AI_HANDLE AI_Hdl, AI_INT channels);
UnetSegAI_API AI_INT       UnetSegAI_SetTargetSpacing(AI_HANDLE AI_Hdl, AI_FLOAT x, AI_FLOAT y, AI_FLOAT z);
UnetSegAI_API AI_INT       UnetSegAI_SetTransposeSettings(AI_HANDLE AI_Hdl, 
                                                                       AI_INT forward_x, AI_INT forward_y, AI_INT forward_z,
                                                                       AI_INT backward_x, AI_INT backward_y, AI_INT backward_z);
UnetSegAI_API AI_INT       UnetSegAI_SetNormalizationType(AI_HANDLE AI_Hdl, const char* type);
UnetSegAI_API AI_INT       UnetSegAI_SetIntensityProperties(AI_HANDLE AI_Hdl, 
                                                                         AI_FLOAT mean, AI_FLOAT std, 
                                                                         AI_FLOAT min_val, AI_FLOAT max_val,
                                                                         AI_FLOAT percentile_00_5, AI_FLOAT percentile_99_5);
UnetSegAI_API AI_INT       UnetSegAI_SetUseMirroring(AI_HANDLE AI_Hdl, AI_BOOL use_mirroring);

// ���ӣ�JSON�����ӿ�
UnetSegAI_API AI_INT       UnetSegAI_SetConfigFromJson(AI_HANDLE AI_Hdl, const char* jsonContent);

// �ͷ���Դ
UnetSegAI_API AI_VOID      UnetSegAI_ReleseObj(AI_HANDLE AI_Hdl);

#endif