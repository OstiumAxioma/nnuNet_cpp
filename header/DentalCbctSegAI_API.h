#ifndef _DentalCbctSegAI_API__h
#define _DentalCbctSegAI_API__h


// ////////////////////////////////////////////////////////////////////////////
// �ļ���DentalCbctSegAI_API.h
// ���ߣ���ά
// ˵�������� ��ǻCBCT�ṹ�ָ� �ӿ�
//
// �������ڣ�2025-4-30

// ////////////////////////////////////////////////////////////////////////////

#define DentalCbctSegAI_API  extern "C" __declspec(dllexport)


#define DentalCbctSegAI_STATUS_SUCCESS          0   // �ɹ�
#define DentalCbctSegAI_STATUS_HANDLE_NULL      1   // �վ���������ȵ��� DentalCbctSegAI_CreateObj() �����������
#define DentalCbctSegAI_STATUS_VOLUME_SMALL     2   // ���������ݹ�С
#define DentalCbctSegAI_STATUS_VOLUME_LARGE     3   // ���������ݹ���
#define DentalCbctSegAI_STATUS_CROP_FAIED       4   // ��λ��������ʧ��
#define DentalCbctSegAI_STATUS_FAIED            5   // �ָ�����ʧ��
#define DentalCbctSegAI_LOADING_FAIED           6   // ����AIģ������ʧ��

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


//�����ݽṹ
typedef struct
{
	AI_SHORT    *ptr_Data;      // ����ָ��
	AI_INT       Width;         // ������
	AI_INT       Height;        // ������
	AI_INT       Depth;         // �������
	AI_FLOAT     VoxelSpacing;  // ���ش�С����λ��mm
	AI_FLOAT     VoxelSpacingX;  // ���ش�С����λ��mm
	AI_FLOAT     VoxelSpacingY;  // ���ش�С����λ��mm
	AI_FLOAT     VoxelSpacingZ;  // ���ش�С����λ��mm
} AI_DataInfo;


// �������
// ���г�ʼ������ȡ�㷨��Ҫ�Ĳ�����ģ�壩
// ��ʼ��ʧ�ܣ�����NULL���ǿձ�ʾ��ʼ���ɹ�
DentalCbctSegAI_API AI_HANDLE    DentalCbctSegAI_CreateObj();

//���÷ָ�ģ���ļ�·��
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_SetModelPath(AI_HANDLE AI_Hdl, AI_STRING fn);

//���û�������������
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_SetTileStepRatio(AI_HANDLE AI_Hdl, AI_FLOAT ratio);

// �ָ��ǻCBCT��CPU�����Լ1���ӣ�
// AI_Hdl: ��ʼ�������ľ����
// srcData: �����ǻCBCTͼ������
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_Infer(AI_HANDLE AI_Hdl, AI_DataInfo *srcData);

// ��ȡ�ָ���
// AI_Hdl: ��ʼ�������ľ����
//�ָ�Mask��ǩ˵����
//1�����ǣ�2�����ǣ�3�����񼣻4������񾭹ܣ�5��������6�������� 0������
//DentalCbctSegAI_API AI_INT       DentalCbctSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData, AI_INT &totalToothNumber, AI_INT &upperToothNumber, AI_INT &lowerToothNumber);
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData);

// �ͷ���Դ
DentalCbctSegAI_API AI_VOID      DentalCbctSegAI_ReleseObj(AI_HANDLE AI_Hdl);

#endif
