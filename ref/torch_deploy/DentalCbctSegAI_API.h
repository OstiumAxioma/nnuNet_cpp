#ifndef _DentalCbctSegAI_API__h
#define _DentalCbctSegAI_API__h


// ////////////////////////////////////////////////////////////////////////////
// 文件：DentalCbctSegAI_API.h
// 作者：阳维
// 说明：定义 口腔CBCT结构分割 接口
//
// 创建日期：2025-4-30

// ////////////////////////////////////////////////////////////////////////////

#define DentalCbctSegAI_API  extern "C" __declspec(dllexport)


#define DentalCbctSegAI_STATUS_SUCCESS          0   // 成功
#define DentalCbctSegAI_STATUS_HANDLE_NULL      1   // 空句柄，请首先调用 DentalCbctSegAI_CreateObj() 函数创建句柄
#define DentalCbctSegAI_STATUS_VOLUME_SMALL     2   // 输入体数据过小
#define DentalCbctSegAI_STATUS_VOLUME_LARGE     3   // 输入体数据过大
#define DentalCbctSegAI_STATUS_CROP_FAIED       4   // 定位牙齿区域失败
#define DentalCbctSegAI_STATUS_FAIED            5   // 分割牙齿失败
#define DentalCbctSegAI_LOADING_FAIED           6   // 载入AI模型数据失败

// --------------------------------------------------------------------
//            预定义类型
// --------------------------------------------------------------------
typedef unsigned char    AI_UCHAR;
typedef unsigned short   AI_USHORT;
typedef bool             AI_BOOL;
typedef short            AI_SHORT;
typedef int              AI_INT;
typedef float            AI_FLOAT;
typedef void             AI_VOID;
typedef void*            AI_HANDLE;
typedef char*            AI_STRING;


//体数据结构
typedef struct
{
	AI_SHORT    *ptr_Data;      // 数据指针
	AI_INT       Width;         // 横断面宽
	AI_INT       Height;        // 横断面高
	AI_INT       Depth;         // 轴向层数
	AI_FLOAT     VoxelSpacing;  // 体素大小，单位：mm
	AI_FLOAT     VoxelSpacingX;  // 体素大小，单位：mm
	AI_FLOAT     VoxelSpacingY;  // 体素大小，单位：mm
	AI_FLOAT     VoxelSpacingZ;  // 体素大小，单位：mm
} AI_DataInfo;


// 创建句柄
// 进行初始化（读取算法需要的参数与模板）
// 初始化失败：返回NULL；非空表示初始化成功
DentalCbctSegAI_API AI_HANDLE    DentalCbctSegAI_CreateObj();

//设置分割模型文件路径
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_SetModelPath(AI_HANDLE AI_Hdl, AI_STRING fn);

//设置滑动窗步长比例
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_SetTileStepRatio(AI_HANDLE AI_Hdl, AI_FLOAT ratio);

//设置标记球直径和是否进行去除处理
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_SetMarkerBallDiameter(AI_HANDLE AI_Hdl, AI_FLOAT diameter, AI_BOOL remove_balls);

// 获取检测到的标记球数量(必须根据标记点数量，分配好float*空间，再调用获取标记点坐标接口)
DentalCbctSegAI_API AI_INT		 DentalCbctSegAI_GetMarkerCount(AI_HANDLE AI_Hdl);

// 获取检测到的标记球坐标
DentalCbctSegAI_API AI_INT		 DentalCbctSegAI_GetMarkerInfo(AI_HANDLE AI_Hdl, float* markerInfo);

// 分割口腔CBCT（CPU计算大约1分钟）
// AI_Hdl: 初始化产生的句柄；
// srcData: 输入口腔CBCT图像数据
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_Infer(AI_HANDLE AI_Hdl, AI_DataInfo *srcData);

// 获取分割结果
// AI_Hdl: 初始化产生的句柄；
//分割Mask标签说明：
//1：下颌骨；2：上颌骨；3：上颌窦；4：下颌神经管；5：上牙；6：下牙； 0：背景
//DentalCbctSegAI_API AI_INT       DentalCbctSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData, AI_INT &totalToothNumber, AI_INT &upperToothNumber, AI_INT &lowerToothNumber);
DentalCbctSegAI_API AI_INT       DentalCbctSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData);

// 释放资源
DentalCbctSegAI_API AI_VOID      DentalCbctSegAI_ReleseObj(AI_HANDLE AI_Hdl);

#endif
