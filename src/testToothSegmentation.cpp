// testToothSegmentation.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <memory>
#include <chrono>
#include <windows.h>
#include <exception>
#include <csignal>
#include <typeinfo>
#include <clocale>
#include <cstdlib>


// onnx 配置
#include "..\header\DentalCbctSegAI_API.h"
#include "..\lib\onnxruntime\include\onnxruntime_cxx_api.h"
#pragma comment(lib, "..\\lib\\DentalCbctOnnxSegDLL.lib")

#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime.lib")
#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime_providers_shared.lib")
#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime_providers_cuda.lib")
//#pragma comment(lib,"..\\utility\\onnxruntime\\lib\\onnxruntime_providers_tensorrt.lib")


/*
// libtorch 231 配置
#include "DentalCbctSegAI_API.h"
#pragma comment(lib, "..\\x64\\Release\\DentalCbctSegDLL.lib")

#pragma comment(lib,"..\\utility\\libtorch231\\lib\\asmjit.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\c10.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\c10_cuda.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\cpuinfo.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\fbgemm.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\torch.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\torch_cpu.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\torch_cuda.lib")
*/


//"/INCLUDE:?warp_size@cuda@at@@YAHXZ"

//113:
//"/INCLUDE:?warp_size@cuda@at@@YAHXZ /INCLUDE:?_torch_cuda_cu_linker_symbol_op_cuda@native@at@@YA?AVTensor@2@AEBV32@@Z"

//230
//" /INCLUDE:?warp_size@cuda@at@@YAHXZ /INCLUDE:"?ignore_this_library_placeholder@@YAHXZ" "

//CImg用于读入存储体数据
#define cimg_display_type 2
#include "..\lib\CImg\CImg.h"

using namespace std;
using namespace cimg_library;

// 信号处理函数
void SignalHandler(int signal) {
    std::cerr << "\n===== Program Exception =====" << std::endl;
    std::cerr << "Signal: " << signal << std::endl;
    switch(signal) {
        case SIGSEGV:
            std::cerr << "Segmentation fault (SIGSEGV)" << std::endl;
            break;
        case SIGFPE:
            std::cerr << "Floating point exception (SIGFPE)" << std::endl;
            break;
        case SIGILL:
            std::cerr << "Illegal instruction (SIGILL)" << std::endl;
            break;
        case SIGABRT:
            std::cerr << "Program abort (SIGABRT)" << std::endl;
            break;
    }
    system("pause");
    exit(1);
}

// SEH异常过滤器
LONG WINAPI MyUnhandledExceptionFilter(EXCEPTION_POINTERS* pExceptionInfo) {
    std::cerr << "\n===== Unhandled SEH Exception =====" << std::endl;
    std::cerr << "Exception Code: 0x" << std::hex << pExceptionInfo->ExceptionRecord->ExceptionCode << std::dec << std::endl;
    
    switch(pExceptionInfo->ExceptionRecord->ExceptionCode) {
        case EXCEPTION_ACCESS_VIOLATION:
            std::cerr << "Access Violation" << std::endl;
            break;
        case EXCEPTION_STACK_OVERFLOW:
            std::cerr << "Stack Overflow" << std::endl;
            break;
        case 0xE06D7363:
            std::cerr << "C++ Exception (0xE06D7363)" << std::endl;
            break;
    }
    
    system("pause");
    return EXCEPTION_EXECUTE_HANDLER;
}

// 单独的函数用于执行可能抛出SEH异常的代码
int SafeInfer(AI_HANDLE AI_Hdl, AI_DataInfo* srcData, AI_INT& AIWorkStatus) {
    __try {
        AIWorkStatus = DentalCbctSegAI_Infer(AI_Hdl, srcData);
        return 0; // 成功
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        DWORD exceptionCode = GetExceptionCode();
        std::cerr << "\n===== SEH Exception in DentalCbctSegAI_Infer =====" << std::endl;
        std::cerr << "Exception Code: 0x" << std::hex << exceptionCode << std::dec << std::endl;
        
        if (exceptionCode == 0xE06D7363) {
            std::cerr << "DLL内部抛出了C++异常" << std::endl;
            // 实际的错误信息应该在DLL的输出日志中
        }
        
        return -1; // 失败
    }
}

int main()
{	
	// 设置locale以支持中文
	//setlocale(LC_ALL, "chs");
	
	
	// 安装信号处理器
	signal(SIGSEGV, SignalHandler);
	signal(SIGFPE, SignalHandler);
	signal(SIGILL, SignalHandler);
	signal(SIGABRT, SignalHandler);
	
	// 设置SEH异常处理器
	SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
	
	try {
		std::cout << "程序开始运行..." << std::endl;
		
		//load raw volume data: 牙齿在前，后脑勺在后；耳朵在左右；下巴在上，头顶在下
		std::cout << "正在加载HDR图像文件..." << std::endl;
		CImg<short> inputCbctVolume;
		inputCbctVolume.load_analyze("..\\..\\..\\img\\Series_5_Acq_2_0000.hdr");
		std::cout << "HDR文件加载成功" << std::endl;

	float VoxelSpacing  = 1.0f; //unit: mm  0.3
	float VoxelSpacingX = 0.5810545086860657f; //unit: mm  0.3
	float VoxelSpacingY = 0.5810545086860657f; //unit: mm  0.3
	float VoxelSpacingZ = 1.0f; //unit: mm  0.3
	
	int Width0 = inputCbctVolume.width();
	int Height0 = inputCbctVolume.height();
	int Depth0 = inputCbctVolume.depth();

	//检查slice方位，显示图像中牙齿应在上方
	//如果牙齿方位不对，可通过rotate XY 90、180、-90度进行调整
	//inputCbctVolume.rotate(180); //90, -90, 180
	//inputCbctVolume -= 1024;

	//CImg<short> slice_z = inputCbctVolume.get_slice( Depth0 / 2);
	//slice_z.display("slice 120");
	//关闭显示窗口后，程序继续运行

	//设置输入体数据信息
	short* ptrCbctData = inputCbctVolume.data();
	AI_DataInfo *srcData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
	srcData->Width = Width0;
	srcData->Height = Height0;
	srcData->Depth = Depth0;
	srcData->VoxelSpacing = VoxelSpacing;
	srcData->VoxelSpacingX = VoxelSpacingX;
	srcData->VoxelSpacingY = VoxelSpacingY;
	srcData->VoxelSpacingZ = VoxelSpacingZ;
	srcData->ptr_Data = ptrCbctData; //CBCT体数据指针

	//初始化牙齿分割结果数据信息
	CImg<short> toothLabelMask = CImg<short>(Width0, Height0, Depth0, 1, 0);

	AI_DataInfo *toothSegData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
	toothSegData->Width = Width0;
	toothSegData->Height = Height0;
	toothSegData->Depth = Depth0;
	toothSegData->VoxelSpacing = VoxelSpacing;
	toothSegData->VoxelSpacingX = VoxelSpacingX;
	toothSegData->VoxelSpacingY = VoxelSpacingY;
	toothSegData->VoxelSpacingZ = VoxelSpacingZ;
	toothSegData->ptr_Data = toothLabelMask.data();//分割label体数据指针


	auto start = std::chrono::steady_clock::now();

	//调用牙齿分割模型
	//初始化分割模型对象
	std::cout << "\nInitializing segmentation model..." << std::endl;
	AI_HANDLE  AI_Hdl = DentalCbctSegAI_CreateObj();
	if (AI_Hdl == NULL) {
		std::cerr << "Error: Model initialization failed!" << std::endl;
		return DentalCbctSegAI_STATUS_HANDLE_NULL; //模型初始化失败
	}
	std::cout << "Model initialized successfully" << std::endl;

	std::cout << "Setting model path..." << std::endl;
	AI_INT status1 = DentalCbctSegAI_SetModelPath(AI_Hdl, const_cast<char*>("..\\..\\..\\model\\kneeseg_test.onnx"));
	std::cout << "SetModelPath返回状态: " << status1 << std::endl;

	// 设置TileStepRatio
	float tileRatio = 0.5f;
	AI_INT status2 = DentalCbctSegAI_SetTileStepRatio(AI_Hdl, tileRatio);
	std::cout << "SetTileStepRatio(" << tileRatio << ")返回状态: " << status2 << std::endl;

	// 打印输入数据信息
	std::cout << "\n输入数据信息:" << std::endl;
	std::cout << "  尺寸: " << srcData->Width << " x " << srcData->Height << " x " << srcData->Depth << std::endl;
	std::cout << "  体素间距: X=" << srcData->VoxelSpacingX << ", Y=" << srcData->VoxelSpacingY << ", Z=" << srcData->VoxelSpacingZ << std::endl;
	std::cout << "  数据指针: " << (void*)srcData->ptr_Data << std::endl;

	// 调用模型推理，并捕获 ONNX Runtime 异常
	std::cout << "\n开始模型推理..." << std::endl;
	
	AI_INT	AIWorkStatus = DentalCbctSegAI_STATUS_FAIED;
	int result = SafeInfer(AI_Hdl, srcData, AIWorkStatus);
	
	if (result != 0) {
		// SEH异常发生
		// 释放资源
		if (AI_Hdl) DentalCbctSegAI_ReleseObj(AI_Hdl);
		if (srcData) free(srcData);
		if (toothSegData) free(toothSegData);
		
		system("pause");
		return -7001;
	}
	
	std::cout << "模型推理完成，状态码: " << AIWorkStatus << std::endl;
	

	//获取牙齿分割结果
	if (AIWorkStatus == DentalCbctSegAI_STATUS_SUCCESS)
		DentalCbctSegAI_GetResult(AI_Hdl, toothSegData);
	else
		return AIWorkStatus;

	//释放对象
	DentalCbctSegAI_ReleseObj(AI_Hdl);
	// 牙齿分割过程结束

	//输出结果toothSegData说明：
	//对于小视野CBCT：
    //totalToothNumber:分割的牙齿总数
    //牙齿编号k=1,2,3,...,totalToothNumber，牙髓label为3k、牙本质label为3k+1、牙冠或金属植入物label为3k+2
    //对于大视野CBCT：
    //upperToothNumber:上牙数量
    //lower_tooth_number:下牙数量
    //上牙齿编号k=1,2,3,...,upperToothNumber，牙髓label为3k、牙本质label为3k+1、牙冠或金属植入物label为3k+2
    //下牙齿编号k=-1,-2,-3,...,-lowerToothNumber，牙髓label为-3k、牙本质label为-3k-1、牙冠或金属植入物label为-3k-2
	

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	//CImg<short> mask_z = toothLabelMask.get_slice(Depth0 / 2);
	//mask_z.display("mask 205");


	//保存分割结果
	//inputCbctVolume.save_analyze("inputCbctVolume.hdr");
	toothLabelMask.save_analyze("finalLabelMask.hdr");

		std::cout << "\n程序执行成功完成!" << std::endl;
		return AIWorkStatus;
		
	} catch (const CImgIOException& e) {
		std::cerr << "\n===== CImg IO异常 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -2001;
	} catch (const CImgException& e) {
		std::cerr << "\n===== CImg异常 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -2002;
	} catch (const Ort::Exception& e) {
		std::cerr << "\n===== ONNX Runtime异常 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -3001;
	} catch (const std::bad_alloc& e) {
		std::cerr << "\n===== 内存分配失败 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -4001;
	} catch (const std::exception& e) {
		std::cerr << "\n===== 标准异常 =====" << std::endl;
		std::cerr << "异常类型: " << typeid(e).name() << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -5001;
	} catch (...) {
		std::cerr << "\n===== 未知异常 =====" << std::endl;
		std::cerr << "捕获到未知类型的异常" << std::endl;
		system("pause");
		return -6001;
	}
}

