// testToothSegmentation.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
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


// onnx ����
#include "..\header\DentalCbctSegAI_API.h"
#include "..\lib\onnxruntime\include\onnxruntime_cxx_api.h"
#pragma comment(lib, "..\\lib\\DentalCbctOnnxSegDLL.lib")

#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime.lib")
#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime_providers_shared.lib")
#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime_providers_cuda.lib")
//#pragma comment(lib,"..\\utility\\onnxruntime\\lib\\onnxruntime_providers_tensorrt.lib")


/*
// libtorch 231 ����
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

//CImg���ڶ���洢������
#define cimg_display_type 2
#include "..\lib\CImg\CImg.h"

using namespace std;
using namespace cimg_library;

// �źŴ�����
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

// SEH�쳣������
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

// �����ĺ�������ִ�п����׳�SEH�쳣�Ĵ���
int SafeInfer(AI_HANDLE AI_Hdl, AI_DataInfo* srcData, AI_INT& AIWorkStatus) {
    __try {
        AIWorkStatus = DentalCbctSegAI_Infer(AI_Hdl, srcData);
        return 0; // �ɹ�
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        DWORD exceptionCode = GetExceptionCode();
        std::cerr << "\n===== SEH Exception in DentalCbctSegAI_Infer =====" << std::endl;
        std::cerr << "Exception Code: 0x" << std::hex << exceptionCode << std::dec << std::endl;
        
        if (exceptionCode == 0xE06D7363) {
            std::cerr << "DLL�ڲ��׳���C++�쳣" << std::endl;
            // ʵ�ʵĴ�����ϢӦ����DLL�������־��
        }
        
        return -1; // ʧ��
    }
}

int main()
{	
	// ����locale��֧������
	//setlocale(LC_ALL, "chs");
	
	
	// ��װ�źŴ�����
	signal(SIGSEGV, SignalHandler);
	signal(SIGFPE, SignalHandler);
	signal(SIGILL, SignalHandler);
	signal(SIGABRT, SignalHandler);
	
	// ����SEH�쳣������
	SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
	
	try {
		std::cout << "����ʼ����..." << std::endl;
		
		//load raw volume data: ������ǰ���������ں󣻶��������ң��°����ϣ�ͷ������
		std::cout << "���ڼ���HDRͼ���ļ�..." << std::endl;
		CImg<short> inputCbctVolume;
		inputCbctVolume.load_analyze("..\\..\\..\\img\\Series_5_Acq_2_0000.hdr");
		std::cout << "HDR�ļ����سɹ�" << std::endl;

	float VoxelSpacing  = 1.0f; //unit: mm  0.3
	float VoxelSpacingX = 0.5810545086860657f; //unit: mm  0.3
	float VoxelSpacingY = 0.5810545086860657f; //unit: mm  0.3
	float VoxelSpacingZ = 1.0f; //unit: mm  0.3
	
	int Width0 = inputCbctVolume.width();
	int Height0 = inputCbctVolume.height();
	int Depth0 = inputCbctVolume.depth();

	//���slice��λ����ʾͼ��������Ӧ���Ϸ�
	//������ݷ�λ���ԣ���ͨ��rotate XY 90��180��-90�Ƚ��е���
	//inputCbctVolume.rotate(180); //90, -90, 180
	//inputCbctVolume -= 1024;

	//CImg<short> slice_z = inputCbctVolume.get_slice( Depth0 / 2);
	//slice_z.display("slice 120");
	//�ر���ʾ���ں󣬳����������

	//����������������Ϣ
	short* ptrCbctData = inputCbctVolume.data();
	AI_DataInfo *srcData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
	srcData->Width = Width0;
	srcData->Height = Height0;
	srcData->Depth = Depth0;
	srcData->VoxelSpacing = VoxelSpacing;
	srcData->VoxelSpacingX = VoxelSpacingX;
	srcData->VoxelSpacingY = VoxelSpacingY;
	srcData->VoxelSpacingZ = VoxelSpacingZ;
	srcData->ptr_Data = ptrCbctData; //CBCT������ָ��

	//��ʼ�����ݷָ���������Ϣ
	CImg<short> toothLabelMask = CImg<short>(Width0, Height0, Depth0, 1, 0);

	AI_DataInfo *toothSegData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
	toothSegData->Width = Width0;
	toothSegData->Height = Height0;
	toothSegData->Depth = Depth0;
	toothSegData->VoxelSpacing = VoxelSpacing;
	toothSegData->VoxelSpacingX = VoxelSpacingX;
	toothSegData->VoxelSpacingY = VoxelSpacingY;
	toothSegData->VoxelSpacingZ = VoxelSpacingZ;
	toothSegData->ptr_Data = toothLabelMask.data();//�ָ�label������ָ��


	auto start = std::chrono::steady_clock::now();

	//�������ݷָ�ģ��
	//��ʼ���ָ�ģ�Ͷ���
	std::cout << "\nInitializing segmentation model..." << std::endl;
	AI_HANDLE  AI_Hdl = DentalCbctSegAI_CreateObj();
	if (AI_Hdl == NULL) {
		std::cerr << "Error: Model initialization failed!" << std::endl;
		return DentalCbctSegAI_STATUS_HANDLE_NULL; //ģ�ͳ�ʼ��ʧ��
	}
	std::cout << "Model initialized successfully" << std::endl;

	std::cout << "Setting model path..." << std::endl;
	AI_INT status1 = DentalCbctSegAI_SetModelPath(AI_Hdl, const_cast<char*>("..\\..\\..\\model\\kneeseg_test.onnx"));
	std::cout << "SetModelPath����״̬: " << status1 << std::endl;

	// ����TileStepRatio
	float tileRatio = 0.5f;
	AI_INT status2 = DentalCbctSegAI_SetTileStepRatio(AI_Hdl, tileRatio);
	std::cout << "SetTileStepRatio(" << tileRatio << ")����״̬: " << status2 << std::endl;

	// ��ӡ����������Ϣ
	std::cout << "\n����������Ϣ:" << std::endl;
	std::cout << "  �ߴ�: " << srcData->Width << " x " << srcData->Height << " x " << srcData->Depth << std::endl;
	std::cout << "  ���ؼ��: X=" << srcData->VoxelSpacingX << ", Y=" << srcData->VoxelSpacingY << ", Z=" << srcData->VoxelSpacingZ << std::endl;
	std::cout << "  ����ָ��: " << (void*)srcData->ptr_Data << std::endl;

	// ����ģ������������ ONNX Runtime �쳣
	std::cout << "\n��ʼģ������..." << std::endl;
	
	AI_INT	AIWorkStatus = DentalCbctSegAI_STATUS_FAIED;
	int result = SafeInfer(AI_Hdl, srcData, AIWorkStatus);
	
	if (result != 0) {
		// SEH�쳣����
		// �ͷ���Դ
		if (AI_Hdl) DentalCbctSegAI_ReleseObj(AI_Hdl);
		if (srcData) free(srcData);
		if (toothSegData) free(toothSegData);
		
		system("pause");
		return -7001;
	}
	
	std::cout << "ģ��������ɣ�״̬��: " << AIWorkStatus << std::endl;
	

	//��ȡ���ݷָ���
	if (AIWorkStatus == DentalCbctSegAI_STATUS_SUCCESS)
		DentalCbctSegAI_GetResult(AI_Hdl, toothSegData);
	else
		return AIWorkStatus;

	//�ͷŶ���
	DentalCbctSegAI_ReleseObj(AI_Hdl);
	// ���ݷָ���̽���

	//������toothSegData˵����
	//����С��ҰCBCT��
    //totalToothNumber:�ָ����������
    //���ݱ��k=1,2,3,...,totalToothNumber������labelΪ3k��������labelΪ3k+1�����ڻ����ֲ����labelΪ3k+2
    //���ڴ���ҰCBCT��
    //upperToothNumber:��������
    //lower_tooth_number:��������
    //�����ݱ��k=1,2,3,...,upperToothNumber������labelΪ3k��������labelΪ3k+1�����ڻ����ֲ����labelΪ3k+2
    //�����ݱ��k=-1,-2,-3,...,-lowerToothNumber������labelΪ-3k��������labelΪ-3k-1�����ڻ����ֲ����labelΪ-3k-2
	

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	//CImg<short> mask_z = toothLabelMask.get_slice(Depth0 / 2);
	//mask_z.display("mask 205");


	//����ָ���
	//inputCbctVolume.save_analyze("inputCbctVolume.hdr");
	toothLabelMask.save_analyze("finalLabelMask.hdr");

		std::cout << "\n����ִ�гɹ����!" << std::endl;
		return AIWorkStatus;
		
	} catch (const CImgIOException& e) {
		std::cerr << "\n===== CImg IO�쳣 =====" << std::endl;
		std::cerr << "������Ϣ: " << e.what() << std::endl;
		system("pause");
		return -2001;
	} catch (const CImgException& e) {
		std::cerr << "\n===== CImg�쳣 =====" << std::endl;
		std::cerr << "������Ϣ: " << e.what() << std::endl;
		system("pause");
		return -2002;
	} catch (const Ort::Exception& e) {
		std::cerr << "\n===== ONNX Runtime�쳣 =====" << std::endl;
		std::cerr << "������Ϣ: " << e.what() << std::endl;
		system("pause");
		return -3001;
	} catch (const std::bad_alloc& e) {
		std::cerr << "\n===== �ڴ����ʧ�� =====" << std::endl;
		std::cerr << "������Ϣ: " << e.what() << std::endl;
		system("pause");
		return -4001;
	} catch (const std::exception& e) {
		std::cerr << "\n===== ��׼�쳣 =====" << std::endl;
		std::cerr << "�쳣����: " << typeid(e).name() << std::endl;
		std::cerr << "������Ϣ: " << e.what() << std::endl;
		system("pause");
		return -5001;
	} catch (...) {
		std::cerr << "\n===== δ֪�쳣 =====" << std::endl;
		std::cerr << "����δ֪���͵��쳣" << std::endl;
		system("pause");
		return -6001;
	}
}

