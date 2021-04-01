#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <CL/opencl.h>
using namespace std;

void InformationAboutPlatforms();
void InformationAboutDevice();


int main()
{
	//clCreateContext();
	//clCreateCommandQueue();//создать очередь упрапвление
	//можем давать задания различным девайсам в каком-то порядке
	//очередь принадлежит девайсу
	/*
	 *разрешить профайлинг
	 *
	 *
	 * clCreateProgramWithSource() -для ускорения процесса компиляции(необходима проверка)
	 * но если поменять видеокарту(девайс), то ничего работать не будет
	 * clBuildProgram ""
	 * clGetProgramBuildInfo()
	 * clCreateKernel()
	 * clSetKernelArg()
	 * onCreateBuffer()
	 */


	InformationAboutDevice();
	



	return 0;
}



void InformationAboutDevice()
{
	cl_uint platformCount = 0;
	int err = clGetPlatformIDs(0, NULL, &platformCount);
	printf("Number of platforms - %i\n", platformCount);
	cl_platform_id* platforms = (cl_platform_id*)malloc(platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);


	cl_uint numberOfDevices;
	cl_device_id* devices;

	int er = CL_INVALID_PLATFORM;

	const char* attributeNames[5] = { "CPU", "GPU", "ACCELERATOR", "DEFAULT", "ALL" };
	const cl_platform_info attributeTypes[5] = {
												CL_DEVICE_TYPE_CPU,
												CL_DEVICE_TYPE_GPU,
												CL_DEVICE_TYPE_ACCELERATOR,
												CL_DEVICE_TYPE_DEFAULT,
												CL_DEVICE_TYPE_ALL };

	int error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 0, NULL, &numberOfDevices);
	//platforms[0] - обычно у меня это дискретная интел видеокарта, если 1, то интегрированная
	devices = (cl_device_id*)malloc(numberOfDevices);
	error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, numberOfDevices, devices, NULL);


	char* paramValue;
	cl_bool res = false;
	cl_uint  numberOfUnits= 0;

	size_t paramValueRet = 0;

	//CL_DEVICE_MAX_COMPUTE_UNITS  у меня на видеокарте их 3
	//CL_DEVICE_HOST_UNIFIED_MEMORY
	//CL_DEVICE_BUILT_IN_KERNELS   не поддерживает
	
	error = clGetDeviceInfo(devices[0], CL_DEVICE_HOST_UNIFIED_MEMORY, 0, NULL, &paramValueRet);
	paramValue = (char*)malloc(paramValueRet);
	error = clGetDeviceInfo(devices[0], CL_DEVICE_HOST_UNIFIED_MEMORY, paramValueRet, &res, NULL);

	/*error = clGetDeviceInfo(devices[0], CL_DEVICE_BUILT_IN_KERNELS, 0, NULL, &paramValueRet);
	paramValue = (char*)malloc(paramValueRet);
	error = clGetDeviceInfo(devices[0], CL_DEVICE_BUILT_IN_KERNELS, paramValueRet, paramValue, NULL);*/

	//printf("%s", paramValue);
	printf("%s", res ? "true" : "false");
}

void InformationAboutPlatforms()
{
	cl_uint platformCount = 0;
	int err = clGetPlatformIDs(0, NULL, &platformCount);
	printf("Number of platforms - %i\n", platformCount);
	cl_platform_id* id = (cl_platform_id*)malloc(platformCount);
	clGetPlatformIDs(platformCount, id, NULL);

	//printf("%u %u", id[0], id[1]);
	

	const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions" };
	const cl_platform_info attributeTypes[5] = {
												CL_PLATFORM_NAME,
												CL_PLATFORM_VENDOR,
												CL_PLATFORM_VERSION,
												CL_PLATFORM_PROFILE,
												CL_PLATFORM_EXTENSIONS };

	int attributeCount = sizeof(attributeNames) / sizeof(attributeNames[0]);// можно просто заменить на 5
	
	cl_platform_id* platforms = id;

	printf("\nAttribute Count = %d ", attributeCount);
	int i, j;
	for (i = 0; i < platformCount; i++) {


		printf("\nPlatform - %d\n ", i + 1);
		size_t infoSize;
		char* info;

		for (j = 0; j < attributeCount; j++) {

			// get platform attribute value size
			clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
			info = (char*)malloc(infoSize);

			// get platform attribute value
			clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

			printf("  %d.%d %-11s: %s\n", i + 1, j + 1, attributeNames[j], info);
		}
		printf("\n\n");

	}

	//почистить память

}

