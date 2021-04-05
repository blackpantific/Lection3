#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
//#include <CL/opencl.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>


using namespace std;

void InformationAboutPlatforms();
cl_device_id InformationAboutDevice(cl_platform_id* platformID);

int main()
{
	cl_platform_id platformID;
	cl_device_id deviceID = InformationAboutDevice(&platformID);
	cl_int status;
	cl_context context;
	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM , (cl_context_properties)platformID, 0};

	context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &status);
	
	
	cl_command_queue queue;
	queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &status);

//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

	char* buf = NULL;
	ifstream in("Program.txt", ios::binary);
	const size_t size = in.seekg(0, ios::end).tellg();
	if (size == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[size + 1];
	in.read(buf, size);
	buf[size] = 0;
	in.close();
	const char* buf_p = buf;
	
	cl_program program;
	program = clCreateProgramWithSource(context, 1, &buf_p, &size, &status);

	const char* parameters = "";
	status = clBuildProgram(program, 1, &deviceID, parameters, NULL, NULL);

	size_t param_value = 0;
	status = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &param_value);

	char* log = NULL;
	if(param_value!=0)
	{
		log = (char*)malloc(sizeof(char));
		status = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, param_value, log, NULL);
		printf("\n%s", log);
	}

	cl_kernel kernel;
	kernel = clCreateKernel(program, "sum", &status);//ошибка


	int a = 100;
	int b = 50;
	int c = 0;
	
	cl_mem arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(int),
		&a, 0, NULL, NULL);
	
	cl_mem arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(int),
		&b, 0, NULL, NULL);

	cl_mem arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(int),
		&c, 0, NULL, NULL);
	


	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);

	int imageHeight = 100;
	int imageWidth = 200;
	size_t globalSize[2] = { imageWidth, imageHeight };

	size_t one = 1;

	status = clEnqueueNDRangeKernel(queue, kernel, one, NULL, &one, NULL, 0,
		NULL, NULL);

	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(int), &c, 0, NULL, NULL);
	//clEnqueueReadBuffer сделать последний вызов синхронным


	
	//CL_INVALID_PROGRAM
	//CL_INVALID_VALUE

	
	/*
		//clCreateContext();
		//clCreateContextFromType()
		//clCreateCommandQueue();//создать очередь упрапвление
		//можем давать задани€ различным девайсам в каком-то пор€дке
		//очередь принадлежит девайсу
		
		 *разрешить профайлинг
		 *
		 *
		 * clCreateProgramWithSource() -дл€ ускорени€ процесса компил€ции(необходима проверка)
		 * но если помен€ть видеокарту(девайс), то ничего работать не будет
		 * clBuildProgram ""
		 * clGetProgramBuildInfo()
		 * clCreateKernel()
		 * clSetKernelArg()
		 * onCreateBuffer()
		 */
	


	return 0;
}



cl_device_id InformationAboutDevice(cl_platform_id *platformID)
{
	cl_uint platformCount;
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



	char* paramValue;
	cl_bool res = false;
	cl_uint  numberOfUnits = 0;

	size_t paramValueRet = 0;
	
	for(int i = 0; i<platformCount; i++)
	{
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numberOfDevices);
		//platforms[0] - обычно у мен€ это дискретна€ интел видеокарта, если 1, то интегрированна€
		devices = (cl_device_id*)malloc(numberOfDevices);
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numberOfDevices, devices, NULL);

		
		err = clGetDeviceInfo(devices[i], CL_DEVICE_HOST_UNIFIED_MEMORY, 0, NULL, &paramValueRet);
		paramValue = (char*)malloc(paramValueRet);
		err = clGetDeviceInfo(devices[i], CL_DEVICE_HOST_UNIFIED_MEMORY, paramValueRet, &res, NULL);
		if(res == false)
		{
			//очистка пам€ти
			//platformID = (cl_platform_id*)malloc(sizeof(cl_platform_id*));
			*platformID = platforms[i];
			return devices[i];
		}
	}
	



	//CL_DEVICE_MAX_COMPUTE_UNITS  у мен€ на видеокарте их 3
	//CL_DEVICE_HOST_UNIFIED_MEMORY
	//CL_DEVICE_BUILT_IN_KERNELS   не поддерживает
	
	

	/*error = clGetDeviceInfo(devices[0], CL_DEVICE_BUILT_IN_KERNELS, 0, NULL, &paramValueRet);
	paramValue = (char*)malloc(paramValueRet);
	error = clGetDeviceInfo(devices[0], CL_DEVICE_BUILT_IN_KERNELS, paramValueRet, paramValue, NULL);*/

	//printf("%s", paramValue);
	//printf("%s", res ? "true" : "false");
	return NULL;
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

	//почистить пам€ть

}

