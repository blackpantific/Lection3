#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>
//#include <CL/opencl.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "conio.h"
#include <vector>
#include <string>


using namespace std;

void InformationAboutPlatforms();
cl_device_id InformationAboutDevice(cl_platform_id* platformID, int numberOfDevice);
void lection3_computing_sum(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel);
void lection4_computing_sum_arrays(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel);
void lection4_multipl_matrix(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel,
	float*& matrix1, float*& matrix2, float*& resultMatrix, int* NKM);
void get_matrixs_from_file(string input_file_path, int NKM[], float*& matrix1, float*& matrix2, float*& resultMatrix);

int main()
{
	int numberOfDevice = 0;//by default
	string pathInputFile = "C:\\Users\\black\\Desktop\\matrix.txt";
	string pathOutputFile = "C:\\Users\\black\\Desktop\\matrixResult.txt";
	int numberOfRealization = 0;//by default

	int NKM[3] = { 0,0,0 };

	float* matrix1 = 0;
	float* matrix2 = 0;
	float* resultMatrix = 0;

	get_matrixs_from_file(pathInputFile, NKM, matrix1, matrix2, resultMatrix);//получаем данные по матрицам из файла


	cl_platform_id platformID;
	cl_device_id deviceID = InformationAboutDevice(&platformID, numberOfDevice);

	cl_int status;
	cl_context context;
	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM , (cl_context_properties)platformID, 0};
	size_t deviceNameSize;
	char* deviceName;

	//display chosen device name
	status = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
	deviceName = (char*)malloc(deviceNameSize);
	status = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, deviceNameSize, deviceName, NULL);
	printf("\nDevice name - %s\n", deviceName);

	context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &status);

	//получаем код программы кернела из файла
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


	cl_command_queue queue;
	queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
	
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
	kernel = clCreateKernel(program, "matrix_multiplication", &status);//ошибка обнаруживается тут

	lection4_multipl_matrix(context, status, queue, kernel, matrix1, matrix2, resultMatrix, NKM);

	return 0;
}

void lection4_multipl_matrix(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel,
	float*& matrix1, float*& matrix2, float*& resultMatrix, int* NKM)
{

	/*for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			auto sum = 0;
			for (int k = 0; k < 2; k++)
			{
				sum += a[i * 2 + k] * b[k * 3 + j];
			}
			c[i * 3 + j] = sum;
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			printf("\nc[%d][%d] = %d ", i, j, c[i*3 + j]);

		}
	}*/

	for (size_t i = 0; i < 9; i++)
	{
		printf("\nc[%d] = %f ", i, resultMatrix[i]);
	}

	//float a[6] = { 1.0, 2, 3, 4, 5, 6 }; //(3;2)
	//float b[6] = { 1, 2, 3, 4, 5, 6 };//(2;3)
	//float* c = (float*)malloc(sizeof(float) * 9);

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;
	auto resultMatrixCapacity = matrix1Rows * matrix2Columns;

	cl_mem arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix1ElementsCount, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(float) * matrix1ElementsCount,
		&matrix1, 0, NULL, NULL);

	cl_mem arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix2ElementsCount, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(float) * matrix2ElementsCount,
		&matrix2, 0, NULL, NULL);

	cl_mem arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * resultMatrixCapacity, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(float) * resultMatrixCapacity,
		&resultMatrix, 0, NULL, NULL);

	//int wA = 2;
	//int wB = 3;

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);
	status |= clSetKernelArg(kernel, 3, sizeof(int), &matrix1Columns);
	status |= clSetKernelArg(kernel, 4, sizeof(int), &matrix2Columns);

	size_t dimentions = 2;
	size_t global_work_size[2];
	global_work_size[0] = matrix1Rows;
	global_work_size[1] = matrix2Columns;

	cl_event ourEvent = 0;

	status = clEnqueueNDRangeKernel(queue, kernel, dimentions, NULL, global_work_size, NULL, 0,
		NULL, &ourEvent);

	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(int) * resultMatrixCapacity, resultMatrix, 0, NULL, NULL);//самый последний ReadBuffer должен быть синхронным(CL_TRUE)

	cl_ulong gstart, gend;
	double gpuTime;

	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);

	gpuTime = (double)(gend - gstart) / 1000000000.0;



	for (int i = 0; i < matrix1Rows; i++)
	{
		for (int j = 0; j < matrix2Columns; j++)
		{
			printf("\nc[%d][%d] = %f ", i, j, resultMatrix[i * matrix1Rows + j]);

		}
	}
}

void get_matrixs_from_file(string input_file_path, int NKM[], float*& matrix1, float*& matrix2, float*& resultMatrix) {

	char* bufIterator = NULL;
	char* buf = NULL;

	ifstream in("C:\\Users\\black\\Desktop\\matrix.txt", ios::binary);
	int size = in.seekg(0, ios::end).tellg();
	if (size == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[size + 1];
	in.read(buf, size);
	buf[size] = 0;
	bufIterator = buf;
	in.close();
	string tempString = "";

	while (true) {


		if (*bufIterator == ' ' || *bufIterator == 13) {


			if (NKM[0] == 0) {
				NKM[0] = stoi(tempString);
				tempString = "";
			}
			else
				if (NKM[1] == 0) {
					NKM[1] = stoi(tempString);
					tempString = "";
				}
				else
					if (NKM[2] == 0) {
						NKM[2] = stoi(tempString);
						tempString = "";
					}

			if (*bufIterator == ' ') {
				bufIterator++;
			}
			else {
				bufIterator++;
				bufIterator++;
				break;
			}

		}
		else {
			tempString += *bufIterator;
			bufIterator++;
		}

	}

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;

	matrix1 = (float*)calloc(matrix1ElementsCount, sizeof(float));
	matrix2 = (float*)calloc(matrix2ElementsCount, sizeof(float));

	int i = 0;
	while (i != matrix1ElementsCount) {

		if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
		{
			tempString += *bufIterator;
			bufIterator++;

		}
		else
		{
			if (tempString == "")
			{
				throw "Wrong number exception";
			}
			matrix1[i] = stod(tempString);
			i++;
			bufIterator++;
			tempString = "";
		}
		if ((int)*bufIterator == 10)
		{
			bufIterator++;
		}

	}


	for (size_t i = 0; i < matrix1ElementsCount; i++)
	{
		cout << endl;
		printf("matrix1[%i] = %f", i, matrix1[i]);
	}

	cout << endl;

	i = 0;
	while (i != matrix2ElementsCount) {

		if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
		{
			tempString += *bufIterator;
			bufIterator++;

		}
		else
		{
			if (tempString == "")
			{
				throw "Wrong number exception";
			}
			matrix2[i] = stod(tempString);
			i++;
			bufIterator++;
			tempString = "";
		}
		if ((int)*bufIterator == 10)
		{
			bufIterator++;
		}

	}


	for (size_t i = 0; i < matrix2ElementsCount; i++)
	{
		cout << endl;
		printf("matrix2[%i] = %f", i, matrix2[i]);
	}

	int resultMatrixCapacity = matrix1Rows * matrix2Columns;

	resultMatrix = (float*)calloc(resultMatrixCapacity, sizeof(float));
}

void lection4_computing_sum_arrays(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel)
{
	int a[5] = {1, 2, 3, 4, 5};
	int b[5] = {6, 7, 8, 9, 10};
	//int c[5] = {0, 0, 0, 0, 0};
	int* c =(int*) malloc(sizeof(int) * 5);

	cl_mem arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*5, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(int)*5,
		&a, 0, NULL, NULL);

	cl_mem arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*5, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(int)*5,
		&b, 0, NULL, NULL);

	cl_mem arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*5, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(int)*5,
		&c, 0, NULL, NULL);



	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);

	//int imageHeight = 100;
	//int imageWidth = 200;
	//size_t globalSize[2] = { imageWidth, imageHeight };

	size_t one = 1;
	size_t global_work_size = 5;

	status = clEnqueueNDRangeKernel(queue, kernel, one, NULL, &global_work_size, NULL, 0,
		NULL, NULL);

	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(int)*5, c, 0, NULL, NULL);

	for(int i = 0; i<5; i++)
	{
		printf("\n%d", c[i]);
	}
}

void lection3_computing_sum(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel)
{
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

	printf("\nResult: %d", c);
}

cl_device_id InformationAboutDevice(cl_platform_id *platformID, int numberOfDevice)
{
	cl_uint platformCount;
	int err = clGetPlatformIDs(0, NULL, &platformCount);//gets number of available platforms
	printf("\nNumber of platforms - %i\n", platformCount);
	cl_platform_id* platforms = (cl_platform_id*)malloc(platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);//gets platform ids


	cl_uint numberOfDevices;
	cl_device_id* devices;

	vector<cl_device_id> devicesDiscreteGPU;
	vector<cl_device_id> devicesIntegratedGPU;
	vector<cl_device_id> devicesCPU;
	vector<cl_device_id> allDevicesIDs;

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
		//поиск и сортировка GPU-устройств, поддерживающих OpenCL
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numberOfDevices);

		if (numberOfDevices != 0 || err == 0) {

			devices = (cl_device_id*)malloc(numberOfDevices);
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numberOfDevices, devices, NULL);

			for (size_t j = 0; j < numberOfDevices; j++)//проверка видеокарты дискретная она или интегрированная
			{
				err = clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, 0, NULL, &paramValueRet);
				//paramValue = (char*)malloc(paramValueRet);
				err = clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, paramValueRet, &res, NULL);
				if (res == false)
				{
					devicesDiscreteGPU.push_back(devices[j]);
				}
				else {
					devicesIntegratedGPU.push_back(devices[j]);
				}
			}
		}

		numberOfDevices = 0;

		//проверка наличия поддержки OpenCL у CPU
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &numberOfDevices);

		if (numberOfDevices != 0 || err == 0) {

			devices = (cl_device_id*)malloc(numberOfDevices);
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, numberOfDevices, devices, NULL);

			for (size_t j = 0; j < numberOfDevices; j++)
			{
				devicesCPU.push_back(devices[j]);
			}
		}
	}

	allDevicesIDs.insert(allDevicesIDs.end(), devicesDiscreteGPU.begin(), devicesDiscreteGPU.end());
	allDevicesIDs.insert(allDevicesIDs.end(), devicesIntegratedGPU.begin(), devicesIntegratedGPU.end());
	allDevicesIDs.insert(allDevicesIDs.end(), devicesCPU.begin(), devicesCPU.end());

	if (numberOfDevice > allDevicesIDs.size()) {
		auto id = allDevicesIDs[0];

		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, 0, NULL, &paramValueRet);
		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, paramValueRet, platformID, NULL);

		return id;
	}
	else {

		auto id = allDevicesIDs[numberOfDevice];

		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, 0, NULL, &paramValueRet);
		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, paramValueRet, platformID, NULL);

		return id;
	}
}

void InformationAboutPlatforms()
{
	cl_uint platformCount = 0;
	int err = clGetPlatformIDs(0, NULL, &platformCount);//get number of platforms
	printf("Number of platforms - %i\n", platformCount);
	cl_platform_id* id = (cl_platform_id*)malloc(platformCount);
	clGetPlatformIDs(platformCount, id, NULL);//get ids of platforms and put them into id array

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

