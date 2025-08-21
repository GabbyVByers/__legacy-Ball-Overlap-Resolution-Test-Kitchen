#pragma once

#include "cuda_runtime.h"
#include "iostream"

template<typename type>

struct SharedArray
{
	size_t size = 0;
	size_t capacity = 1;
	type* hostPtr = nullptr;
	type* devPtr = nullptr;

	SharedArray()
	{
		hostPtr = new type[capacity];
		cudaMalloc((void**)&devPtr, sizeof(type) * capacity);
	}

	void doubleCapacity()
	{
		type* newHostPtr = nullptr;
		type* newDevPtr = nullptr;

		newHostPtr = new type[capacity * 2];
		cudaMalloc((void**)&newDevPtr, sizeof(type) * capacity * 2);

		memcpy(newHostPtr, hostPtr, sizeof(type) * capacity);
		cudaMemcpy(newDevPtr, devPtr, sizeof(type) * capacity, cudaMemcpyDeviceToDevice);

		delete[] hostPtr;
		cudaFree(devPtr);

		hostPtr = newHostPtr;
		devPtr = newDevPtr;

		capacity = capacity * 2;
	}

	void remove(size_t index)
	{
		if (index >= size)
			return;

		if (index == size - 1)
		{
			size--;
			return;
		}

		for (size_t i = index + 1; i < size; i++)
		{
			hostPtr[i - 1] = hostPtr[i];
		}
		size--;
	}

	void add(type element)
	{
		if (size == capacity)
			doubleCapacity();

		hostPtr[size] = element;
		size++;
	}

	void clear()
	{
		delete[] hostPtr;
		cudaFree(devPtr);

		size = 0;
		capacity = 1;

		hostPtr = new type[capacity];
		cudaMalloc((void**)&devPtr, sizeof(type) * capacity);
	}

	void updateHostToDevice() { cudaMemcpy(devPtr, hostPtr, size * sizeof(type), cudaMemcpyHostToDevice); }
	void updateDeviceToHost() { cudaMemcpy(hostPtr, devPtr, size * sizeof(type), cudaMemcpyDeviceToHost); }
	void free() { delete[] hostPtr; cudaFree(devPtr); }
};

