#pragma once

# define IMATH_HALF_NO_LOOKUP_TABLE

#include "gdt/math/vec.h"
#include <windows.h>

#include "cuda_runtime.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace gdt;

class ImageProcessing {
private:
	std::string path;
	int width, height, channels;
	unsigned char* datac;
	float* dataf;
	public:
		ImageProcessing() {};
		~ImageProcessing() {
			if (dataf != nullptr)
			{
				dataf = nullptr;
			}

			if (datac != nullptr)
			{
				datac = nullptr;
			}
		};
		void loadImage(std::string path, int& width_, int& height_, int& channels_, unsigned char*& data_);
		void loadImage(std::string path, int& width_, int& height_, int& channels_, float*& data_);
		void saveImage(std::string path_, unsigned char* data_, int width_, int height_, int channels_, bool filpVertically);
		void saveImage(std::string path_, float* data_, int width_, int height_, int channels_);
		void saveImage(uint32_t h_pixels[], std::string path, int width, int height, int channels);
		void saveImage(uint8_t h_pixels[], std::string path, int width, int height, int channels);
		void saveImage(vec4f h_pixels[], std::string path, int width, int height, int channels);
		void saveAlphaImage(uint32_t h_pixels[], std::string path, int width, int height, int channels);
		void saveAlphaImage(uint8_t h_pixels[], std::string path, int width, int height, int channels);
};
