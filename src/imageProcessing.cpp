#include "imageProcessing.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Function to generate a unique filename if the file already exists
std::string generateUniqueFilename(const std::string& baseFilename) {
	std::string filename = baseFilename;
	std::string extension;

	// Split baseFilename into name and extension
	std::size_t dotPos = baseFilename.find_last_of(".");
	if (dotPos != std::string::npos) {
		extension = baseFilename.substr(dotPos);
		filename = baseFilename.substr(0, dotPos); // filename without extension
	}

	int counter = 1;
	std::string newFilename = filename + extension;

	// Loop until a unique filename is found
	while (std::filesystem::exists(newFilename)) {
		std::ostringstream oss;
		oss << filename << "_" << counter << extension;  // Add a counter to the filename
		newFilename = oss.str();
		counter++;
	}

	return newFilename;
}

void ImageProcessing::loadImage(std::string path, int& width_, int& height_, int& channels_, unsigned char*& data_) {
	datac = stbi_load(path.c_str(), &width_, &height_, &channels_, STBI_rgb_alpha);
	if (datac == NULL) {
		std::cout << "Error: Can't load image " << path << std::endl;
		exit(1);
	}
	data_ = datac;
}

void ImageProcessing::loadImage(std::string path, int& width_, int& height_, int& channels_, float*& data_) {
	dataf = stbi_loadf(path.c_str(), &width_, &height_, &channels_, 0);
	if (dataf == NULL) {
		std::cout << "Error: Can't load image " << path << std::endl;
		exit(1);
	}
	data_ = dataf;
}

void ImageProcessing::saveImage(std::string path_, unsigned char* data_, int width_, int height_, int channels_, bool filpVertically) {
	stbi_flip_vertically_on_write(filpVertically);
	stbi_write_png(path_.c_str(), width_, height_, channels_, data_, width_ * channels_);
}

void ImageProcessing::saveImage(std::string path_, float* data_, int width_, int height_, int channels_) {
	stbi_write_hdr(path_.c_str(), width_, height_, channels_, data_);
}

void ImageProcessing::saveImage(vec4f h_pixels[], std::string path, int width, int height, int nrChannels) {
	unsigned char* bitmap = new unsigned char[width * height * 3]; //RGB
	for (std::size_t i = 0; i < width * height; i++) {
		bitmap[3 * i] = static_cast<unsigned char>(255.0f * clamp(h_pixels[i].x, 0.0f, 1.0f));
		bitmap[3 * i + 1] = static_cast<unsigned char>(255.0f * clamp(h_pixels[i].y, 0.0f, 1.0f));
		bitmap[3 * i + 2] = static_cast<unsigned char>(255.0f * clamp(h_pixels[i].z, 0.0f, 1.0f));
	}

	int stride = nrChannels * width;
	stbi_flip_vertically_on_write(true);
	stbi_write_png(path.c_str(), width, height, nrChannels, bitmap, stride);
	delete[] bitmap;
}

void ImageProcessing::saveImage(uint8_t h_pixels[], std::string path, int width, int height, int nrChannels) {
	path = generateUniqueFilename(path);

	unsigned char* bitmap = new unsigned char[width * height * 3]; //RGB
	for (std::size_t i = 0; i < width * height; i++) {
		bitmap[3 * i] = h_pixels[i * 4 + 0];
		bitmap[3 * i + 1] = h_pixels[i * 4 + 1];
		bitmap[3 * i + 2] = h_pixels[i * 4 + 2];
	}

	int stride = nrChannels * width;
	stbi_flip_vertically_on_write(true);
	stbi_write_png(path.c_str(), width, height, 3, bitmap, stride);
	delete[] bitmap;

	//unsigned char* bitmap = new unsigned char[width * height * 3]; //RGB
	//std::ofstream file(path, std::ios::binary);
	//if (!file.is_open()) {
	//	std::cout << "Error: Can't save image " << path << std::endl;
	//	exit(1);
	//}
	//file.write(reinterpret_cast<const char*>(h_pixels), width * height * 3);
	//file.close();
}

void ImageProcessing::saveImage(uint32_t h_pixels[], std::string path, int width, int height, int nrChannels) {
	if (nrChannels == 4) {
		unsigned char* bitmap = new unsigned char[width * height * 4]; //RGB
		for (int i = 0; i < width * height; i++) {
			bitmap[4 * i] = (h_pixels[i] >> 0) & 0xFF;
			bitmap[4 * i + 1] = (h_pixels[i] >> 8) & 0xFF;
			bitmap[4 * i + 2] = (h_pixels[i] >> 16) & 0xFF;
			bitmap[4 * i + 3] = (h_pixels[i] >> 24) & 0xFF;
		}

		int stride = nrChannels * width;
		stbi_flip_vertically_on_write(true);
		stbi_write_png(path.c_str(), width, height, 4, bitmap, stride);
		delete[] bitmap;
	}
	else if (nrChannels == 3) {
		unsigned char* bitmap = new unsigned char[width * height * 3]; //RGB
		std::ofstream file(path, std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Error opening file: " << path << std::endl;
			return;
		}
		// Write the entire uint32_t array to the file
		file.write(reinterpret_cast<const char*>(h_pixels), sizeof(uint32_t) * width * height);
		file.close();

		//#pragma omp parallel for
		//for (int i = 0; i < width * height; i++) {
		//	bitmap[3 * i] = (h_pixels[i] >> 0) & 0xFF;
		//	bitmap[3 * i + 1] = (h_pixels[i] >> 8) & 0xFF;
		//	bitmap[3 * i + 2] = (h_pixels[i] >> 16) & 0xFF;
		//}
		//int stride = nrChannels * width;
		//stbi_flip_vertically_on_write(true);
		//stbi_write_png(path.c_str(), width, height, 3, bitmap, stride);
		//delete[] bitmap;
	}
}

void ImageProcessing::saveAlphaImage(uint32_t h_pixels[], std::string path, int width, int height, int nrChannels) {
	unsigned char* bitmap = new unsigned char[width * height * nrChannels]; //RGBA
	for (std::size_t i = 0; i < width * height; i++) {
		bitmap[i] = (h_pixels[i] >> 24);
	}
	int stride = nrChannels * width;
	stbi_flip_vertically_on_write(true);
	stbi_write_png(path.c_str(), width, height, 1, bitmap, stride);
	delete[] bitmap;
}

void ImageProcessing::saveAlphaImage(uint8_t h_pixels[], std::string path, int width, int height, int nrChannels) {
	unsigned char* bitmap = new unsigned char[width * height * nrChannels]; //RGBA
	for (std::size_t i = 0; i < width * height; i++) {
		bitmap[i] = h_pixels[i * 4 + 3];
	}
	int stride = nrChannels * width;
	stbi_flip_vertically_on_write(true);
	stbi_write_png(path.c_str(), width, height, 1, bitmap, stride);
	delete[] bitmap;
}