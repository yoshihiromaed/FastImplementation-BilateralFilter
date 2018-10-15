#pragma once
#include <opencv2/opencv.hpp>

// convert a BGR color image into a skipped one channel data: ex BGRBGRBGR... -> BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),...
void splitBGRLineInterleaveAVX(cv::InputArray src, cv::OutputArray dest);
void splitBGRLineInterleaveSSE(cv::InputArray src, cv::OutputArray dest);

// convert a skipped one channel data into a BGR color image
void mergeLineInterleaveBGRAVX(cv::InputArray src, cv::OutputArray dest);

// convert a BGR color image into a skipped one channel data: ex BGRBGRBGR... -> BBBB...(256 bit), GGGG....(256 bit), RRRR....(256 bit), BBBB...(256 bit), GGGG....(256 bit), RRRR....(256 bit),...
void splitBGRUnitInterleaveAVX(cv::InputArray src, cv::OutputArray dest);

//convert a BGR color image into a continued one channel data: ex BGRBGRBGR... -> BBBB...(image size), GGGG....(image size), RRRR....(image size).
//colorconvert 
void cvtColorBGR2PLANE(cv::InputArray src, cv::OutputArray dest);
void cvtColorPLANE2BGR(cv::InputArray src, cv::OutputArray dest);

