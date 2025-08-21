#pragma once

#include "vec2f.h"

inline float randomFloat(float min, float max)
{
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

inline Vec2f randomVec2f(float min, float max)
{
    return
    {
        randomFloat(min, max),
        randomFloat(min, max)
    };
}

inline uchar4 randomColor()
{
    unsigned char R = rand() % 255;
    unsigned char G = rand() % 255;
    unsigned char B = rand() % 255;
    return make_uchar4(R, G, B, 255);
}

