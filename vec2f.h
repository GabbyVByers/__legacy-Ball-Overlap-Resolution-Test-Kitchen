#pragma once

struct Vec2f
{
    float x = 0.0f;
    float y = 0.0f;

    __host__ __device__ Vec2f operator + (Vec2f& other)
    {
        return
        {
            x + other.x,
            y + other.y
        };
    }

    __host__ __device__ Vec2f operator - (Vec2f& other)
    {
        return
        {
            x - other.x,
            y - other.y
        };
    }

    __host__ __device__ Vec2f operator * (float value)
    {
        return
        {
            x * value,
            y * value
        };
    }

    __host__ __device__ Vec2f operator / (float value)
    {
        return
        {
            x / value,
            y / value
        };
    }
};

__host__ __device__ inline float length(Vec2f& vec)
{
    return sqrt((vec.x * vec.x) + (vec.y * vec.y));
}

__host__ __device__ inline float lengthSquared(Vec2f& vec)
{
    return (vec.x * vec.x) + (vec.y * vec.y);
}

__host__ __device__ inline float dot(Vec2f& a, Vec2f& b)
{
    return (a.x * b.x) + (a.y * b.y);
}

__host__ __device__ inline void normalize(Vec2f& vec)
{
    float len = length(vec);
    vec.x /= len;
    vec.y /= len;
}