#pragma once

#include "sharedarray.h"
#include "random.h"
#include <tuple>

struct Ball
{
    uchar4 color = make_uchar4(255, 255, 255, 255);
    Vec2f currPos;
    Vec2f displacement;
    float radius = 0.0f;
    
    int numFriends = 0;
    bool isClipping_Ball = false;
    bool isClipping_Wall = false;
};

struct SimulationState
{
    // global
    int screenWidth = -1;
    int screenHeight = -1;
    float max_u = 0.0f;
    uchar4* pixels = nullptr;
    SharedArray<Ball> balls;

    //device
    unsigned int oddEven = 0;
};

inline void initSimulation(std::tuple<int, int> screenDim, SimulationState& simState)
{
    simState.screenWidth  = std::get<0>(screenDim);
    simState.screenHeight = std::get<1>(screenDim);
    simState.max_u = (simState.screenWidth / (float)simState.screenHeight);;

    int numBalls = 400;
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.color = make_uchar4(rand() % 255, rand() % 255, rand() % 255, 255);
        //ball.radius = randomFloat(0.02f, 0.04f);
        ball.radius = 0.04f;
        ball.currPos.x = randomFloat(-simState.max_u, simState.max_u);
        ball.currPos.y = randomFloat(-1.0f, 1.0f);
        simState.balls.add(ball);
    }
    simState.balls.updateHostToDevice();
}

